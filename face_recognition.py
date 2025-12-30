import os
import cv2
import time
import threading
import queue
import torch
import faiss
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, time as dtime, timedelta
import json
import mysql.connector

# --- Configuration ---
YOLO_MODEL_PATH = 'detection/weights/yolov12n-face.pt'
EMBEDDINGS_FILE = 'known_embeddings.pkl'
STREAM_URL = 'rtsp://admin:1234qwer@@192.168.1.18:554/Streaming/Channels/102?tcp'

# --- THRESHOLDS ---
RECOGNITION_THRESHOLD = 0.95  # Relaxed
STRICT_THRESHOLD = 1.0       # Relaxed
YOLO_CONFIDENCE_THRESHOLD = 0.70

# --- Temporal Consistency Settings ---
CONFIRM_FRAMES = 3
REQUIRED_VOTES = 1 # Instant match
MAX_BOX_DISTANCE = 100

# --- MySQL Config ---
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '', 
    'database': 'system'
}

# --- Shared Resources ---
db_queue = queue.Queue()
recog_queue = queue.Queue(maxsize=2) # Increased buffer slightly 
state_lock = threading.Lock()
# fast_state: Updated by Detection Thread (High FPS) -> [(x1,y1,x2,y2), ...]
shared_detected_boxes = []
shared_detected_ts = 0

# slow_state: Updated by Recognition Thread (Low FPS) -> { track_id: {'name': str, 'id': str} }
shared_identities = {}

shared_latest_frame = None

model_yolo = None
mtcnn = None
resnet = None
faiss_index = None
known_names = []
known_ids = []

# --- 1. Database Thread ---
class DatabaseWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

    def run(self):
        print("[Thread] Database Worker Started")
        while self.running:
            try:
                task = db_queue.get(timeout=1)
                task_type = task[0]
                
                if task_type == "mark_attendance":
                    _, s_id, status = task
                    self._mark_attendance(s_id, status)
                elif task_type == "bulk_absent":
                    self._bulk_absent()
                
                db_queue.task_done()
            except queue.Empty: continue
            except Exception as e: print(f"[DB Error] {e}")

    def _mark_attendance(self, s_id, status):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id FROM students WHERE national_id = %s LIMIT 1", (s_id,))
            student = cursor.fetchone()
            if student:
                iid = student['id']
                today = datetime.now().date()
                cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND date = %s LIMIT 1", (iid, today))
                if not cursor.fetchone():
                    cursor.execute("INSERT INTO attendance (student_id, date, status, created_at, updated_at) VALUES (%s, %s, %s, NOW(), NOW())", (iid, today, status))
                    conn.commit()
                    print(f"  [DB] Marked {status.upper()}: {s_id}")
            conn.close()
        except Exception as e: print(f"  [DB Error] {e}")

    def _bulk_absent(self):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            today = datetime.now().date()
            cursor.execute("""
                INSERT INTO attendance (student_id, date, status, created_at, updated_at)
                SELECT id, %s, 'absent', NOW(), NOW()
                FROM students WHERE id NOT IN (SELECT student_id FROM attendance WHERE date = %s)
            """, (today, today))
            count = cursor.rowcount
            conn.commit()
            conn.close()
            if count > 0: print(f"\n[SYSTEM] 🕒 Time's up! Marked {count} absent records.\n")
        except: pass

# --- 2. Camera Thread ---
class CameraHandler(threading.Thread):
    def __init__(self, stream_url: str):
        super().__init__()
        self.stream_url = stream_url
        self.daemon = True
        self.stopped = False
        self.connected = False
        self.latest_frame = None
        self.lock = threading.Lock()
        self.cap = None

    def run(self):
        print("[Thread] Camera Handler Started")
        while not self.stopped:
            if not self.connected or self.cap is None or not self.cap.isOpened():
                try:
                    if self.cap: self.cap.release()
                    print(f"[Camera] Connecting...")
                    self.cap = cv2.VideoCapture(self.stream_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if self.cap.isOpened():
                        self.connected = True
                        print("[Camera] Connected.")
                    else:
                        time.sleep(2); continue
                except: time.sleep(2); continue

            try:
                success, frame = self.cap.read()
                if success:
                    with self.lock: self.latest_frame = frame
                else: self.connected = False
            except: self.connected = False
                
    def get_frame(self):
        with self.lock: return self.latest_frame if self.connected else None

    def stop(self):
        self.stopped = True
        self.join()
        if self.cap: self.cap.release()

# --- 3. Detection Thread (Fast Path) ---
class DetectionWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True; self.running = True

    def run(self):
        print("[Thread] Detection Worker Started")
        global shared_latest_frame, model_yolo, shared_detected_boxes, shared_detected_ts
        while self.running:
            img = None
            with state_lock:
                if shared_latest_frame is not None: img = shared_latest_frame.copy()
            
            if img is None: time.sleep(0.01); continue

            if model_yolo:
                try:
                    results = model_yolo(img, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                    boxes = []
                    for box in results[0].boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        boxes.append((x1,y1,x2,y2))
                    
                    if boxes:
                        print(f"[YOLO] Found {len(boxes)} faces") # LOGGING

                    # FAST PATH UPDATE
                    with state_lock:
                        shared_detected_boxes = boxes
                        shared_detected_ts = time.time()
                    
                    if boxes:
                        # SLOW PATH HANDOFF
                        try: recog_queue.put_nowait((img, boxes, time.time()))
                        except queue.Full: pass
                except: pass
            
            time.sleep(0.001)

# --- 4. Recognition Thread (Slow Path) ---
class RecognitionWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True; self.running = True; self.history = {} 

    def run(self):
        print("[Thread] Recognition Worker Started")
        global shared_identities, mtcnn, resnet, faiss_index, known_names, known_ids
        
        while self.running:
            try:
                frame, boxes, ts = recog_queue.get(timeout=1)
                confirmed = []
                
                # Prune old history
                now = time.time()
                self.history = {k:v for k,v in self.history.items() if isinstance(v, dict) and now - v.get('last_seen', 0) < 2.0}

                for box in boxes:
                    x1,y1,x2,y2 = box
                    # Clamp
                    h, w = frame.shape[:2]
                    x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
                    
                    # 1. Coordinate Matching for History
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    track_id = self._match_history(cx, cy)
                    
                    if track_id is None:
                        track_id = now + (cx/10000.0)
                        self.history[track_id] = {'votes': [], 'last_seen': now, 'last_center': (cx, cy)}

                    # Safety Check
                    if not isinstance(self.history[track_id], dict):
                         self.history[track_id] = {'votes': [], 'last_seen': now, 'last_center': (cx, cy)}

                    self.history[track_id]['last_seen'] = now
                    self.history[track_id]['last_center'] = (cx, cy)

                    # 2. Recognition
                    name="Unknown"; sid=None
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        emb = self._get_emb(crop)
                        if emb is not None and faiss_index is not None:
                            D, I = faiss_index.search(emb.reshape(1,-1), 1)
                            dist = D[0][0]
                            idx = I[0][0]
                            threshold = RECOGNITION_THRESHOLD**2
                            
                            print(f"  > [Recog] Dist={dist:.4f} vs Thresh={threshold:.4f}")

                            if idx != -1 and dist < threshold:
                                if dist < STRICT_THRESHOLD**2:
                                    name = known_names[idx]
                                    sid = known_ids[idx]
                                    print(f"    ✅ RAW MATCH: {name}")
                                else:
                                    print(f"    ⚠️  AMBIGUOUS: {known_names[idx]}")

                    # 3. Voting / Temporal Consistency
                    try:
                        votes = self.history[track_id].get('votes', [])
                        if not isinstance(votes, list): votes = []
                        
                        votes.append((name, sid))
                        if len(votes) > CONFIRM_FRAMES: votes.pop(0)
                        self.history[track_id]['votes'] = votes # Ensure saved back

                        # Tally votes
                        valid_votes = [v for v in votes if v[0] != "Unknown"]
                        final_name = "Unknown"; final_id = None
                        
                        if len(valid_votes) >= REQUIRED_VOTES:
                            from collections import Counter
                            # valid_votes is list of tuples: [('Name', 'ID'), ...]
                            counts = Counter(valid_votes)
                            if counts:
                                (best_name, best_id), count = counts.most_common(1)[0]
                                if count >= REQUIRED_VOTES:
                                    final_name, final_id = best_name, best_id
                                    print(f"    🏆 CONFIRMED: {final_name}")
                        
                        confirmed.append({'box': box, 'name': final_name, 'id': final_id, 'ts': time.time()})
                    except Exception as e:
                        print(f"    [Voting Error] {e}")
                        confirmed.append({'box': box, 'name': "Unknown", 'id': None, 'ts': time.time()})
                
                with state_lock:
                    shared_identities.update({id(box): c for box, c in zip(boxes, confirmed)})
                    now = time.time()
                    
                    # Safe Pruning Logic (Avoids crashing on 'latest_list' which is a list, not dict)
                    clean_state = {}
                    for k, v in shared_identities.items():
                        if k == 'latest_list': continue # Skip, we update it below
                        if isinstance(v, dict) and (now - v.get('ts', 0) < 1.0):
                            clean_state[k] = v
                    
                    clean_state['latest_list'] = confirmed
                    shared_identities = clean_state 

                recog_queue.task_done()
            except queue.Empty: continue
            except Exception as e: print(f"Recog Worker Loop Error: {e}")

    def _match_history(self, cx, cy):
        best_id = None; min_dist = MAX_BOX_DISTANCE
        for tid, data in self.history.items():
            if 'last_center' in data:
                lcx, lcy = data['last_center']
                dist = np.sqrt((cx-lcx)**2 + (cy-lcy)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid
        
        return best_id

    def _get_emb(self, img):
        if not mtcnn or not resnet: return None
        try:
             face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
             t = mtcnn(face_rgb)
             if t is None:
                 resized = cv2.resize(face_rgb, (160, 160))
                 t = torch.from_numpy(resized).permute(2, 0, 1).float()
                 t = (t - 127.5) / 128.0
             if t is not None:
                 e = resnet(t.unsqueeze(0).to('cpu')).detach().cpu().numpy().flatten()
                 return e[:512] if e.shape[0]==1024 else e
             return None
        except: return None

# --- Load Utilities ---
def load_db():
    global faiss_index, known_names, known_ids
    if not os.path.exists(EMBEDDINGS_FILE): return False
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f: data = pickle.load(f)
        embs, nms, ids = [], [], []
        first = next(iter(data.values()))
        if isinstance(first, dict):
             for k, v in data.items():
                 for e in v.get('embeddings', []): nms.append(v['name']); ids.append(k); embs.append(e)
        else:
             for k, v in data.items():
                 for e in v: nms.append(k); ids.append(k); embs.append(e)
        if not embs: return False
        mat = np.vstack(embs).astype('float32')
        idx = faiss.IndexFlatL2(mat.shape[1]); idx.add(mat)
        faiss_index = idx; known_names = nms; known_ids = ids
        return True
    except: return False

def parse_time(s):
    try: parts=str(s).split(':'); return dtime(int(parts[0]), int(parts[1]))
    except: return dtime(0,0)

def in_range(s, e, n): return s<=n<e if s<=e else n>=s or n<e

# --- MAIN ORCHESTRATOR ---
def real_time_face_recognition():
    global model_yolo, mtcnn, resnet, shared_latest_frame
    
    cfg={'start_time':'06:00','lateness_time':'07:00','absence_time':'10:00'}
    if os.path.exists('config.json'):
         try: cfg.update(json.load(open('config.json')))
         except: pass
    t_start=parse_time(cfg['start_time']); t_late=parse_time(cfg['lateness_time']); t_abs=parse_time(cfg['absence_time'])

    try:
        model_yolo = YOLO(YOLO_MODEL_PATH)
        mtcnn = MTCNN(keep_all=False, device='cpu')
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        load_db()
    except Exception as E: print(f"Init Error: {E}"); return

    cam = CameraHandler(STREAM_URL); cam.start()
    DatabaseWorker().start(); DetectionWorker().start(); RecognitionWorker().start()
    print("--- 🚀 ULTRA-FAST RENDERING ACTIVE 🚀 ---")
    
    local_cache=set(); abs_checked=False

    # --- Recording Setup ---
    rec_dir = "recordings"
    os.makedirs(rec_dir, exist_ok=True)
    video_writer = None
    is_recording = True 

    try:
        while True:
            frame = cam.get_frame()
            if frame is None: time.sleep(0.01); continue
            
            with state_lock:
                shared_latest_frame = frame
                # GET FAST BOXES
                cur_boxes = shared_detected_boxes[:]
                cur_ts = shared_detected_ts
                # GET SLOW NAMES (Use Memory Dict, not just latest snapshot)
                cur_identities = [v for k,v in shared_identities.items() if k != 'latest_list']


            display = frame.copy()
            now = datetime.now().time()
            
            # Draw Logic: Match Fast Box to Closest Slow Identity
            # If Fast Detection is stale (>0.5s), clear it (Fixes Ghosting if detection thread dies)
            if time.time() - cur_ts > 0.5:
                cur_boxes = []

            for (x1,y1,x2,y2) in cur_boxes:
                cx, cy = (x1+x2)//2, (y1+y2)//2
                
                # Find Identity
                name = "Unknown"; sid = None
                min_dist = MAX_BOX_DISTANCE
                
                if cur_identities:
                    for ident in cur_identities:
                        ibox = ident['box']
                        icx, icy = (ibox[0]+ibox[2])//2, (ibox[1]+ibox[3])//2
                        dist = np.sqrt((cx-icx)**2 + (cy-icy)**2)
                        if dist < min_dist:
                            min_dist = dist
                            name = ident['name']
                            sid = ident['id']

                color = (0,0,255); txt = "Unknown"
                if name != "Unknown":
                    color = (0,255,0); status = "present"
                    if in_range(t_late, t_abs, now): status="late"; color=(0,255,255)
                    elif not in_range(t_start, t_late, now):
                        if now >= t_abs: status="absent"; color=(0,0,255)
                        else: status="early"
                    txt = f"{name} [{status.upper()}]"
                    
                    if sid and (sid not in local_cache) and (status not in ['absent', 'early']):
                        db_queue.put(("mark_attendance", sid, status))
                        local_cache.add(sid)

                cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                cv2.putText(display, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- Video Recording ---
            if is_recording:
                if video_writer is None:
                    h, w = display.shape[:2]
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filepath = os.path.join(rec_dir, f"session_{timestamp}.avi")
                    # Increased to 15.0 FPS as 8.0 was too slow (Slow Motion)
                    video_writer = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'XVID'), 15.0, (w, h))
                    print(f"[Recording] Started: {filepath}")
                
                video_writer.write(display)

            if now >= t_abs:
                if not abs_checked:
                    db_queue.put(("bulk_absent", None, None))
                    abs_checked = True
                
                # Stop recording at absence time
                if is_recording:
                    print("[Recording] Absence Time Reached. Stopping Video.")
                    is_recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None

            cv2.imshow("SmartScan Ultra", display)
            if cv2.waitKey(1) == ord('q'): break
    finally:
        if video_writer: video_writer.release()
        cam.stop(); cv2.destroyAllWindows()
