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
RECOGNITION_THRESHOLD = 1.1
STRICT_THRESHOLD = 0.85
YOLO_CONFIDENCE_THRESHOLD = 0.75

# --- Temporal Consistency Settings ---
CONFIRM_FRAMES = 1
REQUIRED_VOTES = 1
MAX_BOX_DISTANCE = 50

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
recog_queue = queue.Queue(maxsize=1) # Small buffer to drop old frames if AI is slow
state_lock = threading.Lock()

shared_latest_frame = None
shared_faces_state: Dict[int, Dict[str, Any]] = {} 

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

# --- 2. Camera Thread (Robust) ---
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
            # 1. Connection Logic
            if not self.connected or self.cap is None or not self.cap.isOpened():
                try:
                    if self.cap: self.cap.release()
                    print(f"[Camera] Connecting to stream...")
                    self.cap = cv2.VideoCapture(self.stream_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Minimize Latency
                    
                    if self.cap.isOpened():
                        self.connected = True
                        print("[Camera] Connected.")
                    else:
                        print("[Camera] Connection failed. Retrying in 2s...")
                        time.sleep(2)
                        continue
                except Exception as e:
                    print(f"[Camera] Error: {e}")
                    time.sleep(2)
                    continue

            # 2. Frame Retrieval (Implicit Frame Drop Handling)
            # We read as fast as allowed. If 'latest_frame' isn't consumed by main thread,
            # it gets overwritten here (effectively dropping old frames + clearing hardware buffer).
            try:
                success, frame = self.cap.read()
                if success:
                    with self.lock:
                        self.latest_frame = frame
                else:
                    print("[Camera] Stream interrupted.")
                    self.connected = False
            except Exception as e:
                print(f"[Camera] Read Error: {e}")
                self.connected = False
                
    def get_frame(self):
        with self.lock:
            return self.latest_frame if self.connected else None

    def stop(self):
        self.stopped = True
        self.join()
        if self.cap: self.cap.release()

# --- 3. Detection Thread ---
class DetectionWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

    def run(self):
        print("[Thread] Detection Worker Started")
        global shared_latest_frame, model_yolo
        while self.running:
            # Get data from Main Thread (via Shared State)
            img = None
            with state_lock:
                if shared_latest_frame is not None:
                    img = shared_latest_frame.copy()
            
            if img is None:
                time.sleep(0.01)
                continue

            if model_yolo:
                try:
                    # Run YOLO
                    results = model_yolo(img, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                    boxes = []
                    for box in results[0].boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        boxes.append((x1,y1,x2,y2))
                    
                    if boxes:
                        # Push to Recognition
                        try:
                            recog_queue.put_nowait((img, boxes, time.time()))
                        except queue.Full: pass # Drop if overloaded
                except: pass
            
            time.sleep(0.001)

# --- 4. Recognition Thread ---
class RecognitionWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True
        self.history = {} 

    def run(self):
        print("[Thread] Recognition Worker Started")
        global shared_faces_state, mtcnn, resnet, faiss_index, known_names, known_ids
        
        while self.running:
            try:
                frame, boxes, ts = recog_queue.get(timeout=1)
                updates = {}
                
                for box in boxes:
                    # Match & ID Logic
                    bx_id = self._match(box)
                    if bx_id is None: bx_id = int(ts*10000) + sum(box)
                    
                    x1,y1,x2,y2 = box
                    crop = frame[y1:y2, x1:x2]
                    name = "Unknown"; sid = None

                    if crop.size > 0:
                        emb = self._get_emb(crop)
                        if emb is not None and faiss_index is not None:
                            D, I = faiss_index.search(emb.reshape(1,-1), 1)
                            if I[0][0] != -1 and D[0][0] < RECOGNITION_THRESHOLD**2:
                                if D[0][0] < STRICT_THRESHOLD**2:
                                    name = known_names[I[0][0]]
                                    sid = known_ids[I[0][0]]

                    # History & Voting
                    if bx_id not in self.history: self.history[bx_id] = []
                    h = self.history[bx_id]
                    h.append((name, sid))
                    if len(h) > CONFIRM_FRAMES: h.pop(0)

                    # Stable determination
                    s_name, s_id = "Unknown", None
                    valid = [x for x in h if x[0] != "Unknown"]
                    if valid:
                         c = {}
                         for v in valid: c[v] = c.get(v, 0)+1
                         best = max(c, key=c.get)
                         if c[best] >= REQUIRED_VOTES: s_name, s_id = best
                    
                    updates[bx_id] = {'box': box, 'name': s_name, 'id': s_id, 'ts': time.time()}

                # Update Global State
                with state_lock:
                    curr_time = time.time()
                    clean = {k:v for k,v in shared_faces_state.items() if curr_time - v['ts'] < 0.5}
                    clean.update(updates)
                    shared_faces_state.clear(); shared_faces_state.update(clean)
                
                recog_queue.task_done()
            except queue.Empty: continue
            except: pass

    def _get_emb(self, img):
        if not mtcnn or not resnet: return None
        try:
             t = mtcnn(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
             if t is None: return None
             e = resnet(t.unsqueeze(0).to('cpu')).detach().cpu().numpy().flatten()
             return e[:512] if e.shape[0]==1024 else e
        except: return None

    def _match(self, box):
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        bid, mdist = None, MAX_BOX_DISTANCE
        with state_lock: cands = shared_faces_state.items()
        for i, d in cands:
             bx = d['box']
             bcx, bcy = (bx[0]+bx[2])/2, (bx[1]+bx[3])/2
             dist = np.sqrt((cx-bcx)**2 + (cy-bcy)**2)
             if dist < mdist: mdist = dist; bid = i
        return bid

# --- Utilities ---
def load_db():
    global faiss_index, known_names, known_ids
    if not os.path.exists(EMBEDDINGS_FILE): return False
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f: data = pickle.load(f)
        embs, nms, ids = [], [], []
        
        # Parse Dict or List format
        first = next(iter(data.values()))
        if isinstance(first, dict):
             for k, v in data.items():
                 for e in v.get('embeddings', []):
                     nms.append(v['name']); ids.append(k); embs.append(e)
        else:
             for k, v in data.items():
                 for e in v:
                     nms.append(k); ids.append(k); embs.append(e)

        if not embs: return False
        mat = np.vstack(embs).astype('float32')
        idx = faiss.IndexFlatL2(mat.shape[1])
        idx.add(mat)
        faiss_index = idx; known_names = nms; known_ids = ids
        print(f"[System] DB Loaded: {len(nms)} vectors")
        return True
    except: return False

def parse_time(s):
    try:
        parts = str(s).split(':')
        return dtime(int(parts[0]), int(parts[1]))
    except: return dtime(0,0)

def in_range(star, end, now):
    if star <= end: return star <= now < end
    else: return now >= star or now < end

# --- MAIN THREAD (Orchestrator) ---
def real_time_face_recognition():
    global model_yolo, mtcnn, resnet, shared_latest_frame
    
    # Init Config
    cfg = {'start_time': '06:00', 'lateness_time': '07:00', 'absence_time': '10:00'}
    if os.path.exists('config.json'):
         try: cfg.update(json.load(open('config.json')))
         except: pass
    t_start = parse_time(cfg['start_time'])
    t_late = parse_time(cfg['lateness_time'])
    t_abs = parse_time(cfg['absence_time'])

    # Init Models
    try:
        model_yolo = YOLO(YOLO_MODEL_PATH)
        mtcnn = MTCNN(keep_all=False, device='cpu')
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        load_db()
    except Exception as e:
        print(f"[Fatal] {e}"); return

    # Start Threads
    cam = CameraHandler(STREAM_URL)
    cam.start()
    DatabaseWorker().start()
    DetectionWorker().start()
    RecognitionWorker().start()
    
    print("\n--- 🚀 MAIN THREAD STARTED: ORCHESTRATING SYSTEM 🚀 ---")
    local_cache = set()
    abs_checked = False

    try:
        while True:
            # 1. ORCHESTRATION: Get Frame from Camera Thread
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue # Wait for camera
            
            # 2. ORCHESTRATION: Update Shared State for Detection Thread
            with state_lock:
                shared_latest_frame = frame
                # Snapshot current results
                to_draw = shared_faces_state.copy()

            # 3. LOGIC: Calculate Times & Status
            display = frame.copy()
            now = datetime.now().time()
            
            for tid, data in to_draw.items():
                x1,y1,x2,y2 = data['box']
                name = data['name']; sid = data['id']
                
                # Determine Status
                color = (0,0,255) # Red/Unknown
                status_txt = ""

                if name != "Unknown":
                    color = (0,255,0) # Green/Present
                    status = "present"
                    
                    if in_range(t_late, t_abs, now):
                        status = "late"; color = (0,255,255)
                    elif not in_range(t_start, t_late, now):
                        if now >= t_abs: status = "absent"; color = (0,0,255)
                        else: status = "early" # purely visual
                    
                    status_txt = f"[{status.upper()}]"
                    
                    # 4. ORCHESTRATION: Dispatch DB Task
                    if sid and (sid not in local_cache) and (status not in ['absent', 'early']):
                        db_queue.put(("mark_attendance", sid, status))
                        local_cache.add(sid)

                cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                cv2.putText(display, f"{name} {status_txt}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 5. LOGIC: Auto-Absent Check
            if now >= t_abs and not abs_checked:
                db_queue.put(("bulk_absent", None, None))
                abs_checked = True

            cv2.imshow("SmartScan Ultra", display)
            if cv2.waitKey(1) == ord('q'): break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
