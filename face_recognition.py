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
import tkinter as tk
from tkinter import messagebox

YOLO_MODEL_PATH = 'detection/weights/yolov12n-face.pt'
EMBEDDINGS_FILE = 'known_embeddings.pkl'
STREAM_URL = 'rtsp://admin:1234qwer@@192.168.1.18:554/Streaming/Channels/102?tcp'

RECOGNITION_THRESHOLD = 0.90  
STRICT_THRESHOLD = 0.85       
YOLO_CONFIDENCE_THRESHOLD = 0.65

CONFIRM_FRAMES = 1
REQUIRED_VOTES = 1
MAX_BOX_DISTANCE = 100

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '', 
    'database': 'system'
}

db_queue = queue.Queue()
recog_queue = queue.Queue(maxsize=2)
state_lock = threading.Lock()
shared_detected_boxes = []
shared_detected_ts = 0

shared_identities = {}

shared_latest_frame = None

model_yolo = None
mtcnn = None
resnet = None
faiss_index = None
known_names = []
known_ids = []

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
                else:
                    print(f"  [DB] Skipped {s_id} (Already Marked)")
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
            if count > 0: print(f"\n[SYSTEM]  Time's up! Marked {count} absent records.\n")
        except: pass

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
        self.error_shown = False

    def run(self):
        print("[Thread] Camera Handler Started")
        while not self.stopped:
            if not self.connected or self.cap is None or not self.cap.isOpened():
                try:
                    if self.cap: self.cap.release()
                    print(f"[Camera] Connecting to {self.stream_url}...")
                    self.cap = cv2.VideoCapture(self.stream_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if self.cap.isOpened() and self.cap.grab():
                        self.connected = True
                        self.error_shown = False
                        print("[Camera] Connected.")
                    else:
                        if not self.error_shown:
                            self._show_error(f"Failed to connect to camera:\n{self.stream_url}\n\nPlease check the URL and network connection.")
                            self.error_shown = True
                        time.sleep(2); continue
                except Exception as e:
                    if not self.error_shown:
                        self._show_error(f"Camera Connection Error:\n{e}")
                        self.error_shown = True
                    time.sleep(2); continue

            try:
                success, frame = self.cap.read()
                if success:
                    with self.lock: self.latest_frame = frame
                else: 
                    self.connected = False
                    if not self.error_shown:
                        self._show_error("Camera connection lost.")
                        self.error_shown = True
            except: self.connected = False

    def _show_error(self, msg):
        def pop():
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showerror("SmartScan - Camera Error", msg)
            root.destroy()
        threading.Thread(target=pop, daemon=True).start()
    def get_frame(self):
        with self.lock: return self.latest_frame if self.connected else None

    def stop(self):
        self.stopped = True
        self.join()
        if self.cap: self.cap.release()

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
                        print(f"[YOLO] Found {len(boxes)} faces")

                    with state_lock:
                        shared_detected_boxes = boxes
                        shared_detected_ts = time.time()
                    if boxes:
                        try: recog_queue.put_nowait((img, boxes, time.time()))
                        except queue.Full: pass
                except: pass
            time.sleep(0.00001)

class RecognitionWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True; self.running = True; self.history = {} 

    def run(self):
        print("[Thread] Recognition Worker Started")
        global shared_identities, resnet, faiss_index, known_names, known_ids
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Recog] Using Device: {device}")
        if resnet: resnet.to(device)
        while self.running:
            try:
                frame, boxes, ts = recog_queue.get(timeout=1)
                confirmed = []
                now = time.time()
                self.history = {k:v for k,v in self.history.items() if isinstance(v, dict) and now - v.get('last_seen', 0) < 2.0}

                batch_crops = []
                batch_meta = []

                for box in boxes:
                    x1,y1,x2,y2 = box
                    h, w = frame.shape[:2]
                    x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    track_id = self._match_history(cx, cy)
                    if track_id is None:
                        track_id = now + (cx/10000.0)
                        self.history[track_id] = {'votes': [], 'last_seen': now, 'last_center': (cx, cy)}

                    if not isinstance(self.history[track_id], dict):
                         self.history[track_id] = {'votes': [], 'last_seen': now, 'last_center': (cx, cy)}

                    self.history[track_id]['last_seen'] = now
                    self.history[track_id]['last_center'] = (cx, cy)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        resized = cv2.resize(crop, (160, 160))
                        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        batch_crops.append(rgb)
                        batch_meta.append((track_id, box))
                    else:
                        self._process_vote(track_id, "Unknown", None, box, confirmed)

                embeddings = []
                if batch_crops and resnet:
                    try:
                        data = np.stack(batch_crops)
                        t = torch.from_numpy(data).permute(0, 3, 1, 2).float()
                        t = (t - 127.5) / 128.0
                        t = t.to(device)
                        with torch.no_grad():
                            embs_out = resnet(t).detach().cpu().numpy()
                            embeddings = embs_out
                    except Exception as e:
                        print(f"[Recog Batch Error] {e}")
                emb_idx = 0
                for idx, (track_id, box) in enumerate(batch_meta):
                    name = "Unknown"; sid = None
                    if emb_idx < len(embeddings):
                        emb = embeddings[emb_idx]
                        emb_idx += 1
                        if faiss_index is not None:
                            D, I = faiss_index.search(emb.reshape(1,-1), 1)
                            dist = D[0][0]
                            idx_db = I[0][0]
                            threshold = RECOGNITION_THRESHOLD**2
                            if idx_db != -1 and dist < threshold:
                                if dist < STRICT_THRESHOLD**2:
                                    name = known_names[idx_db]
                                    sid = known_ids[idx_db]
                                    print(f"    ✅ MATCH: {name} ({dist:.4f})")
                                else:
                                    print(f"    ⚠️  AMBIGUOUS: {known_names[idx_db]} ({dist:.4f})")

                    self._process_vote(track_id, name, sid, box, confirmed)
                with state_lock:
                    shared_identities.update({id(box): c for box, c in zip(boxes, confirmed)})
                    now = time.time()
                    clean_state = {}
                    for k, v in shared_identities.items():
                        if k == 'latest_list': continue
                        if isinstance(v, dict) and (now - v.get('ts', 0) < 1.0):
                            clean_state[k] = v
                    clean_state['latest_list'] = confirmed
                    shared_identities = clean_state 

                recog_queue.task_done()
            except queue.Empty: continue
            except Exception as e: print(f"Recog Worker Loop Error: {e}")

    def _process_vote(self, track_id, name, sid, box, confirmed_list):
        try:
            votes = self.history[track_id].get('votes', [])
            if not isinstance(votes, list): votes = []
            votes.append((name, sid))
            if len(votes) > CONFIRM_FRAMES: votes.pop(0)
            self.history[track_id]['votes'] = votes
            valid_votes = [v for v in votes if v[0] != "Unknown"]
            final_name = "Unknown"; final_id = None
            if len(valid_votes) >= REQUIRED_VOTES:
                from collections import Counter
                counts = Counter(valid_votes)
                if counts:
                    (best_name, best_id), count = counts.most_common(1)[0]
                    if count >= REQUIRED_VOTES:
                        final_name, final_id = best_name, best_id
            confirmed_list.append({'box': box, 'name': final_name, 'id': final_id, 'ts': time.time()})
        except:
             confirmed_list.append({'box': box, 'name': "Unknown", 'id': None, 'ts': time.time()})


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

def real_time_face_recognition(stop_event=None, output_callback=None):
    global model_yolo, mtcnn, resnet, shared_latest_frame

    def log(msg):
        if output_callback: output_callback(f"{msg}\n")
        else: print(msg)
    cfg={
        'start_time':'06:00','lateness_time':'07:00','absence_time':'10:00', 
        'enable_recording': True,
        'camera_url': STREAM_URL
    }
    if os.path.exists('config.json'):
         try: cfg.update(json.load(open('config.json')))
         except: pass
    t_start=parse_time(cfg['start_time']); t_late=parse_time(cfg['lateness_time']); t_abs=parse_time(cfg['absence_time'])

    try:
        model_yolo = YOLO(YOLO_MODEL_PATH)
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        load_db()
    except Exception as E: log(f"Init Error: {E}"); return

    cam = CameraHandler(cfg.get('camera_url', STREAM_URL))
    if stop_event and stop_event.is_set(): return
    cam.start()
    DatabaseWorker().start(); DetectionWorker().start(); RecognitionWorker().start()
    log("---  ULTRA-FAST RENDERING ACTIVE  ---")
    local_cache=set(); abs_checked=False

    rec_dir = "recordings"
    os.makedirs(rec_dir, exist_ok=True)
    video_writer = None
    is_recording = cfg.get('enable_recording', True) 

    try:
        while True:
            if stop_event and stop_event.is_set(): break
            frame = cam.get_frame()
            if frame is None: time.sleep(0.0001); continue
            with state_lock:
                shared_latest_frame = frame
                cur_boxes = shared_detected_boxes[:]
                cur_ts = shared_detected_ts
                cur_identities = [v for k,v in shared_identities.items() if k != 'latest_list']


            display = frame.copy()
            now = datetime.now().time()
            if time.time() - cur_ts > 0.5:
                cur_boxes = []

            for (x1,y1,x2,y2) in cur_boxes:
                cx, cy = (x1+x2)//2, (y1+y2)//2
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
                    if sid and (sid not in local_cache) and (status != 'early'):
                        db_queue.put(("mark_attendance", sid, status))
                        local_cache.add(sid)

                cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                cv2.putText(display, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if is_recording:
                if video_writer is None:
                    h, w = display.shape[:2]
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filepath = os.path.join(rec_dir, f"session_{timestamp}.avi")
                    video_writer = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'XVID'), 15.0, (w, h))
                    log(f"[Recording] Started: {filepath}")
                video_writer.write(display)

            if now >= t_abs:
                if not abs_checked:
                    db_queue.put(("bulk_absent", None, None))
                    abs_checked = True
                    if is_recording:
                        print("[Recording] Absence Time Reached. Stopping Video.")
                        is_recording = False
                        if video_writer: video_writer.release()
                        video_writer = None
                    print("[SYSTEM] Closing system in 3 seconds...")
                    time.sleep(3)
                    break

            cv2.imshow("SmartScan Ultra", display)
            if cv2.waitKey(1) == ord('q') or (stop_event and stop_event.is_set()): break
    finally:
        if video_writer: video_writer.release()
        cam.stop(); cv2.destroyAllWindows()
        log("--- System Stopped ---")