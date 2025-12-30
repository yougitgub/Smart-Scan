import os
import cv2
import time
import threading
import queue
import torch
import faiss
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

# --- Global Shared Resources ---
# Queues
db_queue = queue.Queue()
recog_queue = queue.Queue(maxsize=2)  # Limit queue size to prevent latency buildup

# Shared State (Protected by Lock where necessary)
state_lock = threading.Lock()
shared_latest_frame = None
shared_faces_state: Dict[int, Dict[str, Any]] = {} 
# structure: { trk_id: {'box': (x1,y1,x2,y2), 'name': str, 'id': str, 'status': str, 'last_seen': timestamp} }

# Global Models (Initialized in main function)
model_yolo = None
mtcnn = None
resnet = None
faiss_index = None
known_names = []
known_ids = []

# --- 1. Database Worker Thread ---
class DatabaseWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True # Kill when main exits
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
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[DB Thread Error] {e}")

    def _mark_attendance(self, student_national_id, status_str):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            
            # Get internal ID
            cursor.execute("SELECT id FROM students WHERE national_id = %s LIMIT 1", (student_national_id,))
            student = cursor.fetchone()
            
            if student:
                internal_id = student['id']
                today = datetime.now().date()
                
                # Check if exists
                cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND date = %s LIMIT 1", (internal_id, today))
                if not cursor.fetchone():
                    insert_sql = "INSERT INTO attendance (student_id, date, status, created_at, updated_at) VALUES (%s, %s, %s, NOW(), NOW())"
                    cursor.execute(insert_sql, (internal_id, today, status_str))
                    conn.commit()
                    print(f"  [DB] Marked {status_str}: {student_national_id}")
            conn.close()
        except Exception as e:
            print(f"  [DB Error] {e}")

    def _bulk_absent(self):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            today = datetime.now().date()
            query = """
                INSERT INTO attendance (student_id, date, status, created_at, updated_at)
                SELECT id, %s, 'absent', NOW(), NOW()
                FROM students
                WHERE id NOT IN (SELECT student_id FROM attendance WHERE date = %s)
            """
            cursor.execute(query, (today, today))
            count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if count > 0:
                print(f"\n[SYSTEM] 🕒 Absence Time Reached! Marked {count} remaining students as ABSENT in database.\n")
            else:
                print(f"\n[SYSTEM] 🕒 Absence Time Reached! All students accounted for.\n")
        except Exception as e:
            print(f"  [DB Error] Bulk Absent: {e}")

# --- 2. Detection Worker Thread ---
class DetectionWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

    def run(self):
        print("[Thread] Detection Worker Started")
        global shared_latest_frame, model_yolo
        
        while self.running:
            # 1. Get latest frame
            # We access the shared frame directly (read-only access usually okay in Python for opencv images, 
            # but using lock for safety)
            frame_to_process = None
            with state_lock:
                 if shared_latest_frame is not None:
                     frame_to_process = shared_latest_frame.copy()
            
            if frame_to_process is None:
                time.sleep(0.01)
                continue

            # 2. Run YOLO
            if model_yolo:
                try:
                    results = model_yolo(frame_to_process, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                    boxes = []
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append((x1, y1, x2, y2))
                    
                    if boxes:
                        # 3. Send to Recognition Queue
                        # We send the frame copy along with boxes so recognition runs on the SAME image timestamp
                        try:
                            recog_queue.put_nowait((frame_to_process, boxes, time.time()))
                        except queue.Full:
                            # If recognition is lagging, drop this detection frame to stay real-time
                            pass 
                except Exception as e:
                    print(f"[Detection Error] {e}")
            
            # Small sleep to prevent CPU hogging if YOLO is super fast? 
            # YOLO usually takes 10-20ms, so natural rate limit exists.
            time.sleep(0.001)

# --- 3. Recognition Worker Thread ---
class RecognitionWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True
        self.local_history = {} # Track ID -> History List

    def run(self):
        print("[Thread] Recognition Worker Started")
        global shared_faces_state, mtcnn, resnet, faiss_index, known_names, known_ids
        
        while self.running:
            try:
                # 1. Get Job
                frame, boxes, timestamp = recog_queue.get(timeout=1)
                
                current_updates = {}
                
                # 2. Process Boxes
                for box in boxes:
                    x1, y1, x2, y2 = box
                    
                    # Track matching (Associate with previous IDs)
                    matched_id = self._match_to_history(box)
                    
                    # Generate unique ID if new
                    if matched_id is None:
                        # Simple unique ID generation
                        matched_id = int(timestamp * 10000) + (x1+y1)
                    
                    # Crop Face
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # Recognition Logic
                    name = "Unknown"
                    student_id = None
                    
                    if face_crop.size > 0:
                         emb = self._get_embedding(face_crop)
                         if emb is not None and faiss_index is not None:
                             # Search FAISS
                             search_emb = emb.reshape(1, -1).astype('float32')
                             D, I = faiss_index.search(search_emb, 1)
                             idx = I[0][0]
                             dist = D[0][0]
                             
                             if idx != -1 and dist < (RECOGNITION_THRESHOLD ** 2):
                                 if dist < (STRICT_THRESHOLD ** 2):
                                     name = known_names[idx]
                                     student_id = known_ids[idx]
                    
                    # Voting / Stability
                    # We store history in this thread
                    if matched_id not in self.local_history:
                        self.local_history[matched_id] = []
                    
                    hist = self.local_history[matched_id]
                    hist.append((name, student_id))
                    if len(hist) > CONFIRM_FRAMES: hist.pop(0)
                    
                    # Determine stable result
                    stable_name, stable_id = self._determine_stable(hist)
                    
                    current_updates[matched_id] = {
                        'box': box,
                        'name': stable_name,
                        'id': stable_id,
                        'last_seen': time.time()
                    }

                # 3. Update Global State
                with state_lock:
                    # Merge logic:
                    # - Remove very old faces from shared state
                    # - Update matched ones
                    now = time.time()
                    new_state = {}
                    
                    # Keep existing valid faces from shared state (that weren't updated in this frame but are still recent)
                    for trk_id, data in shared_faces_state.items():
                        if now - data['last_seen'] < 0.5: # Keep for 0.5s
                            new_state[trk_id] = data
                    
                    # Overwrite with new data
                    new_state.update(current_updates)
                    
                    shared_faces_state.clear()
                    shared_faces_state.update(new_state)
                
                recog_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Recog Error] {e}")

    def _get_embedding(self, face_img):
        if mtcnn is None or resnet is None: return None
        try:
             face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
             face_tensor = mtcnn(face_rgb)
             if face_tensor is None: return None
             emb = resnet(face_tensor.unsqueeze(0).to('cpu')).detach().cpu().numpy().flatten()
             if emb.shape[0] == 1024: emb = emb[:512]
             return emb
        except: return None

    def _match_to_history(self, box):
        # find closest box in *recent* shared state
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        best_id = None
        min_dist = MAX_BOX_DISTANCE
        
        # We look at local history or shared state? Shared state is easier as it persists between frames
        with state_lock:
            candidates = shared_faces_state.items()

        for trk_id, data in candidates:
            bx1, by1, bx2, by2 = data['box']
            bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
            dist = np.sqrt((cx-bcx)**2 + (cy-bcy)**2)
            if dist < min_dist:
                min_dist = dist
                best_id = trk_id
        return best_id

    def _determine_stable(self, history):
        if not history: return "Unknown", None
        valid = [x for x in history if x[0] != "Unknown"]
        if not valid: return "Unknown", None
        counts = {}
        for n, i in valid: counts[(n,i)] = counts.get((n,i), 0) + 1
        best = max(counts, key=counts.get)
        if counts[best] >= REQUIRED_VOTES: return best
        return "Unknown", None

# --- Camera Handler (Preserved as requested) ---
class CameraHandler(threading.Thread):
    def __init__(self, stream_url: str):
        super().__init__()
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(stream_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        self.frame = None
        self.success = False
        self.stopped = True
        self.lock = threading.Lock()
        
        if self.cap.isOpened():
            self.success, self.frame = self.cap.read()
            self.stopped = False

    def run(self):
        while not self.stopped:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.frame = frame
                    self.success = True
            else:
                self.success = False

    def read(self):
        with self.lock:
            return self.success, self.frame

    def stop(self):
        self.stopped = True
        self.join() 
        self.cap.release()

# --- Helper Logic ---
def load_known_embeddings() -> bool:
    global faiss_index, known_names, known_ids
    if not os.path.exists(EMBEDDINGS_FILE): return False
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            raw_data = pickle.load(f)
            
        all_embeddings = []
        names = []
        ids = []
        
        # Handle dict format vs list format
        if not raw_data: return False

        first = next(iter(raw_data.values()))
        if isinstance(first, dict): # New format
             for sid, rec in raw_data.items():
                 # Support multiple embeddings per student
                 embs = rec.get('embeddings', [])
                 if isinstance(embs, list):
                     for e in embs:
                         names.append(rec['name'])
                         ids.append(sid)
                         all_embeddings.append(e)
        else: # Old format
            for name, embs in raw_data.items():
                for e in embs:
                    names.append(name)
                    ids.append(name)
                    all_embeddings.append(e)

        if not all_embeddings: return False
        
        mat = np.vstack(all_embeddings).astype('float32')
        index = faiss.IndexFlatL2(mat.shape[1])
        index.add(mat)
        
        faiss_index = index
        known_names = names
        known_ids = ids
        print(f"[System] Database loaded: {len(names)} vectors.")
        return True
    except Exception as e:
        print(f"[Error] Loading DB: {e}")
        return False

def parse_time(s: str) -> dtime:
    try:
        s_val = str(s).strip()
        if ':' in s_val:
            parts = s_val.split(':')
            h = int(parts[0]) if parts[0] != '' else 0
            m = int(parts[1]) if len(parts) > 1 and parts[1] != '' else 0
        else:
            h = int(s_val); m = 0
        return dtime(hour=max(0,min(23,h)), minute=max(0,min(59,m)))
    except:
        return dtime(hour=0, minute=0)

def is_time_in_range(start: dtime, end: dtime, now_t: dtime) -> bool:
    if start <= end: return start <= now_t < end
    else: return now_t >= start or now_t < end

# --- Main Entry Point (Called by main.py) ---
def real_time_face_recognition():
    global model_yolo, mtcnn, resnet, shared_latest_frame
    
    # 1. Load Settings
    cfg = {'start_time': '06:00', 'lateness_time': '07:00', 'absence_time': '10:00'}
    if os.path.exists('config.json'):
        try:
            with open('config.json') as f: cfg.update(json.load(f))
        except: pass
        
    t_start = parse_time(cfg['start_time'])
    t_late = parse_time(cfg['lateness_time'])
    t_abs = parse_time(cfg['absence_time'])
    
    print(f"System Running. Rules: Start={t_start}, Late={t_late}, Absent={t_abs}")
    
    # 2. Init AI Models
    try:
        model_yolo = YOLO(YOLO_MODEL_PATH)
        mtcnn = MTCNN(keep_all=False, device='cpu')
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        if not load_known_embeddings():
            print("WARNING: No faces enrolled!")
    except Exception as e:
        print(f"FATAL ERROR: AI Init failed: {e}")
        return

    # 3. Start Threads
    # Camera
    cam = CameraHandler(STREAM_URL)
    cam.start()
    
    # Workers
    DatabaseWorker().start()
    DetectionWorker().start()
    RecognitionWorker().start()
    
    # 4. Main Display Loop
    print("--- 🚀 High-Performance Multi-Threaded System Started 🚀 ---")
    
    local_attendance_cache = set()
    checked_absence_time = False
    
    try:
        while True:
            # A. Update Frame from Camera Thread
            success, frame = cam.read()
            if not success or frame is None:
                time.sleep(0.01)
                continue
                
            # Update shared frame for detection thread
            with state_lock:
                shared_latest_frame = frame
                # Snapshot of faces to draw
                faces_to_draw = shared_faces_state.copy() 
            
            # B. Display Logic (Draw Boxes)
            display_img = frame.copy()
            now_time = datetime.now().time()
            
            for trk_id, data in faces_to_draw.items():
                x1, y1, x2, y2 = data['box']
                name = data['name']
                sid = data['id']
                
                # Default style (Unknown)
                color = (0, 0, 255)
                text = "Unknown"
                
                if name != "Unknown":
                    # Determine Status
                    color = (0, 255, 0)
                    status = "present"
                    
                    if is_time_in_range(t_late, t_abs, now_time):
                        status = "late"
                        color = (0, 255, 255)
                    elif not is_time_in_range(t_start, t_late, now_time): 
                        # Before start (too early) or after absent
                        if now_time >= t_abs: status = "absent"
                        else: status = "early" # purely visual?
                        if status == "absent": color = (0, 0, 255)
                    
                    text = f"{name} [{status.upper()}]"
                    
                    # C. Check Enrollment / Mark Attendance
                    # Only mark if valid ID and not absent and not already marked today
                    if sid and sid not in local_attendance_cache and status != 'absent' and status != 'early':
                        db_queue.put(("mark_attendance", sid, status))
                        local_attendance_cache.add(sid)
                
                cv2.rectangle(display_img, (x1,y1), (x2,y2), color, 2)
                cv2.putText(display_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # D. Absence Auto-Mark Check
            if now_time >= t_abs and not checked_absence_time:
                db_queue.put(("bulk_absent", None, None))
                checked_absence_time = True
            
            # E. Render
            cv2.imshow("SmartScan Ultra - MultiThreaded", display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        print("Shutting down...")
        cam.stop()
        cv2.destroyAllWindows()
        # Daemon threads will terminate automatically
