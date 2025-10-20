# face_recognition_threaded_fixed_v3.py
"""
CPU/GPU auto threaded face recognition (full updated):
- Camera thread -> Recognition thread -> VideoWriter thread + Excel thread
- Batch embeddings (MTCNN + InceptionResnetV1) with safe handling
- FAISS fallback to numpy
- Rectangle color by status: Present (green), Late (orange), Unknown (red)
- Display: NAME | STATUS | HH:MM:SS (first stable detection time)
- Overlay realtime FPS on frames (also recorded into video)
- VideoWriter writes at fixed OUTPUT_FPS so recorded video plays at normal speed
- Uses GPU (cuda) automatically if available; otherwise CPU
- Schedules absence check at absence_time and stops gracefully
"""

import os
import time
import sys
import cv2
import threading
import pickle
import numpy as np
import queue
import json
from datetime import datetime, timedelta, date, time as dtime
from typing import List, Tuple, Dict, Any, Optional

# device auto-detect
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

# optional faiss
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# Excel
import openpyxl

print(f"[INFO] Using device: {DEVICE}")

# ---------------------------
# CONFIG
# ---------------------------
YOLO_MODEL_PATH = 'detection/weights/yolov12n-best.pt'
EMBEDDINGS_FILE = 'known_embeddings.pkl'
STREAM_URL = 'rtsp://admin:1234qwer@@172.16.0.3:554/Streaming/Channels/102?tcp'  # change to your camera
OUTPUT_DIR = 'recordings'
OUTPUT_VIDEO_FILENAME = 'recording_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.mp4'
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_FILENAME)

# thresholds
RECOGNITION_THRESHOLD = 0.9   # Euclidean distance threshold (tunable)
YOLO_CONF = 0.45

# tracking / analysis tuning
MAX_BOX_DISTANCE = 60
CONFIRM_FRAMES = 2
REQUIRED_VOTES = 1
EMBEDDING_TTL_SECONDS = 6.0

# queue sizes
FRAME_Q_MAX = 4
VIDEO_Q_MAX = 128
EXCEL_Q_MAX = 256

# video writer FPS target for playback (recorded video normal speed)
OUTPUT_FPS = 25.0  # change to 25 or 30 as you like

# device constant exported
device = DEVICE

# global recognized set for absence detection
recognized_names = set()

# ---------------------------
# Globals (populated later)
# ---------------------------
FAISS_INDEX = None
KNOWN_NAMES: List[str] = []
KNOWN_EMBEDDINGS_MATRIX = None

# ---------------------------
# Model loading
# ---------------------------
print("[INFO] Loading models...")
try:
    model_yolo = YOLO(YOLO_MODEL_PATH)
    print("[INFO] YOLO loaded.")
except Exception as e:
    print(f"[WARN] Could not load YOLO: {e}")
    model_yolo = None

try:
    # MTCNN can run on GPU if device is cuda; same for resnet (resnet must be on device)
    mtcnn = MTCNN(keep_all=False, image_size=160, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print(f"[INFO] MTCNN and ResNet loaded on {device}.")
except Exception as e:
    print(f"[ERROR] Face models load error: {e}")
    mtcnn = None
    resnet = None

# ---------------------------
# Utilities & Times
# ---------------------------
def now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(w-1, int(x1))); x2 = max(0, min(w-1, int(x2)))
    y1 = max(0, min(h-1, int(y1))); y2 = max(0, min(h-1, int(y2)))
    return (x1, y1, x2, y2)

def parse_time(s: str) -> dtime:
    try:
        if s is None:
            return dtime(hour=0, minute=0)
        s_val = str(s).strip()
        if ':' in s_val:
            parts = s_val.split(':'); h = int(parts[0]) if parts[0] != '' else 0
            m = int(parts[1]) if len(parts) > 1 and parts[1] != '' else 0
        else:
            h = int(s_val); m = 0
        h = max(0, min(23, h)); m = max(0, min(59, m))
        return dtime(hour=h, minute=m)
    except Exception:
        return dtime(hour=0, minute=0)

def load_times(config_path='config.json'):
    defaults = {'start_time': '06:00', 'lateness_time': '07:00', 'absence_time': '10:00'}
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            merged = defaults.copy(); merged.update({k: v for k, v in cfg.items() if k in merged})
        else:
            merged = defaults
    except Exception:
        merged = defaults
    return parse_time(merged.get('start_time')), parse_time(merged.get('lateness_time')), parse_time(merged.get('absence_time'))

def is_time_in_range(start: dtime, end: dtime, now_t: dtime) -> bool:
    if start <= end:
        return start <= now_t < end
    else:
        return now_t >= start or now_t < end

# ---------------------------
# Embeddings & FAISS helpers
# ---------------------------
def load_known_embeddings(path=EMBEDDINGS_FILE):
    global FAISS_INDEX, KNOWN_NAMES, KNOWN_EMBEDDINGS_MATRIX
    if not os.path.exists(path):
        print(f"[WARN] Embeddings file not found: {path}")
        return False
    try:
        with open(path, 'rb') as f:
            raw = pickle.load(f)
        flat_names = []; flat_embs = []
        for name, embs in raw.items():
            for e in embs:
                flat_names.append(name)
                flat_embs.append(np.asarray(e, dtype='float32'))
        if not flat_embs:
            print("[WARN] embeddings empty"); return False
        mat = np.vstack(flat_embs).astype('float32')
        KNOWN_EMBEDDINGS_MATRIX = mat
        if FAISS_AVAILABLE:
            D = mat.shape[1]; idx = faiss.IndexFlatL2(D); idx.add(mat); FAISS_INDEX = idx
            print(f"[INFO] FAISS index built: {mat.shape[0]} vectors dim={D}")
        else:
            FAISS_INDEX = None
            print("[WARN] Faiss not available; using numpy brute-force")
        KNOWN_NAMES = flat_names
        return True
    except Exception as e:
        print(f"[ERROR] load_known_embeddings: {e}")
        return False

def search_embeddings_batch(query_embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if query_embs is None or len(query_embs) == 0:
        return np.array([]), np.array([])
    if FAISS_AVAILABLE and FAISS_INDEX is not None:
        D, I = FAISS_INDEX.search(query_embs, 1)
        return D, I
    else:
        A = KNOWN_EMBEDDINGS_MATRIX
        D_list = []; I_list = []
        for q in query_embs:
            dists = np.linalg.norm(A - q, axis=1)
            idxs = np.argsort(dists)[:1]
            D_list.append(dists[idxs]); I_list.append(idxs)
        return np.vstack(D_list), np.vstack(I_list)

# ---------------------------
# Batch embedding computation (safe)
# ---------------------------
def compute_embeddings_batch(crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
    if not crops:
        return []

    valid_pils = []
    idx_map = []
    fixed_size = (160, 160)  # حجم موحّد لكل الصور لتجنب مشاكل MTCNN

    for i, c in enumerate(crops):
        if c is None or not isinstance(c, np.ndarray) or c.size == 0:
            continue
        try:
            # توحيد الحجم قبل التحويل
            resized = cv2.resize(c, fixed_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            valid_pils.append(pil)
            idx_map.append(i)
        except Exception:
            continue

    if not valid_pils:
        return [None] * len(crops)

    try:
        with torch.no_grad():
            aligned = mtcnn(valid_pils)  # قد ترجع tensor أو list
            if aligned is None:
                return [None] * len(crops)

            # معالجة النوع list
            if isinstance(aligned, list):
                aligned = [t for t in aligned if t is not None]
                if not aligned:
                    return [None] * len(crops)
                aligned = torch.stack(aligned)

            # إرسال للـ device (GPU أو CPU)
            aligned = aligned.to(device)

            # استخراج الـ embeddings
            embs_t = resnet(aligned)
            embs = embs_t.detach().cpu().numpy()

    except Exception as e:
        print(f"[ERROR] compute_embeddings_batch: {e}")
        return [None] * len(crops)

    # تجهيز النتيجة بنفس ترتيب الصور الأصلية
    result = [None] * len(crops)
    for out_idx, emb in zip(idx_map, embs):
        e = np.asarray(emb, dtype='float32')
        # تأكيد إن الطول = 512
        if e.shape[0] != 512:
            if e.shape[0] > 512:
                e = e[:512]
            else:
                e = np.pad(e, (0, 512 - e.shape[0])).astype('float32')
        result[out_idx] = e

    return result

# ---------------------------
# Threads
# ---------------------------
class CameraThread(threading.Thread):
    def __init__(self, src: str, frame_q: queue.Queue):
        super().__init__(daemon=True)
        self.src = src; self.frame_q = frame_q
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # try to set higher read timeout for reliability
        self.running = False
        if not self.cap.isOpened():
            print(f"[ERROR] Camera open failed: {src}")
        else:
            self.running = True; print("[INFO] CameraThread initialized.")

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.02); continue
            ts = datetime.now()
            try:
                self.frame_q.put_nowait((frame.copy(), ts))
            except queue.Full:
                try:
                    _ = self.frame_q.get_nowait()
                    self.frame_q.put_nowait((frame.copy(), ts))
                except Exception:
                    pass

    def stop(self):
        self.running = False
        try: self.cap.release()
        except Exception: pass
        print("[INFO] CameraThread stopped.")

class VideoWriterThread(threading.Thread):
    def __init__(self, video_q: queue.Queue, filename: str, fps: float, frame_size: Tuple[int, int]):
        super().__init__(daemon=True)
        self.video_q = video_q
        self.filename = filename
        self.fps = fps
        self.frame_size = frame_size
        self.running = True
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(filename, self._fourcc, fps, frame_size)
        self._frame_count = 0
        self._start_time = None
        print(f"[INFO] VideoWriterThread: {filename} {fps} FPS, size={frame_size}")

    def run(self):
        self._start_time = time.time()
        last_log_time = self._start_time
        
        while self.running:
            try:
                # Get the processed frame (with text and rectangles) from queue
                # Use a short timeout to prevent blocking
                frame, _meta = self.video_q.get(timeout=0.1)
                
                if frame is not None:
                    # Ensure frame matches the expected size
                    if (frame.shape[1], frame.shape[0]) != self.frame_size:
                        frame = cv2.resize(frame, self.frame_size)
                    
                    # Write the processed frame immediately
                    self._writer.write(frame)
                    self._frame_count += 1
                    
                    # Log progress every 5 seconds
                    current_time = time.time()
                    if current_time - last_log_time >= 5.0:
                        elapsed = current_time - self._start_time
                        actual_fps = self._frame_count / elapsed
                        print(f"[INFO] Video: {self._frame_count} frames, {actual_fps:.1f} FPS")
                        last_log_time = current_time
                
                try: 
                    self.video_q.task_done()
                except Exception: 
                    pass
                    
            except queue.Empty:
                # If no frame available, the video will have lower FPS but won't stall
                continue
            except Exception as e:
                print(f"[ERROR] VideoWriterThread loop: {e}")
                continue
        
        # Final statistics
        total_time = time.time() - self._start_time
        actual_fps = self._frame_count / total_time if total_time > 0 else 0
        print(f"[INFO] Video recording complete:")
        print(f"[INFO]   Total frames: {self._frame_count}")
        print(f"[INFO]   Total time: {total_time:.2f}s")
        print(f"[INFO]   Actual FPS: {actual_fps:.2f}")
        print(f"[INFO]   Target FPS: {self.fps}")
        
        self._writer.release()
        print("[INFO] VideoWriterThread stopped.")

    def stop(self):
        self.running = False



class ExcelLoggerThread(threading.Thread):
    def __init__(self, excel_q: queue.Queue, path='DashBoard.xlsx'):
        super().__init__(daemon=True)
        self.excel_q = excel_q
        self.path = path
        self.running = True
        self._name_row = {}
        self._wb = None
        self._ws = None
        self.absence_recorded = False
        self._last_save_time = time.time()
        self._save_interval = 30
        self._shutdown_callback = None  # Callback to shutdown the entire system
        self._init_workbook()

    def set_shutdown_callback(self, callback):
        """Set a callback function to shutdown the entire system"""
        self._shutdown_callback = callback

    def _init_workbook(self):
        """Initialize and load the Excel workbook with enhanced error handling"""
        try:
            if os.path.exists(self.path):
                print(f"[INFO] Loading existing Excel file: {self.path}")
                self._wb = openpyxl.load_workbook(self.path)
                self._ws = self._wb.active
            else:
                print(f"[INFO] Creating new Excel file: {self.path}")
                self._wb = openpyxl.Workbook()
                self._ws = self._wb.active
                self._ws.title = 'Attendance Sheet'
                self._ws.cell(row=1, column=1, value='Students Names')
                self._wb.save(self.path)
                print("[INFO] New Excel file created with header 'Students Names'")
            
            self._build_name_mapping()
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize Excel workbook: {e}")
            self._wb = None
            self._ws = None

    def _build_name_mapping(self):
        """Build the name to row mapping with duplicate detection"""
        self._name_row.clear()
        duplicate_count = 0
        
        for r in range(2, self._ws.max_row + 1):
            name = self._ws.cell(row=r, column=1).value
            if name:
                name_upper = str(name).upper().strip()
                if name_upper in self._name_row:
                    print(f"[WARNING] Duplicate name found in Excel: '{name}' at rows {self._name_row[name_upper]} and {r}")
                    duplicate_count += 1
                self._name_row[name_upper] = r
        
        print(f"[INFO] ExcelLogger initialized: {len(self._name_row)} unique names cached")
        if duplicate_count > 0:
            print(f"[WARNING] Found {duplicate_count} duplicate names in Excel file")

    def _auto_save(self):
        """Periodically save the workbook to prevent data loss"""
        current_time = time.time()
        if current_time - self._last_save_time >= self._save_interval:
            try:
                if self._wb:
                    self._wb.save(self.path)
                    self._last_save_time = current_time
                    print("[INFO] Auto-saved Excel file")
            except Exception as e:
                print(f"[ERROR] Auto-save failed: {e}")

    def write_record(self, name: str, status: str):
        """Write a record to Excel with enhanced validation and logging"""
        if self._ws is None:
            print(f"[ERROR] Excel worksheet not initialized, cannot write record for: {name}")
            return
            
        if name in (None, 'Unknown', 'Unknown_Unstable'):
            return
        
        name_upper = str(name).upper().strip()
        
        if name_upper not in self._name_row:
            print(f"[INFO] Name Not Found in Excel: '{name}' - Skipping")
            return
        
        try:
            row = self._name_row[name_upper]
            col = 2
            
            max_columns = 100
            while (self._ws.cell(row=row, column=col).value is not None and 
                   col <= max_columns):
                col += 1
            
            if col > max_columns:
                print(f"[WARNING] Too many records for student: {name}, cannot add more")
                return
            
            timestamp = now_str()
            self._ws.cell(row=row, column=col).value = f"{timestamp} - {status}"
            print(f"[INFO] Logged to Excel: {name} - {status}")
            
            if status in ['Present', 'Late', 'Absent']:
                self._wb.save(self.path)
                self._last_save_time = time.time()
            else:
                self._auto_save()
                
        except Exception as e:
            print(f"[ERROR] Failed to write record for {name}: {e}")

    def process_absence(self):
        """Process absence records with detailed reporting and trigger shutdown"""
        if not KNOWN_NAMES:
            print("[WARNING] No known names available for absence processing")
            return
            
        unique_names = list(set(KNOWN_NAMES))
        absent_count = 0
        not_found_count = 0
        error_count = 0
        
        print(f"[INFO] Starting absence processing for {len(unique_names)} unique students...")
        
        for name in unique_names:
            try:
                if name not in recognized_names:
                    name_upper = str(name).upper().strip()
                    if name_upper in self._name_row:
                        self.write_record(name, 'Absent')
                        absent_count += 1
                    else:
                        print(f"[INFO] Student not in Excel: '{name}' - Skipping absence mark")
                        not_found_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to process absence for {name}: {e}")
                error_count += 1
        
        # Final report
        print(f"[INFO] === ABSENCE PROCESSING COMPLETE ===")
        print(f"[INFO] Marked Absent: {absent_count} students")
        print(f"[INFO] Not Found in Excel: {not_found_count} students") 
        print(f"[INFO] Errors: {error_count} students")
        print(f"[INFO] Total Processed: {len(unique_names)} students")
        
        # Force final save
        try:
            if self._wb:
                self._wb.save(self.path)
                print("[INFO] Final Excel file saved after absence processing")
        except Exception as e:
            print(f"[ERROR] Final save failed: {e}")
        
        self.absence_recorded = True
        
        # Trigger system shutdown after absence processing
        print("[INFO] Absence recording complete. Initiating system shutdown...")
        if self._shutdown_callback:
            self._shutdown_callback()
        else:
            print("[WARNING] No shutdown callback set, using fallback shutdown")
            self._fallback_shutdown()

    def _fallback_shutdown(self):
        """Fallback shutdown method if no callback is set"""
        print("[INFO] Performing fallback shutdown...")
        self.running = False
        # Signal main thread to exit by putting None in queues
        try:
            self.excel_q.put(None)
        except:
            pass

    def run(self):
        """Main thread loop with enhanced task processing"""
        print("[INFO] ExcelLoggerThread started and ready for tasks")
        
        while self.running:
            try:
                task = self.excel_q.get(timeout=0.5)
                if task is None:
                    print("[INFO] Received shutdown signal")
                    break
                    
                ttype, data = task
                
                if ttype == 'LOG':
                    name, status = data
                    self.write_record(name, status)
                    
                elif ttype == 'ABSENCE':
                    print("[INFO] Received ABSENCE task, starting processing...")
                    self.process_absence()
                    # Shutdown will be triggered by process_absence()
                    
                try: 
                    self.excel_q.task_done()
                except Exception as e: 
                    print(f"[WARNING] Failed to mark task done: {e}")
                    
            except queue.Empty:
                self._auto_save()
                continue
            except Exception as e:
                print(f"[ERROR] ExcelLoggerThread loop: {e}")
                continue
                
        print("[INFO] ExcelLoggerThread stopped gracefully")

    def stop(self):
        """Stop the thread gracefully with final save"""
        print("[INFO] Stopping ExcelLoggerThread...")
        self.running = False
        try:
            if self._wb:
                self._wb.save(self.path)
                print("[INFO] Final Excel save completed")
        except Exception as e:
            print(f"[ERROR] Final save during shutdown failed: {e}")

# ---------------------------
# Recognition thread with absence time check
# ---------------------------

class RecognitionThread(threading.Thread):
    def __init__(self, frame_q: queue.Queue, video_q: queue.Queue, excel_q: queue.Queue):
        super().__init__(daemon=True)
        self.frame_q = frame_q; self.video_q = video_q; self.excel_q = excel_q
        self.running = True
        self.tracks: Dict[int, Dict[str, Any]] = {}; self.next_id = 0
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=1)
        # load schedule boundaries
        self.start_time_obj, self.lateness_time_obj, self.absence_time_obj = load_times()

        # FPS tracking for overlay (realtime)
        self._fps_prev_time = time.time()
        self._fps_frame_count = 0
        self.realtime_fps = 0.0

        # NEW: Track if absence has been recorded
        self.absence_recorded = False

    def find_nearest(self, box):
        cx = (box[0] + box[2]) / 2.0; cy = (box[1] + box[3]) / 2.0
        best = None; best_d = float('inf')
        for tid, t in self.tracks.items():
            bx1, by1, bx2, by2 = t['box']
            hx = (bx1 + bx2) / 2.0; hy = (by1 + by2) / 2.0
            d = np.hypot(cx - hx, cy - hy)
            if d < best_d and d < MAX_BOX_DISTANCE:
                best_d = d; best = tid
        return best

    def stable_name(self, hist: List[str]) -> str:
        if not hist: return 'Unknown_Unstable'
        counts = {}
        for n in hist: counts[n] = counts.get(n, 0) + 1
        best = max(counts, key=counts.get)
        if counts[best] >= REQUIRED_VOTES: return best
        return 'Unknown_Unstable'

    def _update_fps(self):
        self._fps_frame_count += 1
        now = time.time()
        elapsed = now - self._fps_prev_time
        if elapsed >= 1.0:
            self.realtime_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_prev_time = now

    
    def run(self):
        global FAISS_INDEX, KNOWN_NAMES, recognized_names
        print("[INFO] RecognitionThread started.")
        last_save = time.time()
        
        while self.running:
            try:
                frame, ts = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                # NEW: Check absence time even when no frames
                continue

            # NEW: Check absence time on each frame

            # update FPS (frame was received)
            self._update_fps()

            H, W = frame.shape[:2]; nowt = ts.time()

            # Check if we're within attendance hours
            if not is_time_in_range(self.start_time_obj, self.absence_time_obj, nowt):
                # Outside attendance hours, just pass frame through without processing
                try:
                    self.video_q.put_nowait((frame, {'time': datetime.now()}))
                except queue.Full:
                    pass
                try: self.frame_q.task_done()
                except Exception: pass
                continue

            try:
                yres = model_yolo(frame, verbose=False, conf=YOLO_CONF)
            except Exception:
                yres = None

            detections = []
            if yres is not None and len(yres) > 0:
                for b in yres[0].boxes:
                    try:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    except Exception:
                        vals = b.xyxy[0].cpu().numpy(); x1, y1, x2, y2 = map(int, vals.tolist())
                    detections.append((x1, y1, x2, y2))

            analyze_crops = []; analyze_tids = []; new_frame_tracks = {}

            for box in detections:
                x1, y1, x2, y2 = clamp_box(box, W, H)
                tid = self.find_nearest((x1, y1, x2, y2))
                if tid is None:
                    tid = self.next_id; self.next_id += 1
                    self.tracks[tid] = {'box': (x1, y1, x2, y2), 'history': [], 'seen_count': 0,
                                        'analysis_runs': 0, 'last_recognition': None,
                                        'cached_embedding': None, 'cached_time': None, 'logged': False,
                                        'first_seen_time': None}
                t = self.tracks[tid]; t['box'] = (x1, y1, x2, y2); t['seen_count'] = t.get('seen_count', 0) + 1

                use_cached = False; cached_time = t.get('cached_time')
                if t.get('cached_embedding') is not None and cached_time:
                    if (datetime.now() - cached_time).total_seconds() <= EMBEDDING_TTL_SECONDS:
                        use_cached = True

                analyze_now = False
                sc = t['seen_count']; ar = t.get('analysis_runs', 0)
                if sc < 3:
                    analyze_now = True
                else:
                    if ar == 0: analyze_now = True
                    elif ar == 1: analyze_now = True
                    else: analyze_now = False

                if use_cached:
                    rec = t.get('last_recognition', 'Unknown')
                    t['history'] = (t['history'] + [rec])[-CONFIRM_FRAMES:]
                elif analyze_now:
                    crop = frame[y1:y2, x1:x2]
                    if crop is None or crop.size == 0:
                        t['history'] = (t['history'] + ['Unknown'])[-CONFIRM_FRAMES:]
                    else:
                        analyze_crops.append(crop); analyze_tids.append(tid)
                else:
                    rec = t.get('last_recognition', 'Unknown')
                    t['history'] = (t['history'] + [rec])[-CONFIRM_FRAMES:]

                t['last_time_seen'] = datetime.now()
                new_frame_tracks[tid] = t

            # batch embeddings
            if analyze_crops:
                future = self.pool.submit(compute_embeddings_batch, analyze_crops)
                try:
                    embeddings = future.result(timeout=2.0)
                except Exception as e:
                    print(f"[WARN] embedding batch failed/timeout: {e}")
                    embeddings = [None] * len(analyze_crops)

                valid_embs = []; valid_tids = []
                for i, emb in enumerate(embeddings):
                    tid = analyze_tids[i]
                    if emb is None:
                        new_frame_tracks[tid]['history'] = (new_frame_tracks[tid]['history'] + ['Unknown'])[-CONFIRM_FRAMES:]
                        new_frame_tracks[tid]['analysis_runs'] = new_frame_tracks[tid].get('analysis_runs', 0) + 1
                        new_frame_tracks[tid]['last_recognition'] = 'Unknown'
                        continue
                    new_frame_tracks[tid]['cached_embedding'] = emb
                    new_frame_tracks[tid]['cached_time'] = datetime.now()
                    new_frame_tracks[tid]['analysis_runs'] = new_frame_tracks[tid].get('analysis_runs', 0) + 1
                    valid_embs.append(emb); valid_tids.append(tid)

                if valid_embs:
                    qmat = np.vstack(valid_embs).astype('float32')
                    D, I = search_embeddings_batch(qmat)
                    for vi, tid in enumerate(valid_tids):
                        dist = float(D[vi][0]); idx = int(I[vi][0])
                        if dist < (RECOGNITION_THRESHOLD ** 2) and idx < len(KNOWN_NAMES):
                            name = KNOWN_NAMES[idx]
                        else:
                            name = 'Unknown'
                        new_frame_tracks[tid]['last_recognition'] = name
                        new_frame_tracks[tid]['history'] = (new_frame_tracks[tid]['history'] + [name])[-CONFIRM_FRAMES:]

          # draw boxes & prepare excel logging (color by status; display name/status/time)
                for tid, t in new_frame_tracks.items():
                    x1, y1, x2, y2 = t['box']
                    hist = t.get('history', [])
                    stable = self.stable_name(hist)

                    # default values
                    display = t.get('last_recognition', 'Unknown')
                    color = (0, 0, 255)
                    status = 'Unknown'
                    ts_text = ''

                    if stable != 'Unknown_Unstable':
                        display = stable
                        current_time = datetime.now().time()

                        # determine current status (Present / Late)
                        if is_time_in_range(self.start_time_obj, self.lateness_time_obj, current_time):
                            status = 'Present'
                            color = (0, 255, 0)
                        elif is_time_in_range(self.lateness_time_obj, self.absence_time_obj, current_time):
                            status = 'Late'
                            color = (0, 140, 255)

                        # --- First detection (store official status/time/color) ---
                        if not t.get('logged', False):
                            t['first_seen_time'] = datetime.now()
                            t['first_status'] = status
                            t['first_color'] = color
                            t['logged'] = True
                            try:
                                if stable not in recognized_names:
                                    self.excel_q.put_nowait(('LOG', (stable, status)))
                                    recognized_names.add(stable)
                            except queue.Full:
                                pass

                        # --- Prepare label texts ---
                        # first line: current detection
                        if display == 'Unknown':
                            status = 'Unknown'
                            color = (0, 0, 255)
                            label_line1 = f"{status}"
                        label_line1 = f"{display} - {status}"

                        # second line: official info from first detection
                        fst = t.get('first_seen_time')
                        if fst:
                            first_status = t.get('first_status', status)
                            first_color = t.get('first_color', color)
                            first_time_str = fst.strftime('%H:%M:%S')
                            label_line2 = f"First: {first_status} | {first_time_str}"
                        else:
                            first_color = color
                            label_line2 = ""
                    else:
                        label_line1 = f"{display} - Unknown"
                        label_line2 = ""
                        first_color = color

                    # --- Draw bounding box ---
                    cv2.rectangle(frame, (x1, y1), (x2, y2), first_color, 2)

                    # --- Draw texts (2 lines) ---
                    y_text_base = max(12, y1 - 25)
                    cv2.putText(frame, label_line1, (x1, y_text_base),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, first_color, 2)
                    if label_line2:
                        cv2.putText(frame, label_line2, (x1, y_text_base + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            # # overlay realtime FPS (top-left)
            # fps_text = f"FPS: {self.realtime_fps:.2f}"
            # cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            # update tracks
            self.tracks = new_frame_tracks

            # send frame to video writer (non-blocking) with timestamp meta
            try:
                # the frame already has overlays (name/status/time/fps)
                self.video_q.put_nowait((frame.copy(), {'time': datetime.now()}))
            except queue.Full:
                pass

            # done
            try: self.frame_q.task_done()
            except Exception: pass

            if time.time() - last_save > 300:
                last_save = time.time()

        print("[INFO] RecognitionThread stopped.")

    def stop(self):
        self.running = False
        try:
            self.pool.shutdown(wait=False)
        except Exception:
            pass

# ---------------------------
# Main function with absence time monitoring
# ---------------------------
def face_recognition():
    global FAISS_INDEX, KNOWN_NAMES, recognized_names
    global cam, vw, xl, rec, video_q, excel_q

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ok = load_known_embeddings()
    if not ok:
        print("[WARN] embeddings not loaded; recognition will be Unknown.")
    else:
        print(f"[INFO] {len(KNOWN_NAMES)} known identities loaded.")

    # queues
    frame_q = queue.Queue(maxsize=FRAME_Q_MAX)
    video_q = queue.Queue(maxsize=VIDEO_Q_MAX)
    excel_q = queue.Queue(maxsize=EXCEL_Q_MAX)

    # threads
    cam = CameraThread(STREAM_URL, frame_q)
    cam.start()

    # wait for first frame to get size
    t0 = time.time(); first_frame = None
    while True:
        try:
            first_frame, ts = frame_q.get(timeout=5.0); frame_q.task_done(); break
        except queue.Empty:
            if time.time() - t0 > 10:
                print("[ERROR] No camera frames, exiting."); cam.stop(); return
            continue

    H, W = first_frame.shape[:2]
    vw = VideoWriterThread(video_q, OUTPUT_VIDEO_PATH, OUTPUT_FPS, (W, H))
    vw.start()
    
    # Create ExcelLoggerThread
    xl = ExcelLoggerThread(excel_q)
    
    def shutdown_system():
        """Callback function to shutdown the entire system"""
        print("[INFO] === SYSTEM SHUTDOWN INITIATED ===")
        print("[INFO] Stopping all threads...")
        
        # Stop all threads in order
        if 'rec' in globals() and rec.is_alive():
            rec.stop()
        if 'vw' in globals() and vw.is_alive():
            vw.stop()
        if 'xl' in globals() and xl.is_alive():
            xl.stop()
        if 'cam' in globals() and cam.is_alive():
            cam.stop()
        
        # Give threads time to stop
        time.sleep(2)
        
        # Clear queues
        for q in [frame_q, video_q, excel_q]:
            try:
                while not q.empty():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except:
                        pass
            except:
                pass
        
        # Proper OpenCV cleanup - this is crucial
        print("[INFO] Closing OpenCV windows...")
        try:
            cv2.destroyAllWindows()
            # Additional OpenCV cleanup
            cv2.waitKey(1)  # Process any pending events
            time.sleep(1)   # Give time for windows to close
        except Exception as e:
            print(f"[WARNING] OpenCV cleanup warning: {e}")
        
        print("[INFO] === SYSTEM SHUTDOWN COMPLETE ===")
        print("[INFO] All threads stopped, program exiting.")
        
        # Use sys.exit() instead of os._exit() for cleaner shutdown
        import sys
        sys.exit(0)
    
    # Set the shutdown callback for Excel thread
    xl.set_shutdown_callback(shutdown_system)
    xl.start()
    
    rec = RecognitionThread(frame_q, video_q, excel_q)
    rec.start()

    print("[INFO] Viewer started. Press 'q' to quit.")
    
    # Load absence time for monitoring
    _, _, absence_time_obj = load_times()
    absence_triggered = False
    
    try:
        while True:
            try:
                frame, _meta = video_q.get(timeout=0.5)
                if frame is not None:
                    cv2.imshow("SmartScan (threaded)", frame)
                try: 
                    video_q.task_done()
                except Exception: 
                    pass
            except queue.Empty:
                pass
            
            # Main thread monitors absence time as backup
            current_time = datetime.now().time()
            if current_time >= absence_time_obj and not absence_triggered:
                print(f"[INFO] Main thread detected absence time {absence_time_obj}")
                try:
                    excel_q.put(('ABSENCE', None), timeout=5.0)
                    absence_triggered = True
                    print("[INFO] Absence task sent to Excel thread")
                except queue.Full:
                    print("[ERROR] Excel queue full, cannot send absence task")
                    shutdown_system()
                    break
                except Exception as e:
                    print(f"[ERROR] Main thread failed to send absence task: {e}")
                    shutdown_system()
                    break
            
            # Manual shutdown with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Manual shutdown requested by user")
                shutdown_system()
                break
                
            # Check if we should break the loop (when shutdown is initiated)
            if hasattr(xl, 'absence_recorded') and xl.absence_recorded:
                print("[INFO] Breaking main loop due to absence completion")
                break
                
    except Exception as e:
        print(f"[ERROR] Main loop error: {e}")
        shutdown_system()
    
    # Final cleanup after loop exits
    print("[INFO] Performing final cleanup...")
    try:
        # Ensure all threads are stopped
        if 'rec' in globals() and rec.is_alive():
            rec.stop()
        if 'vw' in globals() and vw.is_alive():
            vw.stop()
        if 'xl' in globals() and xl.is_alive():
            xl.stop()
        if 'cam' in globals() and cam.is_alive():
            cam.stop()
        
        # Wait for threads to finish
        time.sleep(2)
        
        # Final OpenCV cleanup
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Process pending events
        
    except Exception as e:
        print(f"[ERROR] Final cleanup error: {e}")
    
    print("[INFO] Face Recognition System Ended completely.")

if __name__ == "__main__":
    face_recognition()