import openpyxl
import os
import cv2
import time
import threading
import torch
import faiss
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, time as dtime, timedelta
import json
import queue # New: Thread-safe queue for inter-thread communication

# --- Configuration ---
YOLO_MODEL_PATH = 'detection/weights/yolov12n-best.pt'
EMBEDDINGS_FILE = 'known_embeddings.pkl'
STREAM_URL = 'rtsp://admin:1234qwer@@172.16.0.3:554/Streaming/Channels/102?tcp'
OUTPUT_VIDEO_PATH = 'output_recording.mp4' # New: Path for the recorded video
RECOGNITION_THRESHOLD = 1.0
YOLO_CONFIDENCE_THRESHOLD = 0.75

# --- Temporal Consistency Settings ---
CONFIRM_FRAMES = 1
REQUIRED_VOTES = 1
MAX_BOX_DISTANCE = 50

# --- Global Variables for FAISS ---
FAISS_INDEX = None
KNOWN_NAMES = []
# Workbook and sheet for attendance (optional)
workbook = None
sheet = None
# Runtime set of names already marked/recognized
recognized_names = set()


# --- 1. Model Initialization ---
try:
    model_yolo = YOLO(YOLO_MODEL_PATH)
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    model_yolo = None

try:
    mtcnn = MTCNN(keep_all=True, device='cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    print("MTCNN and InceptionResnetV1 models loaded successfully.")
except Exception as e:
    print(f"Error loading MTCNN/InceptionResnetV1: {e}")
    mtcnn = None
    resnet = None


# --- 2. Data Loading & Utility Functions (No changes needed here) ---

def load_known_embeddings() -> Tuple[Optional[object], List[str], bool]:
    """Loads known face embeddings and builds a FAISS index."""
    global FAISS_INDEX, KNOWN_NAMES

    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            raw_embeddings = pickle.load(f)

            if not raw_embeddings:
                print(f"Warning: {EMBEDDINGS_FILE} is empty.")
                return None, [], False

            all_names: List[str] = []
            all_embeddings: List[np.ndarray] = []

            for name, embeddings in raw_embeddings.items():
                for emb in embeddings:
                    all_names.append(name)
                    all_embeddings.append(emb)

            if not all_embeddings:
                print(f"Warning: No embeddings found after flattening.")
                return None, [], False

            embeddings_matrix = np.vstack(all_embeddings).astype('float32')
            D = embeddings_matrix.shape[1]
            
            index = faiss.IndexFlatL2(D)
            index.add(embeddings_matrix)
            
            print(f"Known embeddings loaded and FAISS index built successfully. ({len(raw_embeddings)} identities, {len(all_names)} total embeddings)")
            
            FAISS_INDEX = index
            KNOWN_NAMES = all_names
            
            return index, all_names, True
            
    except Exception as e:
        print(f"FATAL: Error loading or building FAISS index from {EMBEDDINGS_FILE}: {e}")
        return None, [], False

def get_face_embedding(face_image: np.ndarray, mtcnn_model: object, resnet_model: object) -> Optional[np.ndarray]:
    """Processes a cropped face image to generate a 512-dimensional embedding vector."""
    if mtcnn_model is None or resnet_model is None:
        return None
        
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn_model(face_rgb)
    
    if face_tensor is None:
        return None

    face_embedding = resnet_model(face_tensor).detach().cpu().numpy().flatten()
    
    expected_dim = 512
    current_dim = face_embedding.shape[0]
    
    if current_dim != expected_dim:
        if current_dim == 1024 and expected_dim == 512:
            face_embedding = face_embedding[:expected_dim]
        else:
            return None
            
    return face_embedding

# --- 3. Thread-Safe Utility Classes ---

# Camera Handler (Slightly modified to use a Queue)
class CameraHandler(threading.Thread):
    """Reads frames from the video stream and puts them into a queue."""
    def __init__(self, stream_url: str, frame_queue: queue.Queue):
        super().__init__()
        self.stream_url = stream_url
        self.frame_queue = frame_queue
        self.cap = cv2.VideoCapture(stream_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stopped = True
        self.frame_count = 0

        if not self.cap.isOpened():
            print(f"Error: Could not open stream {stream_url}. Check URL or camera status.")
        else:
            self.stopped = False
            print("CameraHandler started successfully.")

    def run(self):
        """Continuously reads the newest frame and puts it in the queue."""
        while not self.stopped:
            success, frame = self.cap.read()
            
            if success and frame is not None:
                # Put a copy of the frame into the queue along with a timestamp
                self.frame_queue.put((frame.copy(), datetime.now()))
                self.frame_count += 1
            else:
                # Wait a bit before retrying the read
                time.sleep(0.01)

    def stop(self):
        """Signals the thread to stop and releases the video capture."""
        self.stopped = True
        self.join()
        self.cap.release()
        print(f"CameraHandler stopped. Total frames read: {self.frame_count}")

# New: Video Recording Thread
class VideoWriterHandler(threading.Thread):
    """Takes frames from the queue and writes them to a video file."""
    def __init__(self, video_queue: queue.Queue, output_path: str, frame_size: Tuple[int, int], fps: float = 20.0):
        super().__init__()
        self.video_queue = video_queue
        self.output_path = output_path
        self.frame_size = frame_size
        self.fps = fps
        self.stopped = False
        
        # Define the codec and create VideoWriter object
        # Use MP4V for good cross-platform compatibility with MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, frame_size)
        print(f"VideoWriterHandler initialized: {output_path} at {fps} FPS.")

    def run(self):
        while not self.stopped:
            try:
                # Get a frame and its recognition data
                frame, recognition_data = self.video_queue.get(timeout=0.1)
                
                # Draw the recognition results (text/boxes) on the frame
                # This should be the frame AFTER recognition/drawing, coming from the Recognizer
                self.out.write(frame)
                self.video_queue.task_done()
                
            except queue.Empty:
                # This ensures the thread doesn't block indefinitely when stopping
                continue
            except Exception as e:
                print(f"Error in VideoWriterHandler loop: {e}")
                
        # Cleanup
        self.out.release()
        print("VideoWriterHandler stopped and video file closed.")

    def stop(self):
        self.stopped = True
        self.join()

# New: Excel Logging Thread
class ExcelLogger(threading.Thread):
    """Handles all thread-safe writing to the Excel file."""
    def __init__(self, excel_queue: queue.Queue):
        super().__init__()
        self.excel_queue = excel_queue
        self.stopped = False
        self.last_absence_check = datetime.now()
        self.absence_time_obj = dtime(hour=10, minute=0) # Default, updated in main
        self.recognized_names = recognized_names # Reference the global set

        # Initialize workbook/sheet here to keep file I/O contained
        global workbook, sheet
        try:
            if os.path.exists('DashBoard.xlsx'):
                workbook = openpyxl.load_workbook('DashBoard.xlsx')
                sheet = workbook.active
                print("Excel file 'DashBoard.xlsx' loaded.")
            else:
                print("Warning: DashBoard.xlsx not found. Excel logging disabled.")
        except Exception as e:
            print(f"Error initializing Excel: {e}. Logging disabled.")
            workbook = None
            sheet = None

    def write_to_excel(self, name, status):
        global workbook, sheet
        if sheet is None or name in ('none', None, 'Unknown', 'Unknown_Unstable'):
            return

        try:
            for row_idx, row in enumerate(sheet.iter_rows(min_row=2, max_col=1, values_only=True), start=2):
                if row[0] == name:
                    # Find the last empty cell in the row
                    col_idx = 2
                    while sheet.cell(row=row_idx, column=col_idx).value is not None:
                        col_idx += 1

                    # Write the status
                    sheet.cell(row=row_idx, column=col_idx).value = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {status}"
                    workbook.save('DashBoard.xlsx')
                    print(f"EXCEL: {name} recorded as {status}.")
                    self.recognized_names.add(name)
                    break
        except Exception as e:
            print(f"EXCEL Error writing {name} to Excel: {e}")

    def check_absence(self):
        global workbook, sheet
        if sheet is None:
            return

        total = 0
        marked = 0
        try:
            for row_idx, row in enumerate(sheet.iter_rows(min_row=2, max_col=1, values_only=True), start=2):
                name = row[0]
                total += 1
                if name not in self.recognized_names:
                    marked += 1
                    self.write_to_excel(name, "Absent")
            print(f"EXCEL: Absence recorded for {marked} out of {total} names.")
        except Exception as e:
            print(f"EXCEL Error during absence recording: {e}")

    def run(self):
        while not self.stopped:
            try:
                # Get a logging task
                task_type, data = self.excel_queue.get(timeout=0.5)
                
                if task_type == 'LOG':
                    name, status = data
                    self.write_to_excel(name, status)
                elif task_type == 'ABSENCE_CHECK':
                    self.check_absence()
                
                self.excel_queue.task_done()
            except queue.Empty:
                # Periodically check if the absence time has passed
                now = datetime.now()
                if not is_time_in_range(dtime(hour=0, minute=0), self.absence_time_obj, now.time()) and now > self.last_absence_check + timedelta(hours=1):
                    # Check absence once the time passes, and prevent re-check for 1 hour
                    self.check_absence()
                    self.last_absence_check = now
                continue
            except Exception as e:
                print(f"EXCEL Error in logger loop: {e}")
                
        print("ExcelLogger stopped.")

    def stop(self):
        self.stopped = True
        self.join()

# --- Time helpers (No change) ---

def parse_time(s: str) -> dtime:
    """Parse strings like 'HH:MM', 'H' or 'HH' into a datetime.time object."""
    try:
        s_val = str(s).strip()
        if ':' in s_val:
            parts = s_val.split(':')
            h = int(parts[0]) if parts[0] != '' else 0
            m = int(parts[1]) if len(parts) > 1 and parts[1] != '' else 0
        else:
            h = int(s_val)
            m = 0

        h = max(0, min(23, h))
        m = max(0, min(59, m))
        return dtime(hour=h, minute=m)
    except Exception:
        return dtime(hour=0, minute=0)

def is_time_in_range(start: dtime, end: dtime, now_t: dtime) -> bool:
    """Return True if now_t is in the half-open interval [start, end),
    correctly handling intervals that cross midnight."""
    if start <= end:
        return start <= now_t < end
    else:
        # Interval wraps past midnight (e.g., 22:00 -> 02:00)
        return now_t >= start or now_t < end

def find_nearest_tracked_face(current_box: Tuple[int, int, int, int], history: Dict[int, Dict[str, Any]]) -> Optional[int]:
    """Finds the ID of the closest tracked face from the previous frame."""
    cx_current = (current_box[0] + current_box[2]) / 2
    cy_current = (current_box[1] + current_box[3]) / 2
    
    min_dist = float('inf')
    matched_id = None
    
    for face_id, data in history.items():
        hx1, hy1, hx2, hy2 = data['box']
        cx_hist = (hx1 + hx2) / 2
        cy_hist = (hy1 + hy2) / 2
        
        distance = np.sqrt((cx_current - cx_hist)**2 + (cy_current - cy_hist)**2)
        
        if distance < min_dist and distance < MAX_BOX_DISTANCE:
            min_dist = distance
            matched_id = face_id
            
    return matched_id

def determine_stable_name(history_list: List[str]) -> str:
    """Determines the stable name based on the majority vote in the history."""
    if not history_list:
        return "Unknown_Unstable"
        
    counts = {}
    for name in history_list:
        counts[name] = counts.get(name, 0) + 1
        
    stable_name = max(counts, key=counts.get)
    
    if counts[stable_name] >= REQUIRED_VOTES:
        return stable_name
    else:
        return "Unknown_Unstable"


# New: Recognition Thread
class RecognitionHandler(threading.Thread):
    """Pulls frames, runs recognition, and pushes results to Video and Excel queues."""
    def __init__(self, frame_queue: queue.Queue, video_queue: queue.Queue, excel_queue: queue.Queue, models: Tuple[object, object, object, object]):
        super().__init__()
        self.frame_queue = frame_queue
        self.video_queue = video_queue
        self.excel_queue = excel_queue
        self.model_yolo, self.mtcnn, self.resnet, self.faiss_index = models
        self.stopped = False

        # Tracking state (local to this thread)
        self.face_history: Dict[int, Dict[str, Any]] = {}
        self.current_face_id: int = 0
        self.marked_names = set() # Already recorded names for logging

        # Load time boundaries
        self.start_time_obj, self.lateness_time_obj, self.absence_time_obj = self._load_times()
        print(f"RECOGNITION: Attendance window: start={self.start_time_obj}, lateness={self.lateness_time_obj}, absence={self.absence_time_obj}")
        
    def _load_times(self, config_path='config.json'):
        defaults = {'start_time': '06:00', 'lateness_time': '07:00', 'absence_time': '10:00'}
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    merged = defaults.copy()
                    merged.update({k: v for k, v in cfg.items() if k in merged})
            else:
                merged = defaults
        except Exception:
            merged = defaults

        return parse_time(merged.get('start_time')), parse_time(merged.get('lateness_time')), parse_time(merged.get('absence_time'))


    def run(self):
        while not self.stopped:
            try:
                # Get the latest frame and timestamp from the camera
                frame, timestamp = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Check if we are outside the attendance window and should stop recognition
            now_t = timestamp.time()
            if not is_time_in_range(self.start_time_obj, self.absence_time_obj, now_t):
                # Put the raw frame back for video recording, but stop processing
                self.video_queue.put((frame, {})) 
                self.frame_queue.task_done()
                time.sleep(1) # Wait a bit before checking again
                continue


            # 1. Use YOLO to detect faces in the current frame
            yolo_results = self.model_yolo(frame, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
            
            current_frame_detections: Dict[int, Dict[str, Any]] = {}

            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_box = (x1, y1, x2, y2)
                
                # --- Tracking and History Lookup ---
                matched_id = find_nearest_tracked_face(current_box, self.face_history)
                
                if matched_id is None:
                    matched_id = self.current_face_id
                    self.current_face_id += 1
                    history_list = []
                    seen_count = 0
                    analysis_runs = 0
                    last_recognition = None
                else:
                    tracked = self.face_history.get(matched_id, {})
                    history_list = tracked.get('history', [])
                    seen_count = tracked.get('seen_count', 0)
                    analysis_runs = tracked.get('analysis_runs', 0)
                    last_recognition = tracked.get('last_recognition', None)
                
                # Update temporary tracking info for the next frame
                current_frame_detections[matched_id] = {
                    'box': current_box,
                    'history': history_list,
                    'seen_count': seen_count + 1,
                    'analysis_runs': analysis_runs,
                    'last_recognition': last_recognition
                }
                
                recognition_result = "Unknown"
                used_threshold = RECOGNITION_THRESHOLD
                THRESHOLD_INCREMENT = 0.2
                
                # --- Analysis Strategy ---
                sc = current_frame_detections[matched_id]['seen_count']
                ar = current_frame_detections[matched_id]['analysis_runs']
                lr = current_frame_detections[matched_id]['last_recognition']
                
                analyze_now = False
                if sc < 3:
                    analyze_now = True # Warm-up: analyze every frame
                elif ar == 0:
                    analyze_now = True
                    used_threshold = RECOGNITION_THRESHOLD
                elif ar == 1:
                    analyze_now = True
                    used_threshold = RECOGNITION_THRESHOLD + THRESHOLD_INCREMENT
                
                face_crop = frame[y1:y2, x1:x2]

                if analyze_now:
                    embedding = get_face_embedding(face_crop, self.mtcnn, self.resnet)
                    if embedding is not None:
                        query_embedding = embedding.reshape(1, -1).astype('float32')
                        D, I = self.faiss_index.search(query_embedding, 1)
                        min_distance = D[0][0]
                        best_match_index = I[0][0]
                        
                        if best_match_index != -1 and min_distance < (used_threshold ** 2):
                            recognition_result = KNOWN_NAMES[best_match_index]
                        else:
                            recognition_result = "Unknown"
                    
                    current_frame_detections[matched_id]['analysis_runs'] = ar + 1
                    current_frame_detections[matched_id]['last_recognition'] = recognition_result
                else:
                    # Not analyzing this frame: reuse the last recognition result (if any)
                    recognition_result = lr if lr is not None else "Unknown"

                # Update the history
                hist = current_frame_detections[matched_id]['history']
                hist.append(recognition_result)
                if len(hist) > CONFIRM_FRAMES:
                    hist.pop(0)

                # 5. Determine the STABLE name
                stable_name = determine_stable_name(hist)
                
                # --- Drawing and Logging ---
                color = (0, 0, 255) # Default Red/Unknown
                display_name = f"ID:{matched_id} {recognition_result}"
                
                if stable_name != "Unknown_Unstable":
                    # Stable match found (Green box)
                    display_name = f"{stable_name}"
                    color = (0, 255, 0)

                    # Determine status and log to Excel queue
                    if stable_name not in self.marked_names:
                        status = None
                        if is_time_in_range(self.start_time_obj, self.lateness_time_obj, now_t):
                            status = 'Present'
                        elif is_time_in_range(self.lateness_time_obj, self.absence_time_obj, now_t):
                            status = 'Late'
                        
                        if status:
                            self.excel_queue.put(('LOG', (stable_name, status)))
                            self.marked_names.add(stable_name)
                            # Add to global set for absence check
                            recognized_names.add(stable_name)
                
                # Draw the bounding box and name on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, display_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Update the tracking history for the next frame
            self.face_history = current_frame_detections
            
            # Pass the processed frame to the Video Writer and Viewer
            self.video_queue.put((frame, current_frame_detections))
            self.frame_queue.task_done()

        print("RecognitionHandler stopped.")

    def stop(self):
        self.stopped = True
        self.join()


# --- 4. Main Execution Block ---

def real_time_face_recognition_threaded():
    global FAISS_INDEX, KNOWN_NAMES, workbook, sheet, recognized_names

    # 1. Load Embeddings and Models
    faiss_index, known_names, load_success = load_known_embeddings()
    if not load_success or faiss_index is None or model_yolo is None or mtcnn is None or resnet is None:
        print("Exiting: Critical component failed to initialize.")
        return

    # 2. Setup Queues
    frame_queue = queue.Queue(maxsize=5)       # Camera to Recognition
    video_queue = queue.Queue(maxsize=5)       # Recognition to VideoWriter/Viewer
    excel_queue = queue.Queue(maxsize=100)     # Recognition to ExcelLogger

    # 3. Initialize Threads
    camera = CameraHandler(STREAM_URL, frame_queue)
    excel_logger = ExcelLogger(excel_queue)
    recognizer = RecognitionHandler(frame_queue, video_queue, excel_queue, (model_yolo, mtcnn, resnet, faiss_index))

    # Determine frame size for VideoWriter (need one frame first)
    success, initial_frame = camera.cap.read()
    if not success or initial_frame is None:
        print("Exiting: Could not read initial frame to determine video size.")
        camera.stop()
        return
        
    H, W, _ = initial_frame.shape
    video_writer = VideoWriterHandler(video_queue, OUTPUT_VIDEO_PATH, (W, H))

    # 4. Start Threads
    try:
        camera.start()
        video_writer.start()
        excel_logger.start()
        recognizer.start()

        # 5. Main Loop for Viewer (Runs in the main thread)
        print("\n--- Starting Real-Time Recognition Viewer (Press 'q' to exit) ---")
        while True:
            try:
                # Get the frame from the video queue (already recognized and drawn)
                frame, _ = video_queue.get(timeout=0.01) 
                
                # Display the result
                cv2.imshow("Real-Time Face Recognition", frame)
                video_queue.task_done()
                
            except queue.Empty:
                # No frame in the queue, just continue loop
                pass 
            except Exception as e:
                print(f"Viewer Error: {e}")
                time.sleep(0.1)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 6. Cleanup and Stop Threads
        print("\nStopping all threads...")
        recognizer.stop()
        video_writer.stop()
        excel_logger.stop()
        camera.stop()
        
        # Wait for all queues to be empty
        frame_queue.join()
        video_queue.join()
        excel_queue.join()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        import faiss
    except ImportError:
        print("FATAL ERROR: FAISS is not installed. Please install it using: pip install faiss-cpu")
        exit()
        
    real_time_face_recognition_threaded()
    print("\nApplication terminated.")