# face_recognition_super.py
# ------------------------------------------------------------
# Super-threaded version of Smart Scan Face Recognition System
# Features:
# - 5 main threads (Camera, Detection, RecognitionManager, WorkerPool, ExcelWriter)
# - GPU auto-detection
# - YOLO + MTCNN + ResNet + FAISS
# - Time-based attendance logic
# - RTSP camera only
# - Real-time drawing and console logs
# ------------------------------------------------------------

import os, cv2, time, threading, queue, json, pickle, faiss, torch, openpyxl
import numpy as np
from datetime import datetime, timedelta, time as dtime
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
YOLO_MODEL_PATH = 'detection/weights/yolov12n-best.pt'
EMBEDDINGS_FILE = 'known_embeddings.pkl'
EXCEL_FILE = 'DashBoard.xlsx'
STREAM_URL = 'rtsp://admin:1234qwer@@172.16.0.3:554/Streaming/Channels/102?tcp'
RECOGNITION_THRESHOLD = 1.0
YOLO_CONFIDENCE_THRESHOLD = 0.75
CONFIRM_FRAMES = 1
REQUIRED_VOTES = 1
MAX_BOX_DISTANCE = 50

# ------------------------------------------------------------
# Device Auto Detection
# ------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[System] Using device: {device}")

# ------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------
model_yolo = YOLO(YOLO_MODEL_PATH)
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ------------------------------------------------------------
# Load Known Embeddings
# ------------------------------------------------------------
def load_embeddings():
    with open(EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    names, embs = [], []
    for name, vectors in data.items():
        for v in vectors:
            names.append(name)
            embs.append(v)
    matrix = np.vstack(embs).astype('float32')
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    print(f"[System] Loaded {len(names)} embeddings.")
    return index, names

faiss_index, known_names = load_embeddings()

# ------------------------------------------------------------
# Excel Setup
# ------------------------------------------------------------
try:
    workbook = openpyxl.load_workbook(EXCEL_FILE)
    sheet = workbook.active
    print("[Excel] Loaded workbook.")
except Exception as e:
    print(f"[Excel] Error loading workbook: {e}")
    workbook = None
    sheet = None

# ------------------------------------------------------------
# Time Parsing
# ------------------------------------------------------------
def parse_time(s: str) -> dtime:
    try:
        parts = s.split(':')
        return dtime(int(parts[0]), int(parts[1]))
    except:
        return dtime(0,0)

def is_time_in_range(start, end, now):
    if start <= end:
        return start <= now < end
    return now >= start or now < end

# ------------------------------------------------------------
# Queues
# ------------------------------------------------------------
frame_queue = queue.Queue(maxsize=2)
face_queue = queue.Queue(maxsize=20)
result_queue = queue.Queue(maxsize=50)
excel_queue = queue.Queue(maxsize=100)
recognized_names = set()

# ------------------------------------------------------------
# Threads
# ------------------------------------------------------------
class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(STREAM_URL)
        self.running = True
        print("[Camera] Started.")
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and not frame_queue.full():
                frame_queue.put(frame)
            time.sleep(0.01)
    def stop(self):
        self.running = False
        self.cap.release()

class DetectionThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
    def run(self):
        print("[Detection] Started.")
        while self.running:
            if frame_queue.empty():
                time.sleep(0.01)
                continue
            frame = frame_queue.get()
            results = model_yolo(frame, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if not face_queue.full():
                    face_queue.put((crop, (x1,y1,x2,y2), frame))
            print(f"[Detection] Detected {len(results[0].boxes)} faces.")
    def stop(self):
        self.running = False

class RecognitionWorker(threading.Thread):
    def __init__(self, id):
        super().__init__()
        self.id = id
        self.running = True
    def run(self):
        print(f"[RecognitionWorker-{self.id}] Started.")
        while self.running:
            if face_queue.empty():
                time.sleep(0.01)
                continue
            crop, box, frame = face_queue.get()
            try:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = mtcnn(rgb)
                if tensor is None: continue
                tensor = tensor.to(device)
                emb = resnet(tensor).detach().cpu().numpy().flatten().astype('float32')
                D,I = faiss_index.search(emb.reshape(1,-1),1)
                dist, idx = D[0][0], I[0][0]
                name = known_names[idx] if dist < RECOGNITION_THRESHOLD**2 else 'Unknown'
                result_queue.put((name, box, frame))
                print(f"[RecognitionWorker-{self.id}] {name} ({dist:.3f})")
            except Exception as e:
                print(f"[RecognitionWorker-{self.id}] Error: {e}")
    def stop(self):
        self.running = False

class RecognitionManager(threading.Thread):
    def __init__(self, num_workers=2):
        super().__init__()
        self.running = True
        self.workers = [RecognitionWorker(i) for i in range(num_workers)]
    def run(self):
        print("[RecognitionManager] Starting worker pool...")
        for w in self.workers: w.start()
        while self.running:
            if result_queue.empty():
                time.sleep(0.01)
                continue
            name, box, frame = result_queue.get()
            if name not in ['Unknown', None]:
                excel_queue.put((name, datetime.now()))
            x1,y1,x2,y2 = box
            color = (0,255,0) if name!='Unknown' else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow('SmartScan Super', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
    def stop(self):
        self.running = False
        for w in self.workers: w.stop()

class ExcelThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.buffer = []
    def run(self):
        print("[Excel] Writer started.")
        while self.running:
            if excel_queue.empty():
                time.sleep(0.5)
                continue
            name, ts = excel_queue.get()
            self.buffer.append((name, ts))
            if len(self.buffer) >= 5:
                self.flush()
        self.flush()
    def flush(self):
        if not workbook: return
        try:
            for name, ts in self.buffer:
                for row_idx, row in enumerate(sheet.iter_rows(min_row=2, max_col=1, values_only=True), start=2):
                    if row[0] == name:
                        col_idx = 2
                        while sheet.cell(row=row_idx, column=col_idx).value is not None:
                            col_idx += 1
                        sheet.cell(row=row_idx, column=col_idx).value = ts.strftime('%Y-%m-%d %H:%M:%S')
                        break
            workbook.save(EXCEL_FILE)
            print(f"[Excel] Saved {len(self.buffer)} entries.")
            self.buffer.clear()
        except Exception as e:
            print(f"[Excel] Error: {e}")
    def stop(self):
        self.running = False
        self.flush()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    cam = CameraThread()
    det = DetectionThread()
    recman = RecognitionManager(num_workers=2)
    excel = ExcelThread()

    cam.start()
    det.start()
    recman.start()
    excel.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Main] Stopping threads...")
        cam.stop()
        det.stop()
        recman.stop()
        excel.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
