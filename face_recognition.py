import os
import cv2
import time
import threading # Required for the CameraHandler class
import torch
import faiss # New: FAISS for fast nearest-neighbor search
# Importing facenet_pytorch components
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple

# --- Configuration ---
# NOTE: YOU MUST HAVE THESE WEIGHTS DOWNLOADED AND EMBEDDINGS CREATED
YOLO_MODEL_PATH = 'detection/weights/yolov12n-best.pt' # YOLOv8 model for initial face detection
EMBEDDINGS_FILE = 'known_embeddings.pkl'     # File containing known face embeddings
STREAM_URL = 'rtsp://admin:1234qwer@@172.16.0.3:554/Streaming/Channels/102?tcp' # <<< CHANGE THIS TO YOUR IP CAMERA URL
RECOGNITION_THRESHOLD = 1.0                  # Max Euclidean distance for a match (typically 0.9 to 1.1 for FaceNet)
# ADJUSTED: Raised the threshold back up to 0.65. This filters out the false positives 
# (non-face objects) while still being sensitive enough for most faces.
YOLO_CONFIDENCE_THRESHOLD = 0.65             # Minimum confidence score (0.0 to 1.0) for YOLO to accept a detection.

# --- Temporal Consistency Settings (The "Double-Check" Logic) ---
CONFIRM_FRAMES = 5                           # Number of recent frames to check for consistency.
REQUIRED_VOTES = 3                           # Number of votes required in CONFIRM_FRAMES for a stable ID.
MAX_BOX_DISTANCE = 50                        # Max pixel distance between box centers to consider them the same person.

# --- Global Variables for FAISS ---
# These will hold the FAISS index and the corresponding list of names
FAISS_INDEX = None
KNOWN_NAMES = []


# --- 1. Model Initialization ---
# Load YOLOv8 model (used for fast, rough face cropping)
try:
    model_yolo = YOLO(YOLO_MODEL_PATH)  
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    model_yolo = None

# Load MTCNN (for precise face alignment) and InceptionResnetV1 (FaceNet for embeddings)
try:
    # MTCNN is used to detect and align faces before embedding
    mtcnn = MTCNN(keep_all=True, device='cpu') 
    # ResNet is the FaceNet backbone used to generate the 512-D embedding vector
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    print("MTCNN and InceptionResnetV1 models loaded successfully.")
except Exception as e:
    print(f"Error loading MTCNN/InceptionResnetV1: {e}")
    mtcnn = None
    resnet = None


# --- 2. Data Loading & Utility Functions ---

def load_known_embeddings() -> Tuple[faiss.Index or None, List[str], bool]:
    """
    Loads known face embeddings and builds a FAISS index for fast nearest-neighbor search.
    Returns: (faiss_index, list_of_names, success_status)
    """
    global FAISS_INDEX, KNOWN_NAMES
    
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            raw_embeddings = pickle.load(f)
            
            if not raw_embeddings:
                print(f"Warning: {EMBEDDINGS_FILE} is empty.")
                return None, [], False

            # 1. Extract names and embeddings into structured lists
            names_list = list(raw_embeddings.keys())
            embeddings_list = [np.array(emb) for emb in raw_embeddings.values()]
            
            # 2. Convert to a single NumPy matrix (float32 required by FAISS)
            embeddings_matrix = np.vstack(embeddings_list).astype('float32')
            D = embeddings_matrix.shape[1] # Dimension (should be 512)

            # 3. Initialize FAISS Index (Flat L2 for exact Euclidean distance)
            # IndexFlatL2 provides the same exact distance calculation as numpy.linalg.norm, but optimized
            index = faiss.IndexFlatL2(D)
            index.add(embeddings_matrix)
            
            print(f"Known embeddings loaded and FAISS index built successfully. ({len(names_list)} identities)")
            
            # Store globally for use in the recognition loop
            FAISS_INDEX = index
            KNOWN_NAMES = names_list
            
            return index, names_list, True
            
    except Exception as e:
        print(f"FATAL: Error loading or building FAISS index from {EMBEDDINGS_FILE}: {e}")
        return None, [], False

def get_face_embedding(face_image: np.ndarray, mtcnn_model: MTCNN, resnet_model: InceptionResnetV1) -> np.ndarray or None:
    """
    Processes a cropped face image to generate a 512-dimensional embedding vector.
    """
    if mtcnn_model is None or resnet_model is None:
        return None
        
    # 1. Convert BGR (OpenCV) to RGB (PyTorch/MTCNN)
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # 2. Detect and align the face precisely with MTCNN
    # MTCNN returns a tensor: [N, C, H, W] where N=1 for a single crop
    face_tensor = mtcnn_model(face_rgb)
    
    if face_tensor is None:
        return None # No face detected by MTCNN in the cropped area

    # 3. Generate the embedding vector using FaceNet/InceptionResnetV1
    face_embedding = resnet_model(face_tensor).detach().cpu().numpy().flatten()
    
    # Fix for 1024 vs 512 dimension mismatch during live inference
    expected_dim = 512
    current_dim = face_embedding.shape[0]
    
    if current_dim != expected_dim:
        if current_dim == 1024 and expected_dim == 512:
            # Truncate to 512, assuming the embedding is in the first half
            face_embedding = face_embedding[:expected_dim]
        else:
            print(f"FATAL: Embedding size mismatch. Expected {expected_dim}, got {current_dim}. Cannot proceed.")
            return None
        
    return face_embedding

# --- THREADED CAMERA HANDLER FOR STABILITY (User Provided) ---
class CameraHandler(threading.Thread):
    """
    Handles video capture in a separate thread to prevent the main loop from blocking.
    """
    def __init__(self, stream_url: str):
        super().__init__()
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(stream_url)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        self.frame = None
        self.success = False
        self.stopped = True
        self.lock = threading.Lock()
        
        if not self.cap.isOpened():
            print(f"Error: Could not open stream {stream_url}. Check URL or camera status.")
        else:
            self.success, self.frame = self.cap.read()
            self.stopped = False
            print("CameraHandler started successfully.")

    def run(self):
        """The main thread loop that continuously reads the newest frame."""
        while not self.stopped:
            success, frame = self.cap.read()
            
            if success:
                with self.lock:
                    self.frame = frame
                    self.success = True
            else:
                self.success = False

    def read(self):
        """Returns the last successfully read frame and its success status."""
        with self.lock:
            return self.success, self.frame

    def stop(self):
        """Signals the thread to stop and releases the video capture."""
        self.stopped = True
        self.join() 
        self.cap.release()
        print("CameraHandler stopped.")


# --- 3. Consistency Helper Functions ---

# Dictionary to store the recognition history for currently tracked faces
# Structure: { face_id: { 'box': (x1, y1, x2, y2), 'history': ['NameA', 'NameA', 'Unknown', ...] } }
face_history: Dict[int, Dict[str, Any]] = {}
current_face_id: int = 0

def find_nearest_tracked_face(current_box: Tuple[int, int, int, int]) -> int or None:
    """Finds the ID of the closest tracked face from the previous frame."""
    global face_history
    cx_current = (current_box[0] + current_box[2]) / 2
    cy_current = (current_box[1] + current_box[3]) / 2
    
    min_dist = float('inf')
    matched_id = None
    
    for face_id, data in face_history.items():
        hx1, hy1, hx2, hy2 = data['box']
        cx_hist = (hx1 + hx2) / 2
        cy_hist = (hy1 + hy2) / 2
        
        # Calculate Euclidean distance between box centers
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
        
    # Get the name with the highest count
    stable_name = max(counts, key=counts.get)
    
    if counts[stable_name] >= REQUIRED_VOTES:
        return stable_name
    else:
        # If no single name/identity meets the required votes, it is considered unstable
        return "Unknown_Unstable"


# --- 4. Real-Time Recognition Loop ---

def recognize_faces_in_stream():
    global face_history, current_face_id

    # Load the known embeddings and build the FAISS index
    faiss_index, known_names, load_success = load_known_embeddings()
    
    if not load_success or faiss_index is None:
        print("Exiting recognition: FAISS index could not be initialized.")
        return

    # Initialize Camera Handler (IP camera feed)
    camera = CameraHandler(STREAM_URL)
    if camera.stopped: # Check if initialization failed
        return

    camera.start()
    
    try:
        print("\n--- Starting Real-Time Recognition (Press 'q' to exit) ---")
        
        while True:
            success, frame = camera.read()
            
            if not success or frame is None:
                time.sleep(0.01)
                continue

            # 1. Use YOLO to detect faces in the current frame
            yolo_results = model_yolo(frame, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD) 
            
            # Temporary dict to hold associations for the current frame
            current_frame_detections: Dict[int, Dict[str, Any]] = {}
            
            for box in yolo_results[0].boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_box = (x1, y1, x2, y2)
                
                # --- Tracking and History Lookup ---
                matched_id = find_nearest_tracked_face(current_box)
                
                if matched_id is None:
                    # New face detected: assign a new ID
                    matched_id = current_face_id
                    current_face_id += 1
                    # Initialize history for the new face
                    history_list = []
                else:
                    # Existing face: retrieve its history
                    history_list = face_history[matched_id]['history']
                
                # Update the temporary tracker for the next iteration
                current_frame_detections[matched_id] = {'box': current_box, 'history': history_list}
                
                recognition_result = "Unknown" # The single-frame result
                min_distance = float('inf')
                best_match = None
                
                face_crop = frame[y1:y2, x1:x2]
                
                # 2. Generate embedding for the detected face
                embedding = get_face_embedding(face_crop, mtcnn, resnet)
                
                if embedding is not None:
                    # Reshape the embedding for FAISS search (FAISS expects (1, D))
                    query_embedding = embedding.reshape(1, -1).astype('float32')
                    
                    # 3. Compare with known embeddings using FAISS search
                    K = 1 # Search for the single closest neighbor
                    # D: Distances (size K), I: Indices (size K)
                    D, I = faiss_index.search(query_embedding, K) 
                    
                    min_distance = D[0][0] # The distance (L2 norm squared) to the nearest neighbor
                    best_match_index = I[0][0] # The index of the nearest neighbor
                    
                    # Check if a valid index was found
                    if best_match_index != -1: 
                        # Map the FAISS index back to the name
                        best_match = known_names[best_match_index]
                    else:
                        best_match = None

                    # 4. Determine recognition result for THIS single frame
                    # The FAISS L2 distance is the squared Euclidean distance.
                    # We compare it against the square of the RECOGNITION_THRESHOLD.
                    if best_match is not None and min_distance < (RECOGNITION_THRESHOLD ** 2):
                        recognition_result = best_match
                
                # Update the history with the single-frame result
                history_list.append(recognition_result)
                
                # Keep history list limited to CONFIRM_FRAMES
                if len(history_list) > CONFIRM_FRAMES:
                    history_list.pop(0)

                # 5. Determine the STABLE name based on the history (The "Double Check")
                stable_name = determine_stable_name(history_list)
                
                display_name = ""
                
                if stable_name != "Unknown_Unstable":
                    # Stable match found (green box)
                    display_name = f"{stable_name} (Stable)"
                    color = (0, 255, 0) 
                else:
                    # No stable match (red box)
                    # For unstable knowns, still show the best guess but keep the color red
                    if recognition_result != "Unknown":
                        display_name = f"Unstable: {recognition_result}"
                    else:
                        display_name = f"Unknown"
                    color = (0, 0, 255) 

                # 6. Draw the bounding box and name on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, display_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Update the global history for the next frame
            face_history = current_frame_detections
            
            # Display the result
            cv2.imshow("Real-Time Face Recognition", frame)
            
            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()


# --- 5. Main Execution Block ---

if __name__ == "__main__":
    # Ensure FAISS is available
    try:
        import faiss
    except ImportError:
        print("FATAL ERROR: FAISS is not installed. Please install it using: pip install faiss-cpu")
        exit()
        
    if model_yolo is None or mtcnn is None or resnet is None:
        print("\nExiting: Critical models failed to load. Please check model paths and dependencies.")
    else:
        # Run the real-time recognition loop
        recognize_faces_in_stream()

    print("\nApplication terminated.")
