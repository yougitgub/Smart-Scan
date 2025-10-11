import os
import cv2
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
import importlib

# Dynamically import heavy optional packages to avoid static linter errors
try:
    torch = importlib.import_module('torch')
except Exception:
    torch = None

MTCNN = None
InceptionResnetV1 = None
try:
    facenet_module = importlib.import_module('facenet_pytorch')
    MTCNN = getattr(facenet_module, 'MTCNN', None)
    InceptionResnetV1 = getattr(facenet_module, 'InceptionResnetV1', None)
except Exception:
    facenet_module = None

YOLO = None
try:
    ultralytics_module = importlib.import_module('ultralytics')
    YOLO = getattr(ultralytics_module, 'YOLO', None)
except Exception:
    ultralytics_module = None

# --- Configuration ---
KNOWN_EMBEDDINGS_FILE = 'known_embeddings.pkl'
ENROLLMENT_DIRECTORY = "students" # Directory containing images named by student (non-recursive)
RECOGNITION_THRESHOLD = 0.7 # Similarity score threshold for a match (adjust as needed)

# Initialize YOLOv8 model for initial face detection (bounding box)
model = None
if YOLO is not None:
    try:
        # Assuming 'detection/weights/yolov12n-best.pt' is available in your environment
        model = YOLO("detection/weights/yolov12n-best.pt")
        print("YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
else:
    print("Warning: ultralytics.YOLO is not available. YOLO-based detection will be disabled.")

# Initialize MTCNN for precise face alignment/cropping and InceptionResnetV1 for embedding
if torch is not None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
else:
    device = 'cpu'
    print("Warning: torch is not available. Using fallback device string 'cpu'.")

# Initialize MTCNN and ResNet only if facenet_pytorch is available
mtcnn = None
resnet = None
if MTCNN is not None and InceptionResnetV1 is not None and torch is not None:
    try:
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        print("MTCNN and InceptionResnetV1 models loaded successfully.")
    except Exception as e:
        print(f"Error initializing MTCNN/ResNet: {e}")
else:
    print("Warning: facenet_pytorch is not available. Face embedding generation will be disabled.")

# Load known embeddings
KNOWN_EMBEDDINGS: Dict[str, List[np.ndarray]] = {}
try:
    with open(KNOWN_EMBEDDINGS_FILE, 'rb') as f:
        # Load the structure {name: [embedding1, embedding2, ...]}
        KNOWN_EMBEDDINGS = pickle.load(f)
        print(f"Known embeddings loaded successfully ({len(KNOWN_EMBEDDINGS)} unique people).")
except FileNotFoundError:
    print("No known embeddings file found. Starting with an empty dictionary.")
except Exception as e:
    print(f"Error loading embeddings file: {e}. Starting with an empty dictionary.")


def get_face_embedding(face_image: np.ndarray) -> Optional[np.ndarray]:
    """
    Takes a detected face image (cropped by YOLO) and processes it with MTCNN/FaceNet
    to get the 512-dimensional embedding vector. Returns a NumPy array.
    """
    if mtcnn is None or resnet is None:
        return None
        
    try:
        # Check if the image is grayscale (shape only has 2 dimensions) and convert to 3-channel
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
        
        # 1. MTCNN detects the face again (optional, but ensures alignment) and converts to Tensor
        # NOTE: MTCNN expects an RGB image (default behavior of facenet_pytorch)
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_tensor = mtcnn(face_rgb)
        
        if face_tensor is None:
            return None

        # 2. Add batch dimension and move to device
        # Ensure tensor shape is correct: [1, 3, 160, 160]
        face_tensor = face_tensor.unsqueeze(0).to(device)
        
        # 3. Get the 512-D embedding
        with torch.no_grad():
            embedding = resnet(face_tensor)

        # Ensure the final output is a NumPy array
        return embedding.detach().cpu().numpy()
        
    except Exception as e:
        # --- CRITICAL CHANGE: Print the error to debug the root cause ---
        print(f"--- ERROR IN EMBEDDING GENERATION: {e} ---")
        return None

def generate_known_embeddings(directory_path: str):
    """
    Processes images directly inside the given directory (non-recursive) and saves the
    generated embeddings. Each image's filename (without extension) is used as the
    person's name. Structure: students/PersonName.jpg
    """
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    updated_embeddings = 0

    # List only files directly inside the directory (no subfolders)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Skip directories and non-image files
        if not os.path.isfile(file_path):
            continue

        if not filename.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
            continue

        # Use the filename (without extension) as the person's name
        name, _ext = os.path.splitext(filename)
        person_embeddings: List[np.ndarray] = []
        print(f"--- Processing image '{filename}' for student '{name}'... ---")

        img = cv2.imread(file_path)
        if img is None:
            print(f"Error: Unable to read image '{file_path}'.")
            continue

        # 1. YOLOv8 for initial detection
        # Note: If YOLO is failing here, no embeddings will be generated either.
        try:
            results = model(img, verbose=False)
            boxes = results[0].boxes # Boxes object from ultralytics
        except Exception as e:
            print(f"Error during YOLO detection for '{filename}': {e}")
            continue

        # Check if no boxes detected
        if boxes is None or len(boxes) == 0:
            print(f"Notice: No faces detected in '{filename}'.")
            continue

        # Convert box coordinates to numpy array (N x 4 or N x 6) and iterate rows
        xyxy = boxes.xyxy
        if hasattr(xyxy, 'cpu'):
            xyxy = xyxy.cpu().numpy()

        for xy in xyxy:
            x1, y1, x2, y2 = [int(val) for val in xy[:4]]
            face_crop = img[y1:y2, x1:x2]

            # Check if crop is valid
            if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                print(f"Warning: Invalid face crop dimensions for face in '{filename}'.")
                continue

            # 2. Get the FaceNet embedding (returns a NumPy array)
            embedding_tensor = get_face_embedding(face_crop)

            if embedding_tensor is not None:
                # embedding_tensor is already a NumPy array, so remove .numpy()
                person_embeddings.append(embedding_tensor.flatten())
                updated_embeddings += 1
            else:
                print(f"Warning: Could not get embedding for face in '{filename}'.")

        if person_embeddings:
            # If name already exists, extend the embeddings list, otherwise set new
            if name in KNOWN_EMBEDDINGS:
                KNOWN_EMBEDDINGS[name].extend(person_embeddings)
            else:
                KNOWN_EMBEDDINGS[name] = person_embeddings

            print(f"SUCCESS: Saved {len(person_embeddings)} embeddings for '{name}'.")
        else:
            print(f"NOTICE: No embeddings found for '{name}' from image '{filename}'.")

    # Save the updated embeddings dictionary
    if updated_embeddings > 0:
        try:
            with open(KNOWN_EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(KNOWN_EMBEDDINGS, f)
            print(f"\n--- Enrollment Complete: Saved {len(KNOWN_EMBEDDINGS)} unique people to '{KNOWN_EMBEDDINGS_FILE}'. ---")
        except Exception as e:
            print(f"FATAL ERROR: Failed to save embeddings file: {e}")
    else:
        print("\n--- Enrollment Complete: No new embeddings were generated. ---")

