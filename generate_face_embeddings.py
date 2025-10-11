import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple

# --- Configuration ---
KNOWN_EMBEDDINGS_FILE = 'known_embeddings.pkl'
ENROLLMENT_DIRECTORY = "students" # Directory containing images named by person (non-recursive)
RECOGNITION_THRESHOLD = 0.7  # Similarity score threshold for a match (adjust as needed)

# Initialize YOLOv8 model for initial face detection (bounding box)
try:
    # Assuming 'detection/weights/yolov12n-best.pt' is available in your environment
    model = YOLO("detection/weights/yolov12n-best.pt")
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    # Initialize with a basic model name if the path fails, or raise an error
    # For this example, we assume it's functional.

# Initialize MTCNN for precise face alignment/cropping and InceptionResnetV1 for embedding
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("MTCNN and InceptionResnetV1 models loaded successfully.")

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


def get_face_embedding(face_image: np.ndarray) -> Optional[torch.Tensor]:
    """
    Takes a detected face image (cropped by YOLO) and processes it with MTCNN/FaceNet
    to get the 512-dimensional embedding vector.
    """
    try:
        # 1. MTCNN detects the face again (optional, but ensures alignment) and converts to Tensor
        face_tensor = mtcnn(face_image)
        
        if face_tensor is None:
            return None

        # 2. Add batch dimension and move to device
        face_tensor = face_tensor.unsqueeze(0).to(device)
        
        # 3. Get the 512-D embedding
        with torch.no_grad():
            embedding = resnet(face_tensor)
            
        return embedding.detach().cpu()
        
    except Exception as e:
        # Catch errors if MTCNN/ResNet fails for a specific crop
        # print(f"Error generating embedding: {e}")
        return None

def save_embeddings_from_directory(directory_path: str):
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
        print(f"--- Processing image '{filename}' for person '{name}'... ---")

        img = cv2.imread(file_path)
        if img is None:
            print(f"Error: Unable to read image '{file_path}'.")
            continue

        # 1. YOLOv8 for initial detection
        results = model(img, verbose=False)
        boxes = results[0].boxes  # Boxes object from ultralytics

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

            # 2. Get the FaceNet embedding
            embedding_tensor = get_face_embedding(face_crop)

            if embedding_tensor is not None:
                person_embeddings.append(embedding_tensor.numpy().flatten())
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

def recognize_face_from_image(image_path: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Takes a single image path, detects the face, gets the embedding, 
    and compares it to known embeddings to determine the name.
    
    Returns: (name, embedding)
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'.")
        return None, None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image '{image_path}'.")
        return None, None
        
    print(f"\n--- Recognizing face in '{os.path.basename(image_path)}' ---")
    
    # 1. YOLOv8 for initial detection
    results = model(img, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("Result: No face detected.")
        return "No Face", None

    # Use the first detected face (you might need logic for multiple faces)
    xyxy = boxes.xyxy
    if hasattr(xyxy, 'cpu'):
        xyxy = xyxy.cpu().numpy()

    x1, y1, x2, y2 = [int(val) for val in xyxy[0][:4]]
    face_crop = img[y1:y2, x1:x2]
    
    # 2. Get the FaceNet embedding
    current_embedding_tensor = get_face_embedding(face_crop)
    
    if current_embedding_tensor is None:
        print("Result: Face detected but could not generate a quality embedding.")
        return "Unknown", None

    current_embedding = current_embedding_tensor.numpy().flatten()
    
    # 3. Compare with known embeddings
    best_match_name = "Unknown"
    best_similarity = 0.0

    if not KNOWN_EMBEDDINGS:
        print("Warning: No known embeddings available for comparison.")
        return "Unenrolled", current_embedding

    for name, list_of_embeddings in KNOWN_EMBEDDINGS.items():
        if not list_of_embeddings:
            continue
            
        # Compare current embedding to all stored embeddings for this person
        similarities = [
            (current_embedding @ known_emb) / (np.linalg.norm(current_embedding) * np.linalg.norm(known_emb))
            for known_emb in list_of_embeddings
        ]
        
        # Take the maximum similarity score for this person
        max_similarity = max(similarities)
        
        if max_similarity > best_similarity:
            best_similarity = max_similarity
            best_match_name = name

    # 4. Determine the final result
    if best_similarity >= RECOGNITION_THRESHOLD:
        final_name = f"{best_match_name} (Similarity: {best_similarity:.2f})"
        print(f"Result: Matched to {final_name}")
        return final_name, current_embedding
    else:
        final_name = f"Unknown (Best Similarity: {best_similarity:.2f})"
        print(f"Result: {final_name}")
        return "Unknown", current_embedding

# --- Main Execution ---

# 1. Run enrollment process (Saves/Updates embeddings)
save_embeddings_from_directory(ENROLLMENT_DIRECTORY)

# 2. Example of recognizing a face from a new image
# NOTE: Replace 'test_image.jpg' with the actual path to your image file
test_image_path = "test_image.jpg" 

if os.path.exists(test_image_path):
    recognized_name, embedding = recognize_face_from_image(test_image_path)
    
    if recognized_name and embedding is not None:
        print(f"\nRecognized Name: {recognized_name}")
        print(f"Extracted Embedding Shape: {embedding.shape}")
        # print(f"Embedding vector (first 5 values): {embedding[:5]}")
    else:
        print("\nRecognition failed or no embedding was generated.")
else:
    print(f"\nSkipping test recognition: Please create a '{test_image_path}' file to test the recognition function.")
