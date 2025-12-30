import os
import cv2
import pickle
import numpy as np
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import sys
import subprocess

# --- Imports & Models --- #
try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from ultralytics import YOLO
    from PIL import Image, ImageTk  # for displaying images in GUI
    import mysql.connector # Mysql Connection
except ImportError as e:
    print(f"[ERROR] Missing library: {e}", file=sys.stderr)
    if __name__ == "__main__":
        sys.exit(1)
    else:
        # Fallback for gui import
        torch = None; MTCNN = None; InceptionResnetV1 = None; YOLO = None; Image = None; ImageTk = None

# Configuration
KNOWN_EMBEDDINGS_FILE = 'known_embeddings.pkl'
ENROLLMENT_DIRECTORY = "students" 
if torch:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = 'cpu'

# Laravel DB Config
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '', 
    'database': 'system'
}

# 1. Load YOLO (Face Detection)
YOLO_MODEL = None
if YOLO:
    try:
        weights = "detection/weights/yolov12n-face.pt"
        if not os.path.exists(weights):
            weights = "detection/weights/yolov12n-best.pt"
        
        YOLO_MODEL = YOLO(weights)
        print(f"[TECH] YOLO loaded: {weights}", file=sys.stderr)
    except Exception as e:
        print(f"[TECH] YOLO Warning: {e}", file=sys.stderr)

# 2. Load FaceNet (Embeddings)
mtcnn = None
resnet = None
if MTCNN and InceptionResnetV1:
    try:
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            keep_all=False, device=DEVICE
        )
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
        print("[TECH] FaceNet models loaded.", file=sys.stderr)
    except Exception as e:
        print(f"[TECH] FaceNet Error: {e}", file=sys.stderr)
        if __name__ == "__main__":
            sys.exit(1)

# --- Core Logic --- #

def get_php_bcrypt_hash(password):
    """ Use local PHP to generate a Bcrypt hash compatible with Laravel """
    try:
        cmd = ['php', '-r', f"echo password_hash('{password}', PASSWORD_BCRYPT);"]
        
        if sys.platform == 'win32':
             hash_str = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        else:
             hash_str = subprocess.check_output(cmd).decode('utf-8').strip()
             
        return hash_str
    except Exception as e:
        print(f"[TECH] Password hashing failed: {e}", file=sys.stderr)
        return password 

def sync_to_mysql(national_id, name, year, classroom):
    """ Inserts or Updates student info in the Laravel MySQL DB """
    if 'mysql.connector' not in sys.modules:
        return
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # 1. HANDLE PARENT USER
        cursor.execute("SELECT id FROM users WHERE username = %s LIMIT 1", (national_id,))
        parent = cursor.fetchone()
        
        parent_id = None
        if parent:
            parent_id = parent['id']
            print(f"[TECH] Parent account exists (ID: {parent_id})", file=sys.stderr)
        else:
            # Create Parent
            hashed_pw = get_php_bcrypt_hash("password123")
            parent_name = f"Parent of {name}"
            parent_email = f"parent_{national_id}@smartscan.com"
            
            insert_user_sql = """
                INSERT INTO users (username, name, email, password, role, must_change_password, created_at, updated_at)
                VALUES (%s, %s, %s, %s, 'parent', 1, NOW(), NOW())
            """
            cursor.execute(insert_user_sql, (national_id, parent_name, parent_email, hashed_pw))
            parent_id = cursor.lastrowid
            print(f"✓ Created parent account for {name}")
            
        # 2. UPSERT STUDENT
        query = """
        INSERT INTO students (national_id, full_name, year, class, parent_id, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
        ON DUPLICATE KEY UPDATE
        full_name = VALUES(full_name),
        year = VALUES(year),
        class = VALUES(class),
        parent_id = VALUES(parent_id),
        updated_at = NOW()
        """
        
        cursor.execute(query, (national_id, name, year, classroom, parent_id))
        conn.commit()
        conn.close()
        print(f"✓ Synced {name} to database")
        
    except Exception as e:
        print(f"[TECH] MySQL Sync Error: {e}", file=sys.stderr)

def get_face_embedding(image_rgb):
    """
    Generate embedding from an image (crop or full).
    Input: RGB NumPy array.
    """
    if mtcnn is None or resnet is None:
        return None

    try:
        # 1. Try MTCNN to detect/align/normalize
        face_tensor = mtcnn(image_rgb)
        
        # 2. Fallback: If MTCNN returns None (no face detected/aligned), 
        # MANUAL FALLBACK: Just resize the image, assume it IS a face crop.
        if face_tensor is None:
            # Resize
            img_resized = cv2.resize(image_rgb, (160, 160))
            # To Tensor: HxWxC -> CxHxW
            face_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
            # Normalize fixed (standard Inception normalization)
            face_tensor = (face_tensor - 127.5) / 128.0
        
        # 3. Generate Vector
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0) # Add batch dim
            
        face_tensor = face_tensor.to(DEVICE)
        
        with torch.no_grad():
            emb = resnet(face_tensor)
            
        return emb.detach().cpu().numpy().flatten()
        
    except Exception as e:
        print(f"[TECH] Embedding error: {e}", file=sys.stderr)
        return None

def show_gui(name, n_id, image_path=None):
    """ Simple blocking popup for Year/Class with Image Preview """
    try:
        root = tk.Tk()
        root.title("Student Info")
        # Center - slightly taller to accommodate image
        root.geometry(f"300x420+{int(root.winfo_screenwidth()/2-150)}+{int(root.winfo_screenheight()/2-210)}")
        
        res = {'y': None, 'c': None, 'ok': False}
        
        # --- Image Preview ---
        if image_path and os.path.exists(image_path) and Image:
            try:
                pil_img = Image.open(image_path)
                # Resize to fit (e.g., 150x150 max) keeping aspect ratio
                pil_img.thumbnail((150, 150))
                tk_img = ImageTk.PhotoImage(pil_img)
                
                img_lbl = ttk.Label(root, image=tk_img)
                img_lbl.image = tk_img # Keep reference!
                img_lbl.pack(pady=10)
            except Exception as e:
                print(f"[TECH] Image Preview Error: {e}", file=sys.stderr)
                
        ttk.Label(root, text=name, font=("Arial", 12, "bold")).pack(pady=5)
        ttk.Label(root, text=n_id, font=("Arial", 10)).pack(pady=5)
        
        ttk.Label(root, text="Year:").pack()
        y_var = ttk.Combobox(root, values=["1","2","3"], state="readonly")
        y_var.pack()
        
        ttk.Label(root, text="Class:").pack()
        c_var = ttk.Combobox(root, values=[f"{l}-{n}" for l in "ABCDEF" for n in range(1, 11)], state="readonly")
        c_var.pack()
        
        def save():
            if y_var.get() and c_var.get():
                res.update({'y': int(y_var.get()), 'c': c_var.get(), 'ok': True})
                root.destroy()
        
        ttk.Button(root, text="Save", command=save).pack(pady=10)
        
        # Bring to front
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        
        root.mainloop()
        return (res['y'], res['c']) if res['ok'] else (None, None)
    except Exception as e:
        print(f"[TECH] GUI Error: {e}", file=sys.stderr)
        return None, None

def generate_known_embeddings(directory_path: str):
    """ Wrapper for external calls """
    main_logic(enroll=False, on_exist='skip', directory=directory_path)

def main_logic(enroll=False, on_exist='skip', directory=ENROLLMENT_DIRECTORY):
    on_exist = on_exist.lower().strip()
    print(f"━━━ Enrollment Mode: {on_exist.upper()} ━━━")
    interactive = enroll
    
    # Load DB
    db = {}
    if os.path.exists(KNOWN_EMBEDDINGS_FILE):
        try:
            with open(KNOWN_EMBEDDINGS_FILE, 'rb') as f:
                db = pickle.load(f)
                if db and isinstance(list(db.values())[0], list): db = {} # Reset if old format
        except: db = {}

    if not os.path.exists(directory):
        print(f"❌ Directory '{directory}' not found.")
        return

    # Process
    count = 0
    total_students = sum(1 for s in os.listdir(directory) if os.path.isdir(os.path.join(directory, s)))
    current = 0
    
    for student in os.listdir(directory):
        s_path = os.path.join(directory, student)
        if not os.path.isdir(s_path): continue
        
        current += 1
        
        for file in os.listdir(s_path):
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            
            f_path = os.path.join(s_path, file)
            sid, _ = os.path.splitext(file) # ID is filename
            
            # --- 1. SKIP CHECK ---
            if sid in db and on_exist == 'skip':
                 print(f"⏭  Skipped: {student} (already enrolled)")
                 print(f"[TECH] Student {sid} exists in DB, policy=skip", file=sys.stderr)
                 continue
            elif sid in db and on_exist == 'replace':
                 print(f"🔄 Re-enrolling: {student} (Policy: Replace)")
            elif sid in db and on_exist == 'append':
                 pass # Will append later
            
            print(f"[{current}/{total_students}] Processing {student}...", end='')
            print(f"\n[TECH] Processing file: {f_path}", file=sys.stderr)
            
            # Read
            img = cv2.imread(f_path)
            if img is None: 
                print(" ❌ Read Error")
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embeddings = []
            
            # 2. Detect & Generate
            crops = []
            if YOLO_MODEL:
                try:
                    res = YOLO_MODEL(img, verbose=False)
                    if res and len(res[0].boxes) > 0:
                        for box in res[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                            h, w, _ = img.shape
                            crop = img_rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                            crops.append(crop)
                        print(f"[TECH] YOLO found {len(crops)} face(s)", file=sys.stderr)
                except Exception as e:
                    print(f"[TECH] YOLO detection failed: {e}", file=sys.stderr)
            
            if crops:
                for crop in crops:
                    if crop.size > 0:
                        v = get_face_embedding(crop)
                        if v is not None: embeddings.append(v)
            else:
                 # YOLO failed -> full Image
                 print(f"[TECH] No YOLO detections, using full image", file=sys.stderr)
                 v = get_face_embedding(img_rgb)
                 if v is not None: embeddings.append(v)
            
            if not embeddings:
                print(" ❌ No Face Found")
                continue

            # 3. Store in DB
            if sid not in db:
                db[sid] = {'name': student, 'embeddings': [], 'year': None, 'class': None}
            
            # Update Embeddings Logic
            # Update Embeddings Logic
            if on_exist == 'replace' or on_exist == 'skip': 
                # If 'skip', we are here strictly because it's a NEW user (existing were skipped above).
                # If 'replace', we overwrite.
                db[sid]['embeddings'] = embeddings
            elif on_exist == 'append':
                db[sid]['embeddings'].extend(embeddings)
            else:
                pass 
            
            print(f" ✓ {len(embeddings)} face vector(s)")
            count += 1
            
            # 4. Interactive GUI (Input Year/Class)
            if interactive:
                current_y = db[sid].get('year')
                current_c = db[sid].get('class')
                
                # Force GUI if info missing OR strategy is REPLACE
                should_ask = (current_y is None or current_c is None) or (on_exist == 'replace')
                
                if should_ask:
                    print(f"  ⏸  Waiting for user input...")
                    y, c = show_gui(student, sid, image_path=f_path)
                    if y: 
                        db[sid]['year'] = y
                        db[sid]['class'] = c
                        print(f"  ✓ Details saved: Year {y}, Class {c}")
                        sync_to_mysql(sid, student, y, c)
                    else:
                         print(f"  ⏭  Skipped (cancelled)")
                else:
                    print(f"[TECH] Metadata exists, skipping GUI", file=sys.stderr)
                    sync_to_mysql(sid, student, current_y, current_c)
            else:
                # Non-interactive Mode Sync
                y = db[sid].get('year')
                c = db[sid].get('class')
                if y and c:
                     sync_to_mysql(sid, student, y, c)


    if count > 0:
        with open(KNOWN_EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(db, f)
        print(f"\n━━━ Complete! Processed {count} student(s) ━━━")
    else:
        print(f"\n━━━ No changes made ━━━")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enroll', default='false')
    parser.add_argument('--on-exist', default='skip')
    args = parser.parse_args()
    
    main_logic(enroll=(args.enroll.lower() =='true'), on_exist=args.on_exist)

if __name__ == "__main__":
    main()
