import tkinter as tk
import os
# Import the class from gui.py
from gui import TimeConfigApp
# Import the main recognition function
from face_recognition import real_time_face_recognition

if __name__ == "__main__":
    print("--- smart-scan System Startup ---")
    
    # 1. Launch Configuration & Control GUI
    # This window blocks execution until closed (either by "Start System" or closing the window)
    root = tk.Tk()
    app = TimeConfigApp(root)
    root.mainloop()
    
    # 2. Run Real-Time Face Recognition
    # This runs after the GUI window is destroyed
    print("GUI closed. Initializing Face Recognition...")
    try:
        real_time_face_recognition()
    except KeyboardInterrupt:
        print("System stopped by user.")
    except Exception as e:
        print(f"System Error: {e}")
    finally:
        print("Exiting application.")