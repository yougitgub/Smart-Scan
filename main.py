import tkinter as tk
import os
from gui import TimeConfigApp
from face_recognition import real_time_face_recognition

if __name__ == "__main__":
    print("--- smart-scan System Startup ---")
    root = tk.Tk()
    app = TimeConfigApp(root)
    root.mainloop()
    print("Application Exited.")