import tkinter as tk
from tkinter import messagebox
import json
import os
import re # Import for checking the time format

# --- Configuration ---
CONFIG_FILE = 'config.json'
DEFAULT_TIMES = {
    'lateness_time': "07:00",  # Time after start_time that is considered late
    'absence_time': "10:00",   # Time after start_time that is considered an absence
    'start_time': "06:00"      # Required clock-in time in HH:MM format
}
# 'presence_minutes' has been removed from the default settings.

# Regex to validate HH:MM format
TIME_REGEX = re.compile(r'^\d{2}:\d{2}$')

# --- Core Functions ---

def load_settings(file_path: str) -> dict:
    """
    Loads configuration settings from a JSON file. 
    Returns default settings if the file is not found or is invalid.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                settings = json.load(f)
                # Ensure the loaded data has ALL expected keys. Use defaults for missing keys if necessary.
                updated_settings = DEFAULT_TIMES.copy()
                updated_settings.update(settings)
                
                # We check again if all keys are present after updating from file
                if all(key in updated_settings for key in DEFAULT_TIMES):
                    print(f"Loaded settings from {file_path}")
                    return updated_settings
        except (json.JSONDecodeError, IOError, TypeError) as e:
            print(f"Error loading settings from {file_path}: {e}. Using default.")
    
    print(f"Using default settings: {DEFAULT_TIMES}")
    return DEFAULT_TIMES

def save_settings(file_path: str, settings: dict):
    """
    Saves the current configuration settings to a JSON file.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"Settings saved successfully to {file_path}")
    except IOError as e:
        messagebox.showerror("Save Error", f"Could not save settings file: {e}")

def validate_time_format(time_str: str) -> bool:
    """Checks if a string is a valid time in HH:MM format (00:00 to 23:59)."""
    if TIME_REGEX.match(time_str):
        try:
            H, M = map(int, time_str.split(':'))
            # Check hours (0-23) and minutes (0-59) for validity
            if 0 <= H <= 23 and 0 <= M <= 59:
                return True
        except ValueError:
            return False
    return False

# --- Tkinter Application Class ---

class TimeConfigApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Time Policy Configuration")
        # Slightly smaller window and non-resizable for a compact layout
        self.master.geometry("350x320")
        self.master.resizable(False, False)

        # 1. Load initial settings
        self.current_settings = load_settings(CONFIG_FILE)

        # 2. Tkinter variables to hold user input
        # Split existing HH:MM into hours and minutes vars for separate inputs
        def split_time(t: str):
            try:
                h, m = t.split(":")
                return h.zfill(2), m.zfill(2)
            except Exception:
                return "00", "00"

        sh, sm = split_time(self.current_settings.get('start_time', '06:00'))
        lh, lm = split_time(self.current_settings.get('lateness_time', '07:00'))
        ah, am = split_time(self.current_settings.get('absence_time', '10:00'))

        self.start_hour_var = tk.StringVar(value=sh)
        self.start_min_var = tk.StringVar(value=sm)

        self.lateness_hour_var = tk.StringVar(value=lh)
        self.lateness_min_var = tk.StringVar(value=lm)

        self.absence_hour_var = tk.StringVar(value=ah)
        self.absence_min_var = tk.StringVar(value=am)

        self.create_widgets()

    def create_widgets(self):
        """Sets up the UI elements using grid layout."""

        # Configure grid weights for centering
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        # Main content frame with tighter padding
        frame = tk.Frame(self.master, padx=12, pady=12)
        frame.grid(row=0, column=0, columnspan=2, padx=8, pady=8, sticky="nsew")

        # --- Row 0: Required Start Time ---
        tk.Label(frame, text="Required Start Time:", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=8)
        start_frame = tk.Frame(frame)
        start_frame.grid(row=0, column=1, sticky='e', padx=10, pady=6)
        self.start_hour_entry = tk.Entry(start_frame, textvariable=self.start_hour_var, width=3, relief=tk.GROOVE)
        colon_label = tk.Label(start_frame, text=":")
        self.start_min_entry = tk.Entry(start_frame, textvariable=self.start_min_var, width=3, relief=tk.GROOVE)
        # Pack in order: hour, small label, colon, minute, small label
        self.start_hour_entry.pack(side='left')
        tk.Label(start_frame, text='hrs', font=('Helvetica', 9)).pack(side='left', padx=(4,6))
        colon_label.pack(side='left', padx=4)
        self.start_min_entry.pack(side='left')
        tk.Label(start_frame, text='min', font=('Helvetica', 9)).pack(side='left', padx=(4,0))

        # --- Row 1: Lateness Cutoff Time ---
        tk.Label(frame, text="Lateness Cutoff Time:", font=('Helvetica', 10, 'bold')).grid(row=1, column=0, sticky='w', pady=8)
        lateness_frame = tk.Frame(frame)
        lateness_frame.grid(row=1, column=1, sticky='e', padx=10, pady=6)
        self.lateness_hour_entry = tk.Entry(lateness_frame, textvariable=self.lateness_hour_var, width=3, relief=tk.GROOVE)
        colon_label2 = tk.Label(lateness_frame, text=":")
        self.lateness_min_entry = tk.Entry(lateness_frame, textvariable=self.lateness_min_var, width=3, relief=tk.GROOVE)
        self.lateness_hour_entry.pack(side='left')
        tk.Label(lateness_frame, text='hrs', font=('Helvetica', 9)).pack(side='left', padx=(4,6))
        colon_label2.pack(side='left', padx=4)
        self.lateness_min_entry.pack(side='left')
        tk.Label(lateness_frame, text='min', font=('Helvetica', 9)).pack(side='left', padx=(4,0))

        # --- Row 2: Absence Cutoff Time ---
        tk.Label(frame, text="Absence Cutoff Time:", font=('Helvetica', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=8)
        absence_frame = tk.Frame(frame)
        absence_frame.grid(row=2, column=1, sticky='e', padx=10, pady=6)
        self.absence_hour_entry = tk.Entry(absence_frame, textvariable=self.absence_hour_var, width=3, relief=tk.GROOVE)
        colon_label3 = tk.Label(absence_frame, text=":")
        self.absence_min_entry = tk.Entry(absence_frame, textvariable=self.absence_min_var, width=3, relief=tk.GROOVE)
        self.absence_hour_entry.pack(side='left')
        tk.Label(absence_frame, text='hrs', font=('Helvetica', 9)).pack(side='left', padx=(4,6))
        colon_label3.pack(side='left', padx=4)
        self.absence_min_entry.pack(side='left')
        tk.Label(absence_frame, text='min', font=('Helvetica', 9)).pack(side='left', padx=(4,0))

        # --- Action Buttons (placed inside the frame for compactness) ---
        buttons_frame = tk.Frame(frame)
        buttons_frame.grid(row=3, column=0, columnspan=2, pady=(14, 4))

        self.save_button = tk.Button(buttons_frame, text="Start the System", command=self.on_save,
                                     bg='#4CAF50', fg='white', font=('Helvetica', 14, 'bold'),
                                     activebackground='#45a049', activeforeground='white',
                                     relief=tk.RAISED, padx=12, pady=8)
        self.save_button.pack(fill='x', padx=10, pady=(0, 8))

        self.rescan_button = tk.Button(buttons_frame, text="Re-Scan Student Images", command=self.rescan,
                                       bg='#2196F3', fg='white', font=('Helvetica', 10, 'bold'),
                                       activebackground='#0b7dda', activeforeground='white',
                                       relief=tk.RAISED, padx=10, pady=6)
        self.rescan_button.pack(padx=60, pady=(0, 4))

    def on_save(self):
        """Validates input, updates settings, and saves to file."""

        try:
            # 1. Get hours and minutes from inputs and validate ranges
            sh = self.start_hour_var.get()
            sm = self.start_min_var.get()
            lh = self.lateness_hour_var.get()
            lm = self.lateness_min_var.get()
            ah = self.absence_hour_var.get()
            am = self.absence_min_var.get()

            def validate_hm(h_str, m_str, field_name):
                if not (h_str.isdigit() and m_str.isdigit()):
                    raise ValueError(f"{field_name}: Hours and minutes must be numeric.")
                h = int(h_str)
                m = int(m_str)
                if not (0 <= h <= 23):
                    raise ValueError(f"{field_name}: Hours must be between 0 and 23.")
                if not (0 <= m <= 59):
                    raise ValueError(f"{field_name}: Minutes must be between 0 and 59.")
                return f"{h:02d}:{m:02d}"

            start_time = validate_hm(sh, sm, 'Start Time')
            lateness_time = validate_hm(lh, lm, 'Lateness Cutoff Time')
            absence_time = validate_hm(ah, am, 'Absence Cutoff Time')

            # 3. Update the settings dictionary
            new_settings = {
                'start_time': start_time,
                'lateness_time': lateness_time,
                'absence_time': absence_time,
            }

            # 4. Save to JSON file
            save_settings(CONFIG_FILE, new_settings)

            messagebox.showinfo("Success", "All time settings have been saved successfully. System will start now.")
            
            self.master.destroy()  # Close the application after saving
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid data for all fields.\nError: {e}")

    def rescan(self):
        """Trigger re-scan of student images to update embeddings."""
        from generate_face_embeddings import generate_known_embeddings, ENROLLMENT_DIRECTORY, KNOWN_EMBEDDINGS_FILE
        if messagebox.askyesno("Re-Scan Confirmation", "This will re-scan the student images and update embeddings. Continue?"):
            try:
                generate_known_embeddings(ENROLLMENT_DIRECTORY)
                messagebox.showinfo("Re-Scan Complete", "Student images have been re-scanned and embeddings updated.")
            except Exception as e:
                messagebox.showerror("Re-Scan Error", f"An error occurred during re-scan: {e}")
root = tk.Tk()
app = TimeConfigApp(root)
