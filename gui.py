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
        master.title("Time Policy Configuration")
        # Adjusted height to 300 to account for the removed field
        master.geometry("400x300") 
        master.resizable(False, False)

        # 1. Load initial settings
        self.current_settings = load_settings(CONFIG_FILE)

        # 2. Tkinter variables to hold user input
        self.start_time_var = tk.StringVar(value=self.current_settings['start_time'])
        self.lateness_time_var = tk.StringVar(value=self.current_settings['lateness_time'])
        self.absence_time_var = tk.StringVar(value=self.current_settings['absence_time'])   
        # self.presence_var initialization removed

        self.create_widgets()
        
    def create_widgets(self):
        """Sets up the UI elements using grid layout."""
        
        # Configure grid weights for centering
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        
        frame = tk.Frame(self.master, padx=20, pady=20)
        frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # --- Row 0: Required Start Time ---
        tk.Label(frame, text="Required Start Time (HH:MM, 24h):", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=10)
        self.start_time_entry = tk.Entry(frame, textvariable=self.start_time_var, width=15, relief=tk.GROOVE)
        self.start_time_entry.grid(row=0, column=1, sticky='e', padx=10, pady=10)

        # --- Row 1: Lateness Cutoff Time ---
        tk.Label(frame, text="Lateness Cutoff Time (HH:MM, 24h):", font=('Helvetica', 10, 'bold')).grid(row=1, column=0, sticky='w', pady=10)
        self.lateness_entry = tk.Entry(frame, textvariable=self.lateness_time_var, width=15, relief=tk.GROOVE)
        self.lateness_entry.grid(row=1, column=1, sticky='e', padx=10, pady=10)
        
        # --- Row 2: Absence Cutoff Time ---
        tk.Label(frame, text="Absence Cutoff Time (HH:MM, 24h):", font=('Helvetica', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=10)
        self.absence_entry = tk.Entry(frame, textvariable=self.absence_time_var, width=15, relief=tk.GROOVE)
        self.absence_entry.grid(row=2, column=1, sticky='e', padx=10, pady=10)
        
        # --- Row 3: Save Button (Moved up) ---
        self.save_button = tk.Button(self.master, text="Start the System", command=self.on_save, 
                                     bg='#4CAF50', fg='white', font=('Helvetica', 12, 'bold'), 
                                     activebackground='#45a049', activeforeground='white', 
                                     relief=tk.RAISED, padx=10, pady=5)
        self.save_button.grid(row=3, column=0, columnspan=2, pady=20)
        
    def on_save(self):
        """Validates input, updates settings, and saves to file."""
        
        try:
            # 1. Get and validate time strings
            start_time = self.start_time_var.get()
            lateness_time = self.lateness_time_var.get()
            absence_time = self.absence_time_var.get()
            
            # 2. Validation
            if not validate_time_format(start_time):
                raise ValueError("Start Time must be in valid HH:MM format (06:00, 15:30).")
            if not validate_time_format(lateness_time):
                raise ValueError("Lateness Cutoff Time must be in valid HH:MM format (07:00, 16:00).")
            if not validate_time_format(absence_time):
                raise ValueError("Absence Cutoff Time must be in valid HH:MM format (10:00, 18:00).")

            # 3. Update the settings dictionary
            new_settings = {
                'start_time': start_time,
                'lateness_time': lateness_time,
                'absence_time': absence_time,
            }
            # 'presence_minutes' is no longer included here.
            
            # 4. Save to JSON file
            save_settings(CONFIG_FILE, new_settings)
            
            messagebox.showinfo("Success", "All time settings have been saved successfully System will Start now.")
            root.destroy()  # Close the application after saving
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid data for all fields.\nError: {e}")
            
if __name__ == "__main__":
    root = tk.Tk()
    app = TimeConfigApp(root)
    root.mainloop()
