import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import json
import os
import subprocess
import sys
import threading

# --- Configuration ---
CONFIG_FILE = 'config.json'
DEFAULT_TIMES = {
    'lateness_time': "07:00",
    'absence_time': "10:00",
    'start_time': "06:00"
}

def load_settings(file_path: str) -> dict:
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                settings = json.load(f)
                updated_settings = DEFAULT_TIMES.copy()
                updated_settings.update(settings)
                return updated_settings
        except:
            pass
    return DEFAULT_TIMES

def save_settings(file_path: str, settings: dict):
    try:
        with open(file_path, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"Settings saved.")
    except Exception as e:
        messagebox.showerror("Error", f"Could not save settings: {e}")

class TimeConfigApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Attendance Manager")
        self.master.geometry("600x750") # Taller for logs
        self.master.resizable(True, True)
        
        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.colors = {
            'bg': '#f3f4f6', 
            'white': '#ffffff',
            'primary': '#3b82f6', 
            'success': '#10b981', 
            'text': '#1f2937',
            'text_light': '#6b7280',
            'black': '#000000'
        }
        
        self.master.configure(bg=self.colors['bg'])
        self.style.configure('TFrame', background=self.colors['bg'])
        self.style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['text'], font=('Segoe UI', 10))
        self.style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'), foreground=self.colors['primary'])
        self.style.configure('Card.TFrame', background=self.colors['white'], relief='flat')
        self.style.configure('CardLabel.TLabel', background=self.colors['white'])
        
        self.style.configure('Primary.TButton', font=('Segoe UI', 10, 'bold'), borderwidth=0, focuscolor='none')
        self.style.map('Primary.TButton', background=[('active', '#2563eb'), ('!disabled', self.colors['primary'])], foreground=[('!disabled', 'white')])

        self.style.configure('Success.TButton', font=('Segoe UI', 10, 'bold'), borderwidth=0, focuscolor='none')
        self.style.map('Success.TButton', background=[('active', '#059669'), ('!disabled', self.colors['success'])], foreground=[('!disabled', 'white')])

        self.current_settings = load_settings(CONFIG_FILE)
        
        # Vars
        def split_time(t):
            try: return t.split(":") 
            except: return "00", "00"

        sh, sm = split_time(self.current_settings.get('start_time', '06:00'))
        lh, lm = split_time(self.current_settings.get('lateness_time', '07:00'))
        ah, am = split_time(self.current_settings.get('absence_time', '10:00'))

        self.start_h, self.start_m = tk.StringVar(value=sh), tk.StringVar(value=sm)
        self.late_h, self.late_m = tk.StringVar(value=lh), tk.StringVar(value=lm)
        self.abs_h, self.abs_m = tk.StringVar(value=ah), tk.StringVar(value=am)

        self.enroll_interactive = tk.BooleanVar(value=False)
        self.on_exist_var = tk.StringVar(value="skip")

        self.create_widgets()

    def create_widgets(self):
        main_container = ttk.Frame(self.master, padding=20)
        main_container.pack(fill='both', expand=True)

        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill='x', pady=(0, 20))
        ttk.Label(header_frame, text="SmartScan Admin", font=('Segoe UI', 18, 'bold'), foreground='#111827').pack(side='left')
        ttk.Label(header_frame, text="Production Mode", foreground=self.colors['text_light']).pack(side='right', pady=(10,0))

        # Card 1: Time
        self._create_card(main_container, "Attendance Schedule", self._build_time_ui)

        # Card 2: Enrollment
        self._create_card(main_container, "Student Enrollment", self._build_enroll_ui)

        # Card 3: Logs
        self._create_card(main_container, "System Logs", self._build_log_ui)

        # Footer
        footer_frame = ttk.Frame(main_container, padding=(0, 10))
        footer_frame.pack(fill='x', side='bottom')
        
        save_btn = ttk.Button(footer_frame, text="START SYSTEM", style='Success.TButton', command=self.on_start)
        save_btn.pack(fill='x', ipady=8)

    def _create_card(self, parent, title, build_fn):
        card = ttk.Frame(parent, style='Card.TFrame', padding=15)
        card.pack(fill='x', pady=(0, 15))
        title_lbl = ttk.Label(card, text=title.upper(), font=('Segoe UI', 9, 'bold'), foreground=self.colors['text_light'], style='CardLabel.TLabel')
        title_lbl.pack(anchor='w', pady=(0, 15))
        build_fn(card)

    def _build_time_ui(self, parent):
        grid = ttk.Frame(parent, style='Card.TFrame')
        grid.pack(fill='x')
        self._time_row(grid, "Start Time", self.start_h, self.start_m, 0)
        self._time_row(grid, "Late After", self.late_h, self.late_m, 1)
        self._time_row(grid, "Absent After", self.abs_h, self.abs_m, 2)

    def _time_row(self, parent, label, h, m, row):
        parent.columnconfigure(1, weight=1)
        lbl = ttk.Label(parent, text=label, style='CardLabel.TLabel')
        lbl.grid(row=row, column=0, sticky='w', pady=8)
        input_frame = ttk.Frame(parent, style='Card.TFrame')
        input_frame.grid(row=row, column=1, sticky='e')
        ttk.Entry(input_frame, textvariable=h, width=3, justify='center', font=('Segoe UI', 11)).pack(side='left')
        ttk.Label(input_frame, text=" : ", style='CardLabel.TLabel', font=('Segoe UI', 11, 'bold')).pack(side='left')
        ttk.Entry(input_frame, textvariable=m, width=3, justify='center', font=('Segoe UI', 11)).pack(side='left')

    def _build_enroll_ui(self, parent):
        chk = ttk.Checkbutton(parent, text="Adding to Cloud (Main Database)", variable=self.enroll_interactive, style='Switch.TCheckbutton')
        chk.pack(anchor='w', pady=(0, 10))
        f = ttk.Frame(parent, style='Card.TFrame')
        f.pack(fill='x', pady=5)
        ttk.Label(f, text="Duplicate ID Policy:", style='CardLabel.TLabel').pack(side='left')
        ttk.Combobox(f, textvariable=self.on_exist_var, values=["Skip", "Replace"], state="readonly", width=12).pack(side='right')
        
        self.run_btn = ttk.Button(parent, text="Run Enrollment Process", style='Primary.TButton', command=self.run_enrollment)
        self.run_btn.pack(fill='x', pady=(15, 0), ipady=5)

    def _build_log_ui(self, parent):
        # Console Log Area
        self.log_text = ScrolledText(parent, height=8, font=("Consolas", 9), state='disabled', bg="#1e1e1e", fg="#00ff00")
        self.log_text.pack(fill='both', expand=True)

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def run_enrollment(self):
        enroll_arg = "true" if self.enroll_interactive.get() else "false"
        policy_arg = self.on_exist_var.get()
        
        self.run_btn.config(state='disabled')
        self.log("--- Starting Enrollment Subprocess ---\n")
        
        def run_thread():
            cmd = [sys.executable, "generate_face_embeddings.py", "--enroll", enroll_arg, "--on-exist", policy_arg]
            
            # CREATE_NO_WINDOW = 0x08000000 (Windows only) to hide CMD
            creation_flags = 0x08000000 if sys.platform == 'win32' else 0
            
            try:
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    # Drop text=True to handle bytes manually to avoid encoding crashes
                    text=False, 
                    creationflags=creation_flags
                )
                
                for line_bytes in iter(process.stdout.readline, b''):
                    if line_bytes:
                        # manually decode with error replacement
                        line = line_bytes.decode('utf-8', errors='replace')
                        # Schedule GUI update on main thread
                        self.master.after(0, self.log, line)
                
                process.stdout.close()
                return_code = process.wait()
                
                self.master.after(0, self.log, f"\n--- Process Finished with Code {return_code} ---\n")
            except Exception as e:
                self.master.after(0, self.log, f"Error: {e}\n")
            finally:
                 self.master.after(0, lambda: self.run_btn.config(state='normal'))

        threading.Thread(target=run_thread, daemon=True).start()

    def on_start(self):
        try:
            def get_time(h, m):
                if not (h.get().isdigit() and m.get().isdigit()): raise ValueError 
                return f"{int(h.get()):02d}:{int(m.get()):02d}"

            new_settings = {
                'start_time': get_time(self.start_h, self.start_m),
                'lateness_time': get_time(self.late_h, self.late_m),
                'absence_time': get_time(self.abs_h, self.abs_m)
            }
            save_settings(CONFIG_FILE, new_settings)
            self.master.destroy()
        except:
             messagebox.showerror("Invalid Input", "Time fields must be numeric (HH:MM).")

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeConfigApp(root)
    root.mainloop()
