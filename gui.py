import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import json
import os
import subprocess
import sys
import threading

CONFIG_FILE = 'config.json'
DEFAULT_TIMES = {
    'lateness_time': "07:00",
    'absence_time': "10:00",
    'start_time': "06:00",
    'enable_recording': True,
    'camera_url': 'rtsp://admin:1234qwer@@192.168.1.18:554/Streaming/Channels/102?tcp'
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
        self.master.title("Smart Scan")
        self.master.geometry("900x650")
        self.master.resizable(False, False)
        self.active_tab = "Settings"
        self.running_recognition = False
        self.stop_event = None
        self.recog_thread = None
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.colors = {
            'bg': '#f8fafc',         
            'white': '#ffffff',
            'primary': '#4f46e5',    
            'primary_dark': '#4338ca',
            'success': '#10b981',    
            'danger': '#ef4444',     
            'danger_dark': '#dc2626',
            'text': '#0f172a',       
            'text_light': '#64748b', 
            'border': '#e2e8f0'      
        }
        self.master.configure(bg=self.colors['bg'])
        self.style.configure('TFrame', background=self.colors['bg'])
        self.style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['text'], font=('Outfit', 10))
        self.style.configure('Header.TLabel', font=('Outfit', 20, 'bold'), foreground=self.colors['text'])
        self.style.configure('Card.TFrame', background=self.colors['white'], relief='flat')
        self.style.configure('CardLabel.TLabel', background=self.colors['white'], font=('Outfit', 9, 'bold'))
        self.style.configure('Primary.TButton', font=('Outfit', 10, 'bold'), borderwidth=0, focuscolor='none')
        self.style.map('Primary.TButton', background=[('active', self.colors['primary_dark']), ('!disabled', self.colors['primary'])], foreground=[('!disabled', 'white')])

        self.style.configure('Success.TButton', font=('Outfit', 11, 'bold'), borderwidth=0, focuscolor='none')
        self.style.map('Success.TButton', background=[('active', '#059669'), ('!disabled', self.colors['success'])], foreground=[('!disabled', 'white')])

        self.style.configure('Danger.TButton', font=('Outfit', 11, 'bold'), borderwidth=0, focuscolor='none')
        self.style.map('Danger.TButton', background=[('active', self.colors['danger_dark']), ('!disabled', self.colors['danger'])], foreground=[('!disabled', 'white')])
        self.style.configure('Small.Primary.TButton', font=('Outfit', 8, 'bold'))

        self.current_settings = load_settings(CONFIG_FILE)
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
        self.record_var = tk.BooleanVar(value=self.current_settings.get('enable_recording', True))
        self.camera_url_var = tk.StringVar(value=self.current_settings.get('camera_url', DEFAULT_TIMES['camera_url']))

        self.create_widgets()

    def create_widgets(self):
        self.main_layout = ttk.Frame(self.master)
        self.main_layout.pack(fill='both', expand=True)

        self.sidebar = tk.Frame(self.main_layout, bg='#1e293b', width=220) 
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        brand_frame = tk.Frame(self.sidebar, bg='#1e293b', pady=30)
        brand_frame.pack(fill='x')
        tk.Label(brand_frame, text="Smart Scan", font=('Outfit', 18, 'bold'), fg='white', bg='#1e293b').pack()
        tk.Label(brand_frame, text="Facial Recognition System", font=('Outfit', 8, 'bold'), fg='#94a3b8', bg='#1e293b').pack()

        self.nav_items = {
            "Settings": "⚙  Configuration",
            "Database": "📂  Enrollment",
            "Logs": "📝  System Logs"
        }
        self.nav_buttons = {}
        for key, label in self.nav_items.items():
            btn = tk.Button(self.sidebar, text=label, font=('Outfit', 10, 'bold'),
                          fg='#cbd5e1', bg='#1e293b', relief='flat', anchor='w', padx=25, pady=12,
                          activebackground='#334155', activeforeground='white', cursor='hand2',
                          command=lambda k=key: self.switch_tab(k))
            btn.pack(fill='x')
            self.nav_buttons[key] = btn

        action_frame = tk.Frame(self.sidebar, bg='#1e293b', pady=20)
        action_frame.pack(side='bottom', fill='x')

        self.start_btn = tk.Button(action_frame, text="▶  START SYSTEM", font=('Outfit', 10, 'bold'),
                                 fg='white', bg='#10b981', relief='flat', pady=12, cursor='hand2',
                                 command=self.toggle_recognition)
        self.start_btn.pack(fill='x', padx=20, pady=5)

        self.exit_btn = tk.Button(action_frame, text="✕  EXIT", font=('Outfit', 10, 'bold'),
                                fg='white', bg='#ef4444', relief='flat', pady=10, cursor='hand2',
                                command=self.on_exit)
        self.exit_btn.pack(fill='x', padx=20, pady=5)

        self.content_area = tk.Frame(self.main_layout, bg='#f8fafc') 
        self.content_area.pack(side='right', fill='both', expand=True)

        self.switch_tab("Settings")

    def switch_tab(self, tab_key):
        for key, btn in self.nav_buttons.items():
            if key == tab_key:
                btn.config(bg='#334155', fg='white')
            else:
                btn.config(bg='#1e293b', fg='#cbd5e1')

        for widget in self.content_area.winfo_children():
            widget.destroy()

        header = tk.Label(self.content_area, text=self.nav_items[tab_key].split('  ')[-1], 
                         font=('Outfit', 24, 'bold'), fg='#0f172a', bg='#f8fafc', padx=40, pady=30)
        header.pack(anchor='w')

        scroll_frame = ttk.Frame(self.content_area, padding=(40, 0, 40, 40))
        scroll_frame.pack(fill='both', expand=True)

        if tab_key == "Settings":
            self._build_scrollable_container(scroll_frame, self._build_settings_page)
        elif tab_key == "Database":
            self._build_database_page(scroll_frame)
        elif tab_key == "Logs":
             self._build_log_page(scroll_frame)

    def _build_scrollable_container(self, parent, content_fn):
        canvas = tk.Canvas(parent, bg='#f8fafc', highlightthickness=0)
        scrollable_frame = tk.Frame(canvas, bg='#f8fafc')
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=600)

        canvas.pack(side="left", fill="both", expand=True)
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        content_fn(scrollable_frame)

    def _build_settings_page(self, parent):
        self._create_card(parent, "Attendance Schedule", self._build_time_ui)
        self._create_card(parent, "Camera Configuration", self._build_camera_ui)
        self._create_card(parent, "Session Preferences", self._build_session_ui)

    def _build_database_page(self, parent):
        self._create_card(parent, "Student Enrollment", self._build_enroll_ui)
    def _build_log_page(self, parent):
        self.log_text = ScrolledText(parent, font=("Consolas", 10), state='disabled', bg="#0f172a", fg="#10b981", borderwidth=0)
        self.log_text.pack(fill='both', expand=True, pady=(0, 20))
        self.log("--- Monitoring System Ready ---\n")

    def _create_card(self, parent, title, build_fn):
        card = ttk.Frame(parent, style='Card.TFrame', padding=25)
        card.pack(fill='x', pady=(0, 20))
        tk.Label(card, text=title.upper(), font=('Outfit', 9, 'bold'), fg=self.colors['text_light'], bg='white').pack(anchor='w', pady=(0, 15))
        build_fn(card)

    def _build_time_ui(self, parent):
        grid = tk.Frame(parent, bg='white')
        grid.pack(fill='x')
        self._time_row(grid, "Start Session", self.start_h, self.start_m, 0)
        self._time_row(grid, "Late Threshold", self.late_h, self.late_m, 1)
        self._time_row(grid, "Absence Marker", self.abs_h, self.abs_m, 2)

    def _time_row(self, parent, label, h, m, row):
        tk.Label(parent, text=label, font=('Outfit', 10), fg='#475569', bg='white').grid(row=row, column=0, sticky='w', pady=10)
        entry_frame = tk.Frame(parent, bg='white')
        entry_frame.grid(row=row, column=1, sticky='e', padx=(150, 0))
        tk.Entry(entry_frame, textvariable=h, font=('Outfit', 11), width=4, justify='center', relief='flat', highlightbackground='#e2e8f0', highlightthickness=1).pack(side='left')
        tk.Label(entry_frame, text=":", font=('Outfit', 12, 'bold'), bg='white', padx=5).pack(side='left')
        tk.Entry(entry_frame, textvariable=m, font=('Outfit', 11), width=4, justify='center', relief='flat', highlightbackground='#e2e8f0', highlightthickness=1).pack(side='left')

    def _build_camera_ui(self, parent):
        tk.Label(parent, text="Stream URL:", font=('Outfit', 10), fg='#475569', bg='white').pack(anchor='w', pady=(0, 5))
        row = tk.Frame(parent, bg='white')
        row.pack(fill='x')
        entry = tk.Entry(row, textvariable=self.camera_url_var, font=('Consolas', 10), relief='flat', highlightbackground='#e2e8f0', highlightthickness=1)
        entry.pack(side='left', fill='x', expand=True, ipady=8, padx=(0, 10))
        tk.Button(row, text="💾 SAVE", font=('Outfit', 9, 'bold'), fg='white', bg=self.colors['primary'], 
                 relief='flat', padx=15, pady=5, cursor='hand2', command=self.quick_save_url).pack(side='right')

    def _build_session_ui(self, parent):
        chk = tk.Checkbutton(parent, text="Enable Session Video Recording", variable=self.record_var, 
                           font=('Outfit', 10), bg='white', activebackground='white', cursor='hand2')
        chk.pack(anchor='w')

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
        pass

    def log(self, message):
        def _update():
            try:
                if hasattr(self, 'log_text') and self.log_text.winfo_exists():
                    self.log_text.config(state='normal')
                    self.log_text.insert(tk.END, str(message))
                    self.log_text.see(tk.END)
                    self.log_text.config(state='disabled')
            except:
                pass
        self.master.after(0, _update)

    def run_enrollment(self):
        enroll_arg = "true" if self.enroll_interactive.get() else "false"
        policy_arg = self.on_exist_var.get()
        self.run_btn.config(state='disabled')
        self.log("--- Starting Enrollment Subprocess ---\n")
        def run_thread():
            cmd = [sys.executable, "generate_face_embeddings.py", "--enroll", enroll_arg, "--on-exist", policy_arg]
            creation_flags = 0x08000000 if sys.platform == 'win32' else 0
            try:
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=False, 
                    creationflags=creation_flags
                )
                for line_bytes in iter(process.stdout.readline, b''):
                    if line_bytes:
                        line = line_bytes.decode('utf-8', errors='replace')
                        self.master.after(0, self.log, line)
                process.stdout.close()
                return_code = process.wait()
                self.master.after(0, self.log, f"\n--- Process Finished with Code {return_code} ---\n")
            except Exception as e:
                self.master.after(0, self.log, f"Error: {e}\n")
            finally:
                 self.master.after(0, lambda: self.run_btn.config(state='normal'))

        threading.Thread(target=run_thread, daemon=True).start()

    def quick_save_url(self):
        new_url = self.camera_url_var.get().strip()
        self.current_settings['camera_url'] = new_url
        save_settings(CONFIG_FILE, self.current_settings)
        messagebox.showinfo("Success", "Camera URL updated and saved.")

    def on_start(self):
        from face_recognition import real_time_face_recognition
        try:
            def get_time(h, m):
                if not (h.get().isdigit() and m.get().isdigit()): raise ValueError 
                return f"{int(h.get()):02d}:{int(m.get()):02d}"

            new_settings = {
                'start_time': get_time(self.start_h, self.start_m),
                'lateness_time': get_time(self.late_h, self.late_m),
                'absence_time': get_time(self.abs_h, self.abs_m),
                'enable_recording': self.record_var.get(),
                'camera_url': self.camera_url_var.get().strip()
            }
            save_settings(CONFIG_FILE, new_settings)
            self.stop_event = threading.Event()
            self.recog_thread = threading.Thread(
                target=real_time_face_recognition, 
                args=(self.stop_event, self.log),
                daemon=True
            )
            self.recog_thread.start()
            return True
        except Exception as e:
             messagebox.showerror("Invalid Input", f"Could not start: {e}")
             return False

    def toggle_recognition(self):
        if not self.running_recognition:
            if self.on_start():
                self.running_recognition = True
                self.lock_ui()
                self.switch_tab("Logs")
        else:
            if self.stop_event:
                self.stop_event.set()
                self.log("\n[SYSTEM] Stopping recognition...\n")
                self.running_recognition = False
                self.unlock_ui()

    def lock_ui(self):
        self.start_btn.config(text="■  STOP RECOGNITION", bg='#fbbf24', fg='#92400e')
        for key, btn in self.nav_buttons.items():
            if key != "Logs":
                btn.config(state='disabled', cursor='arrow')

    def unlock_ui(self):
        self.start_btn.config(text="▶  START SYSTEM", bg='#10b981', fg='white')
        for key, btn in self.nav_buttons.items():
            btn.config(state='normal', cursor='hand2')

    def on_exit(self):
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeConfigApp(root)
    root.mainloop()