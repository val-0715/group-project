#!/usr/bin/env python3
import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EDVIN_SCRIPT = os.path.join(SCRIPT_DIR, "edvin.py")
PYTHON_EXEC = sys.executable


def display_available():
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


class EdvinGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Edvin Camera GUI")
        self.geometry("520x520")
        self.resizable(False, False)
        self.proc = None
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Camera index or device path:").grid(row=0, column=0, sticky=tk.W)
        self.camera_var = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self.camera_var, width=30).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(frame, text="Optional video file:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.input_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.input_var, width=30).grid(row=1, column=1, sticky=tk.W, pady=(10, 0))

        ttk.Label(frame, text="Capture backend:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.backend_var = tk.StringVar(value="auto")
        backend_combo = ttk.Combobox(frame, textvariable=self.backend_var, state="readonly", width=28)
        backend_combo["values"] = ["auto", "v4l2", "dshow", "avfoundation", "gstreamer"]
        backend_combo.grid(row=2, column=1, sticky=tk.W, pady=(10, 0))

        # Default to showing preview windows for convenience on GUI machines
        self.show_gui_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Show preview windows", variable=self.show_gui_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        ttk.Label(frame, text="MJPEG port (browser view):").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        # Default empty port (no MJPEG) — user can set 8080 or similar
        self.mjpeg_port_var = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.mjpeg_port_var, width=10).grid(row=4, column=1, sticky=tk.W, pady=(10, 0))

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=(18, 10), sticky=tk.W)

        self.list_button = ttk.Button(button_frame, text="List Cameras", command=self.list_cameras)
        self.list_button.grid(row=0, column=0, padx=(0, 8))
        self.run_button = ttk.Button(button_frame, text="Run Edvin", command=self.run_edvin)
        self.run_button.grid(row=0, column=1, padx=(0, 8))
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_edvin, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=2)

        ttk.Label(frame, text="Command output:").grid(row=6, column=0, columnspan=2, sticky=tk.W)
        self.log_box = scrolledtext.ScrolledText(frame, width=60, height=16, state=tk.DISABLED)
        self.log_box.grid(row=7, column=0, columnspan=2, pady=(6, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.status_var).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

    def append_log(self, text):
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def list_cameras(self):
        self.append_log("Listing cameras...")
        try:
            result = subprocess.run([PYTHON_EXEC, EDVIN_SCRIPT, "--list-cameras"], capture_output=True, text=True, check=False)
            output = result.stdout.strip() or result.stderr.strip() or "No output."
            self.append_log(output)
            messagebox.showinfo("Cameras", output)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to list cameras:\n{exc}")

    def build_args(self):
        args = [PYTHON_EXEC, EDVIN_SCRIPT]
        camera_value = self.camera_var.get().strip()
        input_value = self.input_var.get().strip()
        if input_value:
            args += ["--input", input_value]
        elif camera_value:
            args += ["--camera", camera_value]
        backend = self.backend_var.get().strip()
        if backend and backend != "auto":
            args += ["--backend", backend]
        if self.show_gui_var.get():
            args.append("--show-gui")
        port = self.mjpeg_port_var.get().strip()
        if port and port != "0":
            args += ["--mjpeg-port", port]
        return args

    def run_edvin(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Already running", "The Edvin process is already running.")
            return

        args = self.build_args()
        self.append_log("Starting Edvin with: " + " ".join(args))
        try:
            self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as exc:
            messagebox.showerror("Failed to start", f"Could not start Edvin:\n{exc}")
            return

        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set("Running...")
        threading.Thread(target=self.poll_output, daemon=True).start()
        threading.Thread(target=self.wait_process, daemon=True).start()

    def poll_output(self):
        if not self.proc or self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            self.append_log(line.rstrip())

    def wait_process(self):
        if not self.proc:
            return
        self.proc.wait()
        self.append_log(f"Process exited with code {self.proc.returncode}.")
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Stopped")

    def stop_edvin(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.append_log("Terminating process...")
            self.proc.wait(timeout=5)
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Stopped")

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            if messagebox.askyesno("Quit", "Edvin is still running. Stop it and quit?"):
                self.stop_edvin()
            else:
                return
        self.destroy()

if __name__ == '__main__':
    if not display_available():
        print("No GUI display available in this environment. Use edvin.py directly with --mjpeg-port PORT or run edvin_gui.py on a machine with a display.")
        sys.exit(1)
    app = EdvinGui()
    app.mainloop()
