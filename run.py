import tkinter as tk
import subprocess
import time
from tkinter import messagebox
import sys

process = None
start_time = None
running = False

def update_timer():
    if running:
        elapsed = int(time.time() - start_time)
        status_label.config(text=f"Status: Running ({elapsed}s)")
        root.after(1000, update_timer)

def start_script():
    global process, start_time, running
    if process is None or process.poll() is not None:
        process = subprocess.Popen(['python', 'app.py'])
        start_time = time.time()
        running = True
        update_timer()
        print("Script started")
    else:
        print("Script already running")

def stop_script():
    global process, running
    if process and process.poll() is None:
        process.terminate()
        process.wait()
        process = None
        running = False
        status_label.config(text="Status: Stopped")
        print("Script stopped")
    else:
        print("No running script to stop")

def refresh_gui():
    stop_script()  # Stop the running model (app.py)
    # Restart the 'run.py' script (without stopping app.py)
    root.destroy()  # Close the current window
    subprocess.Popen([sys.executable, 'run.py'])  # Reopen the 'run.py' script
    print("GUI refreshed (run.py restarted)")

def refresh_script():
    
    stop_script()
    start_script()
    print("Script refreshed")

def on_close():
    if process and process.poll() is None:
        if messagebox.askyesno("Confirm Exit", "App is still running. Stop app.py before closing?\n\nDo you want to stop app.py and close?"):
            stop_script()
            root.destroy()
        else:
            return  # cancel closing
    else:
        root.destroy()

# --- Tkinter UI ---
root = tk.Tk()
root.title("App Controller")
root.geometry("300x250")

title_label = tk.Label(root, text="Hand Gesture Controller", font=("Arial", 14))
title_label.pack(pady=10)

start_btn = tk.Button(root, text="Start Model", width=20, command=start_script)
start_btn.pack(pady=5)

stop_btn = tk.Button(root, text="Stop Model", width=20, command=stop_script)
stop_btn.pack(pady=5)

refresh_btn = tk.Button(root, text="Refresh Model", width=20, command=refresh_script)
refresh_btn.pack(pady=5)


refresh_gui_btn = tk.Button(root, text="Refresh GUI", width=20, command=refresh_gui)
refresh_gui_btn.pack(pady=5)

status_label = tk.Label(root, text="Status: Idle", font=("Arial", 10))
status_label.pack(pady=10)

# catch window close
root.protocol("WM_DELETE_WINDOW", on_close)


root.mainloop()
