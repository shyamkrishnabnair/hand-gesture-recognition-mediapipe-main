#customTkinter.v6.py
import sys, csv, copy, time, subprocess, logging
from collections import Counter, deque
import cv2 as cv
cv.setUseOptimized(True)
cv.setNumThreads(2)
import mediapipe as mp
from PIL import Image, ImageTk
import customtkinter as ctk

# Model imports
from model import KeyPointClassifier, PointHistoryClassifier
from utils.calculate import calc_bounding_rect, calc_landmark_list
from utils.pre_process import pre_process_landmark, pre_process_point_history
from utils.log import logging as custom_logging
from utils.draw import draw_info_text, draw_bounding_rect, draw_point_history, draw_info, draw_landmarks
from utils import MidiSoundPlayer, CTkTextboxHandler
from utils.frame_utils import preprocess_frame, detect_hands_and_classify, count_fingers
from utils.app_state import AppState as app_state

# Incremental logs (from your custom logging module)
log_file = custom_logging()
app_state.keypoint_classifier = KeyPointClassifier()
app_state.point_history_classifier = PointHistoryClassifier()
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f)]

# CustomTkinter setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Global vars
app_state.running = False
cap = None
app_state.point_history = deque(maxlen=16)
app_state.finger_gesture_history = deque(maxlen=16)
app_state.video_label = None  # placeholder globally
app_state.status_label = None

# Initialize MIDI sound player (no need for .sf2 or external tools)
app_state.player = MidiSoundPlayer()
app_state.player.set_instrument(0)  # Acoustic Grand Piano

app_state.app_logger = logging.getLogger(__name__)
app_state.app_logger.setLevel(logging.INFO)

def refresh():
    stop_camera()
    app.destroy()
    subprocess.Popen([sys.executable, 'customTkinter.v7.py'])
    app_state.app_logger.info("GUI refreshed (customTkinter.v7.py restarted)")
    log_file.close()

def quit_app():
    stop_camera()
    app.destroy()
    app_state.app_logger.info("GUI closed")
    log_file.close()

def update_timer():
    if app_state.running:
        elapsed = int(time.time() - app_state.start_time)
        app_state.status_label.configure(text=f"Status: Running ({elapsed}s)")
        app.after(1000, update_timer) # Schedule next update after 1 second

def start_camera():
    if app_state.running:
        app_state.app_logger.warning("Camera is already running.")
        return
    app_state.running = True
    app_state.start_time = time.time()
    global cap
    cap = cv.VideoCapture(0) # Open default webcam (index 0)
    if not cap.isOpened():
        app_state.app_logger.error("Failed to open webcam. Please check if it's connected and not in use.")
        app_state.running = False
        app_state.status_label.configure(text="Status: Error (Webcam not found)")
        return
    update_timer()
    update_frame(cap)
    app_state.app_logger.info("Camera started.")

def stop_camera():
    if not app_state.running:
        app_state.app_logger.info("Camera is already stopped.")
        return
    app_state.running = False
    if cap:
        cap.release()
        cap = None
    app_state.status_label.configure(text="Status: Stopped")
    app_state.app_logger.info("Camera stopped.")

def toggle_mute():
    app_state.player.muted = not app_state.player.muted
    app_state.mute_btn.configure(text="Unmute" if app_state.player.muted else "Mute")
    app_state.app_logger.info("Muted" if app_state.player.muted else "Unmuted")

# 2025-06-07 01:55:48,345 - ERROR - Error processing frame: cannot unpack non-iterable NoneType object
# Traceback (most recent call last):
#   File "e:\GithubRepo\hand-gesture-recognition-mediapipe-main\customTKinter.v7.py", line 114, in update_frame
#     hand_label, hand_sign_id, most_common_fg_id, landmark_list, hand_landmarks_xy = detect_hands_and_classify(rgb_image, hands, debug_image, app_state)
# TypeError: cannot unpack non-iterable NoneType object

def update_frame(cap):
    if not app_state.running or not cap:
        return
    frame_start_time = time.time()
    app_state.total_finger_count = 0
    
    try:
        rgb_image, debug_image = preprocess_frame(cap)

        hand_label, hand_sign_id, most_common_fg_id, landmark_list, hand_landmarks_xy = detect_hands_and_classify(rgb_image, hands, debug_image, app_state)

        count_fingers(hand_label, hand_landmarks_xy, hand_sign_id, most_common_fg_id, app_state)

        app_state.player.handle_note_playing(app_state.total_finger_count, app_state) 

    except Exception as e:
        import traceback
        error_info = traceback.format_exc()
        app_state.app_logger.error(f"Error processing frame: {e}\n{error_info}") # Log detailed error to UI 

    # Draw point history and FPS on the debug image
    debug_image = draw_point_history(debug_image, app_state.point_history)
    fps = int(1 / max(0.01, time.time() - frame_start_time)) # Calculate FPS, avoid division by zero
    debug_image = draw_info(debug_image, fps, 0, 0)

    # Convert OpenCV image to PhotoImage for CustomTkinter display
    img = Image.fromarray(cv.cvtColor(debug_image, cv.COLOR_BGR2RGB))
    frame_width = 720
    frame_height = 480
    resized_img = img.resize((frame_width, frame_height))  # PIL resize

    
    imgtk = ImageTk.PhotoImage(image=resized_img)
    app_state.video_label.imgtk = imgtk
    app_state.video_label.configure(image=imgtk)

    app_state.status_label.configure(text=f"Status: Running ({int(time.time() - app_state.start_time)}s)")
    app_state.video_label.after(10, update_frame) # Schedule the next frame update


app = ctk.CTk()
app.title("🤖 Hand Gesture Recognition")
app.geometry("1200x900")

title = ctk.CTkLabel(app, text="🤖 Hand Gesture Controller", font=("Segoe UI", 26, "bold"))
title.pack(pady=20)

instrument_label = ctk.CTkLabel(app, text="Instrument: Acoustic Grand Piano", font=("Segoe UI", 18))
instrument_label.pack(pady=5)

main_frame = ctk.CTkFrame(app)
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

# Left control panel
left_panel = ctk.CTkFrame(main_frame, width=120)
app_state.mute_btn = ctk.CTkButton(left_panel, text="Mute", command=toggle_mute)
app_state.mute_btn.pack(pady=20, padx=10)
left_panel.pack_propagate(False)  # Prevent it from resizing to contents
left_panel.pack(side="left", fill="y", padx=(0, 10))

# Center for camera
center_panel = ctk.CTkFrame(main_frame)
app_state.video_label = ctk.CTkLabel(center_panel, text="")
app_state.video_label.pack(padx=20, pady=10, fill="both", expand=True)
center_panel.pack(side="left", expand=True)

# Right panel (for future)
right_panel = ctk.CTkFrame(main_frame, width=120)
right_panel.pack(side="left", fill="y", padx=(10, 0))

# --- Add a logging area to the UI ---
log_frame = ctk.CTkFrame(app, fg_color="transparent")
log_frame.pack(padx=20, pady=10, fill="x", expand=False) # Pack above the buttons

log_label = ctk.CTkLabel(log_frame, text="Live Log:", font=("Segoe UI", 12, "bold"))
log_label.pack(side="left", padx=(0, 5), anchor="nw") # Label for the log area

log_textbox = ctk.CTkTextbox(log_frame, width=900, height=150, activate_scrollbars=True, wrap="word", font=("Consolas", 10))
log_textbox.pack(side="left", fill="x", expand=True) # The actual textbox for displaying logs

# --- Configure the standard logging module to output to the CTkTextbox ---
log_handler = CTkTextboxHandler(log_textbox)
# Define the format for log messages (timestamp - level - message)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
app_state.app_logger.addHandler(log_handler) # Add our custom handler to the application logger

button_frame = ctk.CTkFrame(app, fg_color="#2B2A45")
button_frame.pack(pady=20) # Pack below the log area

start_btn = ctk.CTkButton(button_frame, text="▶ Start Model", width=160, command=start_camera)
start_btn.grid(row=0, column=0, padx=10, pady=10)

stop_btn = ctk.CTkButton(button_frame, text="⏹ Stop Model", width=160, command=stop_camera)
stop_btn.grid(row=0, column=1, padx=10, pady=10)

refresh_btn = ctk.CTkButton(button_frame, text="🔁 Refresh", width=160, command=refresh)
refresh_btn.grid(row=1, column=0, padx=10, pady=10)

exit_btn = ctk.CTkButton(button_frame, text="❌ Exit", width=160, command=quit_app)
exit_btn.grid(row=1, column=1, padx=10, pady=10)

app_state.status_label = ctk.CTkLabel(app, text="Status: Idle", font=("Segoe UI", 14))
app_state.status_label.pack(pady=10)

def on_close():
    stop_camera()
    app_state.player.stop()
    app.destroy()
    try:
        if log_file:
            log_file.close()
    except AttributeError:
        pass # Ignore if log_file doesn't have a close method or is None

app.protocol("WM_DELETE_WINDOW", on_close) # Bind the on_close function to the window's close button
app.mainloop() # Start the CustomTkinter event loop