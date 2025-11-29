#main.py
import sys, copy, time, subprocess, logging
from collections import deque
import cv2 as cv
cv.setUseOptimized(True)
cv.setNumThreads(2)
import mediapipe as mp

import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from PIL import Image, ImageTk
import customtkinter as ctk

# Model imports
from utils.calculate import calc_bounding_rect, calc_landmark_list
from utils.pre_process import pre_process_landmark, pre_process_point_history
from utils.log import logging as custom_logging
from utils.log import log_note_play
from utils.draw import draw_info_text, draw_bounding_rect, draw_point_history, draw_info, draw_landmarks
from utils import MidiSoundPlayer, CTkTextboxHandler
from utils import NotationPanel

log_file = custom_logging()

# CustomTkinter setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Mediapipe setup
mp_hands = None
mp_solutions = getattr(mp, "solutions", None)
if mp_solutions is not None:
    mp_hands = getattr(mp_solutions, "hands", None)

if mp_hands is None:
    try:
        from mediapipe.python.solutions import hands as mp_hands  # type: ignore
    except Exception as e:
        raise ImportError("Could not import MediaPipe 'hands' module") from e

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Global vars
running = False
cap = None
start_time = None
pinch_mode = False
point_history = deque(maxlen=16)
finger_gesture_history = deque(maxlen=16)
last_finger_count = 0
last_note_time = 0
note_cooldown = 0.5  # seconds between two notes
last_volume_level = 50
is_muted = False
pinch_start_x = 0
left_hand_pinch_state = False
current_instrument_index = 0
instrument_scroll_mode = False
instrument_scroll_start_x = 0
video_label = None  # placeholder globally 
video_label_img = None  # keep a reference to the current PhotoImage to prevent GC and avoid adding attributes to CTkLabel
note_mapping = {
    1: 60,  # C4
    2: 62,  # D4
    3: 64,  # E4
    4: 65,  # F4
    5: 67,  # G4
    6: 69,  # A4
    7: 71,  # B4
    8: 72,  # C5
    9: 74,  # D5
    10: 76  # E5
}
instrument_names = [
    "Acoustic Grand Piano 🎹", "Bright Piano 🎹", "Electric Piano 🎹", "Harpsichord 🎹", "Drawbar Organ 🎹", "Acoustic Guitar 🎸", "Overdriven Guitar 🎸", "Bass Guitar 🎸", "Violin 🎻", "String Ensemble 🎻", "Trumpet 🎺", "Saxophone 🎷", "Flute 🎶", "Synth Lead 🎹", "Synth Pad 🎹"
]
instrument_ids = [0, 1, 4, 6, 16, 24, 29, 33, 40, 48, 56, 65, 73, 80, 88]
log_finger_count = 0

RECORDINGS_DIR = "exports/json"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Initialize MIDI sound player (no need for .sf2 or external tools)
player = MidiSoundPlayer()
player.set_instrument(0)  # Acoustic Grand Piano

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)

def refresh():
    stop_camera()
    app.destroy()
    subprocess.Popen([sys.executable, 'main.py'])
    app_logger.info("GUI refreshed (main.py restarted)")
    log_file.close()

def quit_app():
    stop_camera()
    app.destroy()
    app_logger.info("GUI closed")
    log_file.close()

def update_timer():
    if running:
        if start_time is None:
            elapsed = 0
        else:
            elapsed = int(time.time() - start_time)
        status_label.configure(text=f"Status: Running ({elapsed}s)")
        app.after(1000, update_timer) # Schedule next update after 1 second

def start_camera():
    global running, cap, start_time
    if running:
        app_logger.warning("Camera is already running.")
        return
    running = True
    start_time = time.time()
    cap = cv.VideoCapture(0) # Open default webcam (index 0)
    if not cap.isOpened():
        app_logger.error("Failed to open webcam. Please check if it's connected and not in use.")
        running = False
        status_label.configure(text="Status: Error (Webcam not found)")
        return
    update_timer()
    update_frame()
    update_recording_list()
    app_logger.info("Camera started.")

def stop_camera():
    global running, cap
    if not running:
        app_logger.info("Camera is already stopped.")
        return
    running = False
    if cap:
        cap.release()
        cap = None
    status_label.configure(text="Status: Stopped")
    app_logger.info("Camera stopped.")

def toggle_mute():
    player.muted = not player.muted
    mute_btn.configure(text="Unmute" if player.muted else "Mute")
    app_logger.info("Muted" if player.muted else "Unmuted")

def update_frame():
    if not running or not cap:
        return

    frame_start_time = time.time()

    ret, image = cap.read()
    if not ret:
        app_logger.error("Failed to read frame from camera.")
        return

    image = cv.flip(image, 1) 
    debug_image = copy.deepcopy(image) # Create a copy for drawing annotations
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Convert to RGB for MediaPipe
    results = hands.process(image) # Process the image for hand landmarks

    total_finger_count = 0
    try:
        if getattr(results, "multi_hand_landmarks", None) is not None:
            multi_landmarks = getattr(results, "multi_hand_landmarks", None)
            multi_handedness = getattr(results, "multi_handedness", None)
            if multi_landmarks is not None and multi_handedness is not None:
                for hand_landmarks, handedness in zip(multi_landmarks, multi_handedness):
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Classify finger gesture (dynamic movement)
                    finger_gesture_id = 0

                    finger_gesture_history.append(finger_gesture_id)

                    global last_volume_level, pinch_mode, pinch_start_x, is_muted, left_hand_pinch_state
                    
                    hand_label = handedness.classification[0].label # "Left" or "Right"
                    hand_landmarks_xy = [[lm.x, lm.y] for lm in hand_landmarks.landmark] # Normalized coordinates

                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]

                    h, w, _ = debug_image.shape
                    thumb_px = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    index_px = int(index_tip.x * w), int(index_tip.y * h)

                    # Draw line and small circle between thumb and index finger tips
                    center_px = ((thumb_px[0] + index_px[0]) // 2, (thumb_px[1] + index_px[1]) // 2)
                    cv.line(debug_image, thumb_px, index_px, (255, 255, 255), 2)
                    cv.circle(debug_image, center_px, 10, (0, 255, 0), 2)
                    
                    # Calculate Euclidean distance between thumb and index tips
                    dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

                    # Detect pinch based on distance threshold
                    is_pinch = dist < 0.07 # Threshold for pinch detection
                    screen_width = debug_image.shape[1]
                    drag_threshold = screen_width * 0.2  # 20% of screen width
                    # Left Hand: Mute/Unmute Toggle

                    if 'pinch_mode' not in globals():
                        pinch_mode = False
                        pinch_start_x = 0

                    global instrument_scroll_mode, instrument_scroll_start_x, instrument_ids, instrument_names, current_instrument_index
                    
                    if hand_label == "Left":
                        if is_pinch:
                            cv.circle(debug_image, center_px, 15, (255, 255, 0), 3)
                            if not instrument_scroll_mode:
                                instrument_scroll_mode = True
                                instrument_scroll_start_x = center_px[0]
                            else:
                                print(f"Instrument scroll mode active, start X: {instrument_scroll_start_x}, center X: {center_px[0]}")
                                delta_x = center_px[0] - instrument_scroll_start_x

                                if abs(delta_x) >= drag_threshold:
                                    if delta_x > 0:
                                        new_index = (current_instrument_index + 1) % len(instrument_ids)
                                    else:
                                        new_index = (current_instrument_index - 1) % len(instrument_ids)
                                    # if fire_once("instrument_scroll", 0.5):
                                    set_instrument_by_index(new_index, confirm=True)
                                    instrument_scroll_start_x = center_px[0] 
                        else:
                            instrument_scroll_mode = False
                            
                    # Right Hand: Volume Control===
                    elif hand_label == "Right":
                        if is_pinch:
                            cv.circle(debug_image, center_px, 15, (0, 255, 0), 3) # Green circle

                            if not pinch_mode:
                                pinch_mode = True
                                pinch_start_x = center_px[0] # Record start X for horizontal movement
                            else:
                                if not player.muted:
                                    delta_x = center_px[0] - pinch_start_x
                                    if abs(delta_x) > drag_threshold/20:

                                        # Volume sensitivity (drag 1% screen = 1 step)
                                        volume_delta = delta_x / (screen_width * 0.01)
                                        volume_delta = max(-10, min(10, volume_delta))  # Cap delta for control

                                        # Apply new volume
                                        new_volume = player.volume + (volume_delta / 100)
                                        player.volume = max(0.0, min(1.0, new_volume))

                                        # Sync last_volume_level with current MIDI player volume
                                        last_volume_level = int(player.volume * 100)

                                        app_logger.debug(f"Volume: {last_volume_level}%")

                                # Always update start position for next drag, even if muted
                                pinch_start_x = center_px[0]

                            # Draw volume bar on debug image
                            bar_x, bar_y = 10, 85
                            bar_width, bar_height = 150, 10
                            filled_width = int(bar_width * (last_volume_level / 100))
                            cv.rectangle(debug_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1) # Background
                            cv.rectangle(debug_image, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1) # Foreground
                            cv.putText(debug_image, f"Volume: {last_volume_level}%", (bar_x, bar_y - 10),
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                        else:
                            cv.circle(debug_image, center_px, 15, (0, 0, 255), 2) # Blue circle when not pinching
                            pinch_mode = False # Reset pinch mode 
                    
                    if player.volume < 0.01 or player.muted:
                        cv.putText(debug_image, "Muted", (debug_image.shape[1] - 50, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 0, cv.LINE_AA)
                        mute_btn.configure(text="Unmute" if player.muted else "Mute")
                        # if fire_once("mute_log", 2.0):
                        # app_logger.info("Muted")
                        app_logger.info("Muted" if player.muted else "Unmuted")

                    # Finger Counting Logic
                    finger_count = 0
                    # Thumb (landmark 4 vs 3): Check horizontal position relative to its base
                    if (hand_label == "Left" and hand_landmarks_xy[4][0] > hand_landmarks_xy[3][0]) or \
                    (hand_label == "Right" and hand_landmarks_xy[4][0] < hand_landmarks_xy[3][0]):
                        finger_count += 1
                    # Other fingers (tips vs PIP joints): Check vertical position

                    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                        if hand_landmarks_xy[tip][1] < hand_landmarks_xy[pip][1]:
                            finger_count += 1
                    total_finger_count += finger_count # Accumulate finger count from all hands

                    # Log detected hand info to the UI
                    global log_finger_count
                    if log_finger_count != finger_count:
                        app_logger.info(f"Hand: {hand_label}, Fingers: {finger_count}")
                        log_finger_count = finger_count

                    # Draw annotations on the debug image
                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        finger_count,
                    )
                    
                    current_time = time.time()
                    global last_finger_count, last_note_time
                    if (
                        total_finger_count in note_mapping 
                        and 
                        (total_finger_count != last_finger_count or current_time - last_note_time > note_cooldown)
                        ):
                        note = note_mapping[total_finger_count]
                        try:
                            # --- play note (live) only when playback is NOT running ---
                            if not getattr(recording_panel, "is_playing_back", False):
                                # playback is NOT running, so play live note
                                # if fire_once(f"note_{note}", 0.3): # remove this check if the playback acts bad
                                player.play_note(note, duration=10)
                                notation_panel.add_gesture(total_finger_count)
                                log_note_play(note, total_finger_count, current_instrument_index)
                                record_gesture(total_finger_count)
                                
                                # event_time = time.time() - getattr(recording_panel, "timeline_start_time")
                                # event = {
                                # "gesture": total_finger_count,
                                # "time": event_time,
                                # "instrument": current_instrument_index
                                # }
                                # recording_panel.timeline_events.append(event)
                                # add_timed_event(event, recording_panel.timeline_start_time)
                                # 💾 5. If user is recording, ALSO append to recording_data
                                # if getattr(recording_panel, "is_recording", False):
                                    # recording_panel.recording_data.append(event)
                            else:
                                # playback *is* running, skip live note
                                print("Skipping live note because playback is running!")
                            last_finger_count = total_finger_count
                            last_note_time = current_time
                        except Exception as e:
                            print(f"Sound error: {e}")
        else:
            point_history.append([0, 0]) # Append dummy point if no hands detected
            app_logger.debug("No hands detected in frame.")
            player.note_cooldowns.clear()  # Reset if no hands detected

    except Exception as e:
        import traceback
        error_info = traceback.format_exc()
        app_logger.error(f"Error processing frame: {e}\n{error_info}") # Log detailed error to UI

    # Draw point history and FPS on the debug image
    debug_image = draw_point_history(debug_image, point_history)
    fps = int(1 / max(0.01, time.time() - frame_start_time)) # Calculate FPS, avoid division by zero
    debug_image = draw_info(debug_image, fps, 0, 0)

    # Convert OpenCV image to PhotoImage for CustomTkinter display
    img = Image.fromarray(cv.cvtColor(debug_image, cv.COLOR_BGR2RGB))
    frame_width = 720
    frame_height = 480
    resized_img = img.resize((frame_width, frame_height))  # PIL resize
    global video_label, video_label_img
    imgtk = ImageTk.PhotoImage(image=resized_img)
    # store reference in a module-level variable to prevent garbage collection
    # and avoid assigning arbitrary attributes to CTkLabel (static checkers complain)
    video_label_img = imgtk
    if video_label is not None:
        # Only update the UI widget if it exists
        video_label.configure(image=video_label_img)
    else:
        app_logger.warning("video_label is None; skipping image update.")
    
    if globals().get("status_label") is not None:
        if start_time is None:
            elapsed = 0
        else:
            elapsed = int(time.time() - start_time)
        status_label.configure(text=f"Status: Running ({elapsed}s)")
    else:
        app_logger.debug("status_label not yet initialized; skipping status update.")
    
    # Schedule the next frame update using the label if available, otherwise fallback to app.after
    if video_label is not None:
        video_label.after(10, update_frame)  # Schedule the next frame update
    else:
        try:
            app.after(10, update_frame)
        except Exception:
            pass

"""
UI BASED ON CUSTOM TKINTER
"""

app = ctk.CTk()
app.title("🤖 Hand Gesture Recognition")
app.geometry("1200x900+375+30")

# ================= Row 1 ============================
title = ctk.CTkLabel(app, text="🤖 Hand Gesture Controller", font=("Segoe UI", 26, "bold")) 
title.pack(pady=20)

# ================== Row 2: MAIN ROW ==================
main_row = ctk.CTkFrame(app)
main_row.pack(fill="x", padx=15, pady=10)
main_row.configure(height=520)
main_row.pack_propagate(False)

main_row.grid_columnconfigure(0, weight=0)   # Logs
main_row.grid_columnconfigure(1, weight=1)   # Instrument Selector
main_row.grid_columnconfigure(2, weight=3)   # Camera
main_row.grid_columnconfigure(3, weight=2)   # Recording
main_row.grid_columnconfigure(4, weight=2)   # Recordings list

main_row.grid_rowconfigure(0, weight=1)

# ================== Row 2, Col 1: LEFT PANEL(log and mute) ==================
left_panel = ctk.CTkFrame(main_row)
left_panel.grid(row=0, column=0, sticky="nsew", padx=5)

# Left control panel
mute_btn = ctk.CTkButton(left_panel, text="Mute", command=toggle_mute)
mute_btn.pack(pady=20, padx=10)

# --- Add a logging area to the left panel (second column after mute) ---
log_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

log_label = ctk.CTkLabel(log_frame, text="Live Log:", font=("Segoe UI", 12, "bold"))
log_label.pack(anchor="nw", padx=(0, 5), pady=(0, 5))

log_textbox = ctk.CTkTextbox(
    log_frame, 
    # width=260,
    height=400,  # adjust width/height to taste
    activate_scrollbars=True, wrap="word", font=("Consolas", 10)
)
log_textbox.pack(fill="both", expand=True)

# --- Configure the standard logging module ---
log_handler = CTkTextboxHandler(log_textbox)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
app_logger.addHandler(log_handler)

# ================== Row 2, Col 2: INSTRUMENT PANEL ================
instrument_panel = ctk.CTkFrame(main_row, width=280)
instrument_panel.grid(row=0, column=1, sticky="ns", padx=5)
instrument_panel.pack_propagate(False)

instrument_title = ctk.CTkLabel(
    instrument_panel,
    text="🎛 Instruments",
    font=("Segoe UI", 16, "bold")
)
instrument_title.pack(pady=(1, 0))

# Scrollable container
instrument_scroll = ctk.CTkScrollableFrame(
    instrument_panel,
    width=260,
    height=420
)
instrument_scroll.pack(padx=5, pady=10)

instrument_buttons = []

# ================== SHARED STATE UPDATE FUNCTION ==================
def set_instrument_by_index(index, confirm=True):
    global current_instrument_index

    current_instrument_index = index
    new_instrument = instrument_ids[index]

    player.set_instrument(new_instrument)
    if confirm and not getattr(recording_panel, "is_playing_back", False):
        player.play_note(60, duration=0.9, ignore_cooldown=True)

    setattr(recording_panel, "current_instrument", new_instrument)

    app_logger.info(f"Instrument: {instrument_names[index]}")

    update_instrument_ui()

# ================== UI HIGHLIGHT UPDATE ==================
def update_instrument_ui():
    for i, btn in enumerate(instrument_buttons):
        if i == current_instrument_index:
            btn.configure(
                fg_color="#00ffaa",   # highlighted
                text_color="black"
            )
        else:
            btn.configure(
                fg_color="transparent",
                text_color="white"
            )

# ================== CREATE BUTTON LIST ==================
for i, name in enumerate(instrument_names):
    btn = ctk.CTkButton(
        instrument_scroll,
        text=name,
        anchor="w",
        width=240,
        command=lambda i=i: set_instrument_by_index(i)
    )
    btn.pack(pady=3, padx=5)
    instrument_buttons.append(btn)

# Initial highlight
update_instrument_ui()

# ================== Row 2, Col 3: CAMERA ==================
camera_panel = ctk.CTkFrame(main_row)
camera_panel.grid(row=0, column=2, sticky="nsew", padx=5)
camera_panel.configure(height=500)
camera_panel.pack_propagate(True)

# Camera
video_label = ctk.CTkLabel(camera_panel, text="")
video_label.pack(padx=10, pady=10, fill="both", expand=True)

# ================== Row 2, Col 4: RECORDNG ==================
recording_panel = ctk.CTkFrame(main_row)
recording_panel.grid(row=0, column=3, sticky="nsew", padx=5)
setattr(recording_panel, "current_instrument", instrument_ids[current_instrument_index])
setattr(recording_panel, "recording_start_time", 0.0)
setattr(recording_panel, "recording_data", [])
setattr(recording_panel, "selected_recording_data", None)
setattr(recording_panel, "selected_recording_name", None)
# class RecordingPanel(ctk.CTkFrame):
#     def __init__(self, master, **kwargs):
#         super().__init__(master, **kwargs)
#         self.is_playing_back: bool = False
#         # self.timeline_events: list[dict] = []
#         # self.timeline_start_time: float = 0.0
#         self.is_recording: bool = False
#         self.playback_loop: bool = False

# ================== Row 2, Col 5: RECORDINGS LIBRARY ==================
recordings_list_panel = ctk.CTkFrame(main_row)
recordings_list_panel.grid(row=0, column=4, sticky="nsew", padx=5)

recording_label = ctk.CTkLabel(recording_panel, text="🎵 Recording Panel", font=("Segoe UI", 16, "bold"))
recording_label.pack(pady=10)

recording_status = ctk.CTkLabel(
    recording_panel,
    text="Status: Idle",
    width=260,
    anchor="w",
    fg_color=None
)
recording_status.pack(pady=5)

setattr(recording_panel, "is_recording", False)

# Start / Stop Recording buttons
def start_recording():
    print("Recording started...")

    # ✅ Clear selected library recording
    setattr(recording_panel, "selected_recording_data", None)
    setattr(recording_panel, "selected_recording_name", None)

    highlight_selected_recording(None)

    # ✅ Clear live buffer
    setattr(recording_panel, 'recording_data', [])
    setattr(recording_panel, "recording_start_time", time.time())
    setattr(recording_panel, "is_recording", True)

    recording_status.configure(text="🎵 Recording... ")

def stop_recording():
    setattr(recording_panel, "is_recording", False)
    data_len = len(getattr(recording_panel, 'recording_data', []))
    print("Recording stopped. Captured", data_len, "events")
    recording_status.configure(text=f"Stopped. Captured {data_len} events")

start_record_btn = ctk.CTkButton(recording_panel, text="⏺ Start Recording", command=start_recording)
start_record_btn.pack(pady=5, padx=10)

stop_record_btn = ctk.CTkButton(recording_panel, text="⏹ Stop Recording", command=stop_recording)
stop_record_btn.pack(pady=5, padx=10)

recordings_scroll = ctk.CTkScrollableFrame(
    recordings_list_panel,
    width=200,
    height=420
)
recordings_scroll.pack(fill="both", expand=True, padx=5, pady=5)
recordings_actions = ctk.CTkFrame(recordings_list_panel)
recordings_actions.pack(fill="x", padx=5, pady=5)

delete_btn = ctk.CTkButton(
    recordings_actions,
    text="🗑 Delete",
    width=90,
    command=lambda: delete_selected_recording()
)
delete_btn.pack(side="left", padx=5)

rename_btn = ctk.CTkButton(
    recordings_actions,
    text="✏ Rename",
    width=90,
    command=lambda: rename_selected_recording()
)
rename_btn.pack(side="left", padx=5)

recording_buttons = []

def delete_selected_recording():
    name = getattr(recording_panel, "selected_recording_name", None)
    if not name:
        recording_status.configure(text="⚠ No recording selected")
        return

    path = os.path.join(RECORDINGS_DIR, name)

    if os.path.exists(path):
        os.remove(path)

    # Clear selection
    setattr(recording_panel, "selected_recording_data", None)
    setattr(recording_panel, "selected_recording_name", None)
    highlight_selected_recording(None)

    recording_status.configure(text="🗑 Recording deleted")

    update_recording_list()

def rename_selected_recording():
    name = getattr(recording_panel, "selected_recording_name", None)
    if not name:
        recording_status.configure(text="⚠ No recording selected")
        return

    popup = ctk.CTkToplevel()
    popup.title("Rename Recording")
    popup.geometry("300x120")
    popup.grab_set()

    label = ctk.CTkLabel(popup, text="New name:")
    label.pack(pady=5)

    entry = ctk.CTkEntry(popup, width=260)
    entry.insert(0, name.replace(".json", ""))
    entry.pack(pady=5)

    def confirm():
        new_name = entry.get().strip()
        if not new_name:
            return

        old_path = os.path.join(RECORDINGS_DIR, name)
        new_path = os.path.join(RECORDINGS_DIR, new_name + ".json")

        if os.path.exists(new_path):
            recording_status.configure(text="⚠ Name already exists")
            popup.destroy()
            return

        os.rename(old_path, new_path)

        setattr(recording_panel, "selected_recording_name", new_name + ".json")

        recording_status.configure(text=f"Renamed to: {new_name}")
        popup.destroy()

        update_recording_list()
        highlight_selected_recording(new_name + ".json")

    confirm_btn = ctk.CTkButton(popup, text="Rename", command=confirm)
    confirm_btn.pack(pady=8)

def update_recording_list():
    for btn in recording_buttons:
        btn.destroy()
    recording_buttons.clear()

    files = [f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".json")]

    for fname in files:
        display_name = fname.replace(".json", "")  # ✅ hide extension

        def make_loader(file_name=fname, display=display_name):
            def load_and_select():
                with open(os.path.join(RECORDINGS_DIR, file_name), "r") as f:
                    data = json.load(f)

                setattr(recording_panel, "selected_recording_data", data)
                setattr(recording_panel, "selected_recording_name", file_name)

                # ✅ Show only clean name in status
                short = display[:25] + "..." if len(display) > 28 else display
                recording_status.configure(text=f"Selected: {short}")

                highlight_selected_recording(display)
            return load_and_select

        btn = ctk.CTkButton(
            recordings_scroll,
            text=display_name,
            anchor="w",
            width=180,
            command=make_loader()
        )
        btn.pack(pady=2, padx=5)
        recording_buttons.append(btn)
update_recording_list()

def highlight_selected_recording(name):
    for btn in recording_buttons:
        if btn.cget("text") == name:
            btn.configure(fg_color="#00ffaa", text_color="black")
        else:
            btn.configure(fg_color="transparent", text_color="white")

# save recordings
def save_recording(name="recording"):
    data = getattr(recording_panel, 'recording_data', [])
    if not data:
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RECORDINGS_DIR, f"{name}_{timestamp}.json")

    with open(path, "w") as f:
        json.dump(data, f)

    pdf_dir = "exports/pdf"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{name}_{timestamp}.pdf")

    update_recording_list()
    # export_notation_pdf(data, notation_panel, pdf_path)
    # print("Saved exports: JSON & PDF")
    setattr(recording_panel, "selected_recording_data", None)
    setattr(recording_panel, "selected_recording_name", None)
    highlight_selected_recording(None)

# def add_timed_event(self, event):
#     # diff in seconds from current
#     # negative = future (don’t draw)
#     # positive = should be on screen
#     now = time.time() - recording_panel.timeline_start_time
#     age = now - event["time"]
#     # convert age into X offset
#     # same scroll speed as PDF spacing
#     x = self.canvas.winfo_width() - age * self.scroll_speed_px_per_sec
#     symbol = self.get_note_symbol(event["gesture"])
#     y = self.get_note_y(event["gesture"])
#     text_id = self.canvas.create_text(x, y, text=symbol, fill="white", font=("Segoe UI", 20, "bold"))
#     self.events.append((text_id, event))
# def export_notation_pdf(recording_data, notation_panel, file_path):
#     # Import reportlab lazily so missing optional dependency doesn't break module import.
#     try:
#         from reportlab.pdfgen import canvas #type: ignore
#         from reportlab.lib.pagesizes import A4, landscape #type: ignore
#     except Exception:
#         app_logger.error("reportlab is not installed; cannot export PDF. Install it with: pip install reportlab")
#         raise RuntimeError("reportlab is required to export PDF. Install it with: pip install reportlab")
#     c = canvas.Canvas(file_path, pagesize=landscape(A4))
#     width, height = landscape(A4)
#     # draw staff lines
#     staff_y = [100, 115, 130, 145, 160]
#     for y in staff_y:
#         c.line(50, y, width - 50, y)
#     x = 60
#     spacing = 35  # how far each note is horizontally
#     for event in recording_data:
#         gesture = event["gesture"]
#         symbol = notation_panel.get_note_symbol(gesture)
#         y = notation_panel.get_note_y(gesture)
#         # PDF coordinate system starts bottom-left, so adjust:
#         pdf_y = height - (y + 200)
#         c.setFont("Helvetica-Bold", 24)
#         c.drawString(x, pdf_y, symbol)
#         x += spacing
#         # If x exceeds page, create new page
#         if x > width - 60:
#             c.showPage()
#             x = 60
#             for y2 in staff_y:
#                 c.line(50, y2, width - 50, y2)
#     c.save()

stop_record_btn = ctk.CTkButton(recording_panel, text="Save Recording", command=save_recording)
stop_record_btn.pack(pady=5, padx=10)

# Playback button
def playback_recording(index=0, start_time=None):
    setattr(recording_panel, "is_playing_back", True)
    buffer_data = getattr(recording_panel, "recording_data", [])
    selected_data = getattr(recording_panel, "selected_recording_data", None)

    # ✅ Priority Logic
    data = buffer_data if buffer_data else selected_data

    if not data:
        recording_status.configure(text="⚠ Nothing to play")
        setattr(recording_panel, "is_playing_back", False)
        app_logger.info("Playback finished")
        return

    if start_time is None:
        start_time = time.time()

    if index >= len(data):
        if getattr(recording_panel, 'playback_loop', False):
            playback_recording(0)
        else:
            recording_status.configure(text="Playback finished!")
            setattr(recording_panel, "is_playing_back", False)
            app_logger.info("Playback finished")
        return

    event = data[index]

    delay = event["time"] - (time.time() - start_time)
    delay = max(0, int(delay * 1000))

    def play_event():
        note = note_mapping.get(event["gesture"], 60)

        # ✅ Instrument → UI sync
        if event["instrument"] in instrument_ids:
            new_index = instrument_ids.index(event["instrument"])
            set_instrument_by_index(new_index, confirm=False) # confirm = false will not play the ping

        player.play_note(note, duration=1.5)
        notation_panel.add_gesture(event["gesture"])

        # add_timed_event(event, recording_panel.timeline_start_time)

        playback_recording(index + 1, start_time)

    recording_panel.after(delay, play_event)

playback_btn = ctk.CTkButton(recording_panel, text="▶ Playback", command=playback_recording)
playback_btn.pack(pady=10)
setattr(recording_panel, "playback_loop", False)  # initially off

def toggle_loop():
    current = getattr(recording_panel, 'playback_loop', False)
    new = not current
    setattr(recording_panel, 'playback_loop', new)
    loop_btn.configure(text=f"Loop: {'ON' if new else 'OFF'}")
    loop_btn.configure(text=f"Loop: {'ON' if getattr(recording_panel, 'playback_loop', False) else 'OFF'}")

# Loop toggle button
loop_btn = ctk.CTkButton(recording_panel, text="Loop: OFF", command=toggle_loop)
loop_btn.pack(pady=5, padx=10)


def record_gesture(gesture_id):
    if getattr(recording_panel, 'is_recording', False):
        # Use getattr to safely read the optional attribute and provide a default
        recording_start = getattr(recording_panel, 'recording_start_time', 0.0)
        timestamp = time.time() - recording_start

        # Ensure recording_data exists and is a list before appending
        data = getattr(recording_panel, 'recording_data', None)
        if data is None:
            data = []
            setattr(recording_panel, 'recording_data', data)

        data.append({
            "gesture": gesture_id,
            "time": timestamp,
            "instrument": instrument_ids[current_instrument_index]  # store ID 🎯
        })


# ============== Third Row: Notation Panel ==================
notation_frame = ctk.CTkFrame(app, fg_color="black")
notation_frame.pack(padx=20, pady=10, fill="x")
notation_panel = NotationPanel(notation_frame)

# ============== Fourth Row: CONTROL PANEL ==================
footer = ctk.CTkFrame(app)
footer.pack(fill="x", pady=15)

button_frame = ctk.CTkFrame(footer, fg_color="#2B2A45")
button_frame.pack(pady=5)

start_btn = ctk.CTkButton(button_frame, text="▶ Start Model", width=160, command=start_camera)
start_btn.grid(row=0, column=0, padx=10, pady=10)

stop_btn = ctk.CTkButton(button_frame, text="⏹ Stop Model", width=160, command=stop_camera)
stop_btn.grid(row=0, column=1, padx=10, pady=10)

refresh_btn = ctk.CTkButton(button_frame, text="🔁 Refresh", width=160, command=refresh)
refresh_btn.grid(row=1, column=0, padx=10, pady=10)

exit_btn = ctk.CTkButton(button_frame, text="❌ Exit", width=160, command=quit_app)
exit_btn.grid(row=1, column=1, padx=10, pady=10)

status_label = ctk.CTkLabel(footer, text="Status: Idle", font=("Segoe UI", 14))
status_label.pack(pady=5)

def on_close():
    stop_camera()
    player.stop()
    app.destroy()
    try:
        if log_file:
            log_file.close()
    except AttributeError:
        pass

app.protocol("WM_DELETE_WINDOW", on_close) 
app.mainloop()