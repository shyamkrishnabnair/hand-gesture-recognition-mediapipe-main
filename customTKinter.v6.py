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

# Incremental logs (from your custom logging module)
log_file = custom_logging()
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()
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
instrument_ids = [0, 1, 4, 6, 16, 24, 29, 33, 40, 48, 56, 65, 73, 80, 88]
instrument_names = [
    "Acoustic Grand Piano 🎹", "Bright Piano 🎹", "Electric Piano 🎹", "Harpsichord 🎹", "Drawbar Organ 🎹", "Acoustic Guitar 🎸", "Overdriven Guitar 🎸", "Bass Guitar 🎸", "Violin 🎻", "String Ensemble 🎻", "Trumpet 🎺", "Saxophone 🎷", "Flute 🎶", "Synth Lead 🎹", "Synth Pad 🎹"
]

# Initialize MIDI sound player (no need for .sf2 or external tools)
player = MidiSoundPlayer()
player.set_instrument(0)  # Acoustic Grand Piano

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)

def refresh():
    stop_camera()
    app.destroy()
    subprocess.Popen([sys.executable, 'customTkinter.v6.py'])
    app_logger.info("GUI refreshed (customTkinter.v6.py restarted)")
    log_file.close()

def quit_app():
    stop_camera()
    app.destroy()
    app_logger.info("GUI closed")
    log_file.close()

def update_timer():
    if running:
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
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                # Classify hand sign (static pose)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2: # Assuming hand_sign_id 2 corresponds to a specific gesture for history
                    point_history.append(landmark_list[8]) # Append index finger tip (landmark 8)
                else:
                    point_history.append([0, 0]) # Append dummy point if not the specific gesture

                # Classify finger gesture (dynamic movement)
                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == 32: # Check if history is full for classification
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

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
                cv.putText(debug_image, f"Instrument: {instrument_names[current_instrument_index]}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
                
                if hand_label == "Left":
                    if is_pinch:
                        cv.putText(debug_image, f"Instrument: {instrument_names[current_instrument_index]}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

                        cv.circle(debug_image, center_px, 15, (255, 255, 0), 3)  # Yellow for left hand
                        if not instrument_scroll_mode:
                            instrument_scroll_mode = True
                            instrument_scroll_start_x = center_px[0]
                        else:
                            print(f"Instrument scroll mode active, start X: {instrument_scroll_start_x}, center X: {center_px[0]}")
                            delta_x = center_px[0] - instrument_scroll_start_x

                            if abs(delta_x) >= drag_threshold:
                                if delta_x > 0:
                                    current_instrument_index = (current_instrument_index + 1) % len(instrument_ids)
                                else:
                                    current_instrument_index = (current_instrument_index - 1) % len(instrument_ids)

                                new_instrument = instrument_ids[current_instrument_index]
                                player.set_instrument(new_instrument)
                                player.play_note(60, duration=0.9)  # C4 note for 0.9s 
                                app_logger.info(f"Instrument: {instrument_names[current_instrument_index]}")
                                instrument_label.configure(text=f"Instrument: {instrument_names[current_instrument_index]}")
                                instrument_scroll_start_x = center_px[0] 
                    else:
                        instrument_scroll_mode = False
                        
                # Right Hand: Volume Control
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
                app_logger.info(f"Hand: {hand_label}, Fingers: {finger_count}, Sign: {keypoint_classifier_labels[hand_sign_id]}, Gesture: {point_history_classifier_labels[most_common_fg_id[0][0]]}")

                # Draw annotations on the debug image
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                    finger_count,
                )
                
                current_time = time.time()
                global last_finger_count, last_note_time
                if (
                    total_finger_count in note_mapping and
                    (total_finger_count != last_finger_count or current_time - last_note_time > note_cooldown)
                ):
                    note = note_mapping[total_finger_count]
                    try:
                        player.play_note(note, duration=10)
                        print(f"Playing MIDI note: {note} for finger count: {total_finger_count}")
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

    global video_label
    imgtk = ImageTk.PhotoImage(image=resized_img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    status_label.configure(text=f"Status: Running ({int(time.time() - start_time)}s)")
    video_label.after(10, update_frame) # Schedule the next frame update


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
mute_btn = ctk.CTkButton(left_panel, text="Mute", command=toggle_mute)
mute_btn.pack(pady=20, padx=10)
left_panel.pack_propagate(False)  # Prevent it from resizing to contents
left_panel.pack(side="left", fill="y", padx=(0, 10))

# Center for camera
center_panel = ctk.CTkFrame(main_frame)
video_label = ctk.CTkLabel(center_panel, text="")
video_label.pack(padx=20, pady=10, fill="both", expand=True)
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
app_logger.addHandler(log_handler) # Add our custom handler to the application logger

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

status_label = ctk.CTkLabel(app, text="Status: Idle", font=("Segoe UI", 14))
status_label.pack(pady=10)

def on_close():
    stop_camera()
    player.stop()
    app.destroy()
    try:
        if log_file:
            log_file.close()
    except AttributeError:
        pass # Ignore if log_file doesn't have a close method or is None

app.protocol("WM_DELETE_WINDOW", on_close) # Bind the on_close function to the window's close button
app.mainloop() # Start the CustomTkinter event loop