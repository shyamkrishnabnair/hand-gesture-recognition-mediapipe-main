# customTKinter.v2.py
import sys

import csv
import copy
from collections import Counter, deque
import cv2 as cv
import mediapipe as mp
from PIL import Image, ImageTk
import time
import math
import subprocess
import pygame
import numpy as np
import customtkinter as ctk #type: ignore
try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"Warning: Could not initialize mixer: {e}")

# Model imports
from model import KeyPointClassifier, PointHistoryClassifier
from utils.calculate import calc_bounding_rect, calc_landmark_list
from utils.pre_process import pre_process_landmark, pre_process_point_history
from utils.draw import draw_info_text, draw_bounding_rect, draw_point_history, draw_info, draw_landmarks

# CustomTkinter setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Global vars
running = False
cap = None
start_time = None
last_volume_level = 50
pinch_mode = False
pinch_start_x = 0
sounds_mapping = {
    1: "sounds/kick-bass.mp3",
    2: "sounds/crash.mp3",
    3: "sounds/snare.mp3",
    4: "sounds/tom-1.mp3",
    5: "sounds/tom-2.mp3",
    6: "sounds/tom-3.mp3",
    7: "sounds/cr78-Cymbal.mp3",
    8: "sounds/cr78-Guiro 1.mp3",
    9: "sounds/tempest-HiHat Metal.mp3",
    10: "sounds/cr78-Bongo High.mp3"
}

# Load classifiers & labels once
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f)]

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

point_history = deque(maxlen=16)
finger_gesture_history = deque(maxlen=16)

def refresh():
    stop_camera()
    app.destroy()
    subprocess.Popen([sys.executable, 'customTKinter.v1.py'])
    print("GUI refreshed (customTKinter.v1.py restarted)")

def quit_app():
    stop_camera()
    app.destroy()
    print("GUI closed")

def update_timer():
    if running:
        elapsed = int(time.time() - start_time)
        status_label.configure(text=f"Status: Running ({elapsed}s)")
        app.after(1000, update_timer)

def start_camera():
    global running, cap, start_time
    if running:
        return
    running = True
    start_time = time.time()
    cap = cv.VideoCapture(0)
    update_timer()
    update_frame()
    print("Camera started")

def stop_camera():
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    status_label.configure(text="Status: Stopped")
    print("Camera stopped")

def update_frame():
    if not running or not cap:
        return

    frame_start_time = time.time()

    ret, image = cap.read()
    if not ret:
        return

    image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(image)
    total_finger_count = 0
    try:
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == 32:
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # === FINGER COUNT LOGIC ===
                hand_label = handedness.classification[0].label
                hand_landmarks_xy = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                
                # Get thumb tip (4) and index finger tip (8) positions
                global last_volume_level, pinch_mode, pinch_start_x
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                # Convert normalized coords to pixels
                h, w, _ = debug_image.shape
                thumb_px = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_px = int(index_tip.x * w), int(index_tip.y * h)

                
                # Draw line and small box/circle between thumb and index
                center_px = ((thumb_px[0] + index_px[0]) // 2, (thumb_px[1] + index_px[1]) // 2)
                cv.line(debug_image, thumb_px, index_px, (255, 255, 255), 2)
                cv.circle(debug_image, center_px, 10, (0, 255, 0), 2)
                
                # Calculate distance between thumb and index
                dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

                # Normalize distance (tune min/max based on your hand size/camera)
                dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
                min_dist, max_dist = 0.02, 0.20  # Tune for your hand/camera
                clamped_dist = max(min(dist, max_dist), min_dist)
                volume_level = int(((clamped_dist - min_dist) / (max_dist - min_dist)) * 100)

                # Detect pinch
                is_pinch = dist < 0.07  # Normalized threshold (~7% of screen)

                ## Update volume level only during pinch
                if is_pinch:
                    cv.circle(debug_image, center_px, 15, (0, 255, 0), 3)  # Green circle when pinching
                    if not pinch_mode:
                        pinch_mode = True
                        pinch_start_x = center_px[0]  # Start tracking horizontal movement
                    else:
                        delta_x = center_px[0] - pinch_start_x
                        volume_delta = int(delta_x / 5)  # Adjust sensitivity here
                        last_volume_level = max(0, min(100, last_volume_level + volume_delta))
                        pinch_start_x = center_px[0]  # Update for next frame
                    pygame.mixer.music.set_volume(last_volume_level / 100.0)
                    # Draw volume bar (always draw last known level)
                    bar_x, bar_y = 10, 85
                    bar_width, bar_height = 150, 10
                    filled_width = int(bar_width * (last_volume_level / 100))
                    cv.rectangle(debug_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                    cv.rectangle(debug_image, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)
                    cv.putText(debug_image, f"Volume: {last_volume_level}%", (bar_x, bar_y - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                    print("Volume Level (0-100):", volume_level)
                else:
                    cv.circle(debug_image, center_px, 15, (0, 0, 255), 2)  # Red circle when not pinching
                    pinch_mode = False

                finger_count = 0
                #Thumb
                if (hand_label == "Left" and hand_landmarks_xy[4][0] > hand_landmarks_xy[3][0]) or (hand_label == "Right" and hand_landmarks_xy[4][0] < hand_landmarks_xy[3][0]):
                    finger_count += 1

                #Other fingers
                for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                    if hand_landmarks_xy[tip][1] < hand_landmarks_xy[pip][1]:
                        finger_count += 1

                total_finger_count += finger_count

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

                # Play sound based on finger count
                if total_finger_count in sounds_mapping:
                    sound_file = sounds_mapping[total_finger_count]
                    pygame.mixer.music.load(sound_file)
                    pygame.mixer.music.play()
                    pygame.time.delay(100 if total_finger_count <= 5 else 200)     
                    pygame.mixer.music.stop()
        else:
            point_history.append([0, 0])

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing frame: {e}")

    debug_image = draw_point_history(debug_image, point_history)
    fps = int(1 / max(0.01, time.time() - frame_start_time))
    debug_image = draw_info(debug_image, fps, 0, 0)

    img = Image.fromarray(cv.cvtColor(debug_image, cv.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    status_label.configure(text=f"Status: Running ({int(time.time() - start_time)}s)")
    video_label.after(10, update_frame)


# === UI Setup with customtkinter ===
app = ctk.CTk()
app.title("🤖 Hand Gesture Recognition")
app.geometry("1000x800")

title = ctk.CTkLabel(app, text="🤖 Hand Gesture Controller", font=("Segoe UI", 26, "bold"))
title.pack(pady=20)

video_label = ctk.CTkLabel(app, text="")
video_label.pack(padx=20, pady=10, fill="both", expand=True)

button_frame = ctk.CTkFrame(app, fg_color="#2B2A45")
button_frame.pack(pady=20)

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
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_close)
app.mainloop()