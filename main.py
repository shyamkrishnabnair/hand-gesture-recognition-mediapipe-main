#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ignore warnings
import warnings
import os
import absl.logging
import sys
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0'=all logs, '1'=filter INFO, '2'=filter WARNING, '3'=filter ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
absl.logging.set_verbosity(absl.logging.ERROR)
sys.stderr = open(os.devnull, 'w') 
import subprocess

#library imports
import csv
import copy
from collections import Counter
from collections import deque

import cv2 as cv
import mediapipe as mp


#model imports
from utils import CvFpsCalc
from utils.get_args import get_args
from model import KeyPointClassifier
from model import PointHistoryClassifier
from utils.select_mode import select_mode
from utils.calculate import calc_bounding_rect, calc_landmark_list
from utils.pre_process import pre_process_landmark, pre_process_point_history
from utils.log import logging_csv
from utils.draw import draw_info_text, draw_bounding_rect, draw_point_history, draw_info, draw_landmarks

#tkinter UI
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import time


# Global vars
running = False
cap = None
video_label = None
start_time = None


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
    root.destroy() 
    subprocess.Popen([sys.executable, 'main.py'])
    print("GUI refreshed (main.py restarted)")

def quit():
    stop_camera()
    root.destroy() 
    print("GUI closed")

def update_timer():
    if running:
        elapsed = int(time.time() - start_time)
        status_label.config(text=f"Status: Running ({elapsed}s)")
        root.after(1000, update_timer)

def start_camera():
    global running, cap, start_time
    if running:
        return
    running = True
    start_time = time.time()
    cap = cv2.VideoCapture(0)
    update_timer()
    update_frame()
    print("Camera started")

def stop_camera():
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    status_label.config(text="Status: Stopped")
    print("Camera stopped")

def update_frame():
    if not running or not cap:
        return

    ret, image = cap.read()
    if not ret:
        return

    image = cv2.flip(image, 1)
    debug_image = copy.deepcopy(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    number, mode = 0, 0

    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
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

            debug_image = draw_bounding_rect(True, debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmark_list)
            debug_image = draw_info_text(debug_image, brect, handedness,
                                         keypoint_classifier_labels[hand_sign_id],
                                         point_history_classifier_labels[most_common_fg_id[0][0]])

    else:
        point_history.append([0, 0])

    debug_image = draw_point_history(debug_image, point_history)
    fps = int(1 / max(0.01, time.time() - start_time))
    debug_image = draw_info(debug_image, fps, 0, 0)

    img = Image.fromarray(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    status_label.config(text=f"Status: Running ({int(time.time() - start_time)}s)")

    # schedule next frame
    video_label.after(10, update_frame)

# GUI
root = tk.Tk()
root.title("Hand Gesture Controller")
root.geometry("1200x1200")

tk.Label(root, text="Hand Gesture Controller", font=("Arial", 16)).pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

start_btn = tk.Button(root, text="Start Model", width=20, command=start_camera)
start_btn.pack(pady=5)

stop_btn = tk.Button(root, text="Stop Model", width=20, command=stop_camera)
stop_btn.pack(pady=5)

refresh_gui_btn = tk.Button(root, text="Refresh GUI", width=20, command=refresh)
refresh_gui_btn.pack(pady=5)

status_label = tk.Label(root, text="Status: Idle", font=("Arial", 10))
status_label.pack(pady=10)

exit_btn = tk.Button(root, text="Exit", width=20, command=quit)
exit_btn.pack(pady=5)

def on_close():
    stop_camera()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()