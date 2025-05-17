import copy
import cv2 as cv
# import numpy as np
# import mediapipe as mp
from collections import deque, Counter
# from model import PointHistoryClassifier
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 3 = only fatal
from utils.draw_utils import (
    draw_landmarks,
    draw_bounding_rect,
    draw_info_text,
    draw_point_history,
    draw_info
)
from utils.pipeline import pre_process_landmark, pre_process_point_history
from utils.classifiers import load_classifiers
from utils.pipeline import logging_csv
from utils.draw_utils import calc_bounding_rect, calc_landmark_list
from utils.cvfpscalc import CvFpsCalc

# Load models and helpers
hands, keypoint_classifier, point_history_classifier, keypoint_classifier_labels, point_history_classifier_labels = load_classifiers()

# Internal state
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

# Settings
use_brect = True

# FPS calculator
cvFpsCalc = CvFpsCalc()


def process_frame(frame, mode=0, number=-1):
    fps = cvFpsCalc.get()
    debug_image = copy.deepcopy(frame)

    # Process image
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    finger_gesture_id = 0
    hand_sign_id = -1
    label = ""

    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Bounding box
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            # Landmark
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            # Preprocessing
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

            # Logging
            logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

            # Keypoint classification
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id == 2:  # Point gesture
                point_history.append(landmark_list[8])
            else:
                point_history.append([0, 0])

            # Point history classification
            if len(pre_processed_point_history_list) == history_length * 2:
                finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
            finger_gesture_history.append(finger_gesture_id)
            most_common_fg_id = Counter(finger_gesture_history).most_common()[0][0]

            # Draw
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmark_list)
            debug_image = draw_info_text(debug_image, brect, handedness,
                                         keypoint_classifier_labels[hand_sign_id],
                                         point_history_classifier_labels[most_common_fg_id])
            label = point_history_classifier_labels[most_common_fg_id]
    else:
        point_history.append([0, 0])

    debug_image = draw_point_history(debug_image, point_history)
    debug_image = draw_info(debug_image, fps, mode, number)

    return debug_image, {"gesture": label, "hand_sign": keypoint_classifier_labels[hand_sign_id] if hand_sign_id != -1 else "None"}
