import copy
import cv2 as cv
import csv
from collections import deque, Counter
import itertools

from utils.draw_utils import calc_bounding_rect, calc_landmark_list
from utils.draw_utils import (
    draw_landmarks,
    draw_bounding_rect,
    draw_info_text,
    draw_point_history,
    draw_info
)
from utils.classifiers import load_classifiers
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

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

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

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history