
import cv2 as cv
import time
import copy
import csv
from collections import deque, Counter
from model import KeyPointClassifier, PointHistoryClassifier
from utils.calculate import calc_bounding_rect, calc_landmark_list
from utils.pre_process import pre_process_landmark, pre_process_point_history
from utils.draw import draw_info_text, draw_bounding_rect, draw_point_history, draw_info, draw_landmarks
from utils import MidiSoundPlayer  # <- NEW UTILITY
import mediapipe as mp

# Load models and labels
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f)]

# Initialize MIDI sound player (no need for .sf2 or external tools)
player = MidiSoundPlayer()
player.set_instrument(0)  # Acoustic Grand Piano

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Tracking
point_history = deque(maxlen=16)
finger_gesture_history = deque(maxlen=16)
last_finger_count = 0
last_note_time = 0
note_cooldown = 0.5  # seconds between two notes
last_volume_level = 50
is_muted = False
pinch_mode = False
pinch_start_x = 0
left_hand_pinch_state = False
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

# === Main Execution ===
cap = cv.VideoCapture(0)
if not cap.isOpened():
    exit()

try:
    while True:
        frame_start_time = time.time()

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        total_finger_count = 0

        if results.multi_hand_landmarks:
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

                hand_label = handedness.classification[0].label
                hand_landmarks_xy = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                h, w, _ = debug_image.shape
                thumb_px = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_px = int(index_tip.x * w), int(index_tip.y * h)
                center_px = ((thumb_px[0] + index_px[0]) // 2, (thumb_px[1] + index_px[1]) // 2)
                cv.line(debug_image, thumb_px, index_px, (255, 255, 255), 2)
                cv.circle(debug_image, center_px, 10, (0, 255, 0), 2)

                dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
                is_pinch = dist < 0.07

                if hand_label == "Left":
                    if is_pinch:
                        if not left_hand_pinch_state:
                            left_hand_pinch_state = True
                            pinch_start_x = center_px[0]  # Save starting X position
                        else:
                            delta_x = center_px[0] - pinch_start_x
                            volume_delta = int(delta_x / 5)

                            # Change volume toward mute/unmute
                            new_volume = max(0, min(100, last_volume_level + volume_delta))
                            if new_volume <= 10:
                                is_muted = True                                     
                                player.set_volume(0.0)
                            else:
                                is_muted = False
                                player.set_volume(new_volume / 100.0)

                    else:
                        left_hand_pinch_state = False

                elif hand_label == "Right":
                    if is_pinch:
                        if not pinch_mode:
                            pinch_mode = True
                            pinch_start_x = center_px[0]
                        else:
                            delta_x = center_px[0] - pinch_start_x
                            volume_delta = int(delta_x / 5)
                            last_volume_level = max(0, min(100, last_volume_level + volume_delta))
                            if not is_muted:
                                player.set_volume(last_volume_level / 100.0)
                        pinch_start_x = center_px[0]
                    else:
                        pinch_mode = False

                finger_count = 0
                if (hand_label == "Left" and hand_landmarks_xy[4][0] > hand_landmarks_xy[3][0]) or \
                   (hand_label == "Right" and hand_landmarks_xy[4][0] < hand_landmarks_xy[3][0]):
                    finger_count += 1
                for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                    if hand_landmarks_xy[tip][1] < hand_landmarks_xy[pip][1]:
                        finger_count += 1
                total_finger_count += finger_count

                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness,
                                             keypoint_classifier_labels[hand_sign_id],
                                             point_history_classifier_labels[most_common_fg_id[0][0]],
                                             finger_count)

                current_time = time.time()

                if (
                    total_finger_count in note_mapping and
                    (total_finger_count != last_finger_count or current_time - last_note_time > note_cooldown)
                ):
                    note = note_mapping[total_finger_count]
                    try:
                        player.play_note(note, duration=10)
                        print(f"🎵 Playing MIDI note: {note} for finger count: {total_finger_count}")
                        last_finger_count = total_finger_count
                        last_note_time = current_time
                    except Exception as e:
                        print(f"Sound error: {e}")
        else:
            point_history.append([0, 0])
            player.note_cooldowns.clear()  # Reset if no hands

        debug_image = draw_point_history(debug_image, point_history)
        fps = int(1 / max(0.01, time.time() - frame_start_time))
        debug_image = draw_info(debug_image, fps, 0, 0)

        cv.imshow("Hand Gesture Recognition", debug_image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    player.stop()
    cv.destroyAllWindows()