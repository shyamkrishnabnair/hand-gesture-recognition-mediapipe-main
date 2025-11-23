import cv2
import csv
import os
import time
import mediapipe as mp
from utils import calc_landmark_list, pre_process_landmark

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def record_gesture(label, output_dir='dataset/gesture'):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    print(f"Recording for label: {label}. Press 's' to save frame, 'q' to quit.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(image, hand_landmarks)
                preprocessed = pre_process_landmark(landmark_list)
                cv2.putText(image, "Hand detected. Press 's' to save.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Gesture Recorder", image)
        key = cv2.waitKey(1)

        if key == ord('s') and results.multi_hand_landmarks:
            filename = os.path.join(output_dir, f"{label}.csv")
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(preprocessed)
            print(f"Saved frame to {filename}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
