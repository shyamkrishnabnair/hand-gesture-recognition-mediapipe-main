import mediapipe as mp
import csv
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier

def load_classifiers():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    kpc = KeyPointClassifier()
    phc = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        k_labels = [row[0] for row in csv.reader(f)]

    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        p_labels = [row[0] for row in csv.reader(f)]

    return hands, kpc, phc, k_labels, p_labels
