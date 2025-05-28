import csv
import os
import sys
from datetime import datetime

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

def logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    existing_logs = [f for f in os.listdir(log_dir) if f.startswith("log") and f.endswith(".log")]
    log_numbers = [int(f[3:-4]) for f in existing_logs if f[3:-4].isdigit()]
    next_log_number = max(log_numbers, default=0) + 1
    log_filename = f"log{next_log_number}.log"
    log_path = os.path.join(log_dir, log_filename)

    log_file = open(log_path, "w")
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"[{datetime.now()}] - Log started: {log_filename}")
    return log_file