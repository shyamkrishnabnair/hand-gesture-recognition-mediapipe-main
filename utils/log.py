import os
import sys
from datetime import datetime
import time
from typing import Any, Dict

def logging():
    log_dir = "exports/logs"
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

last_note_info: Dict[str, Any] = {
    "note": None,
    "start_time": None,
    "finger_count": None,
    "instrument": None,
    "last_log_time": 0.0
}

NOTE_STABILITY_WINDOW = 0.5  # Ignore notes shorter than this (likely jitter)
NOTE_LOG_COOLDOWN = 2.5      # Cooldown before same note can be re-logged


def log_note_play(note: int, finger_count: int, instrument_index: int) -> None:
    """
    Logs note events in a clean, non-repetitive way.
    Skips jittery repeats and merges continued notes gracefully.
    """
    global last_note_info

    now = time.time()
    current_time = time.strftime("%H:%M:%S", time.localtime(now))

    # Handle same-note repeats (jitter or too soon)
    if (
        last_note_info["note"] == note and
        last_note_info["instrument"] == instrument_index and
        now - last_note_info.get("last_log_time", 0.0) < NOTE_LOG_COOLDOWN
    ):
        return  # Skip re-logging same note within cooldown window

    # Gracefully log note continuation if same note after cooldown
    if (
        last_note_info["note"] == note and
        last_note_info["instrument"] == instrument_index and
        now - last_note_info.get("last_log_time", 0.0) >= NOTE_LOG_COOLDOWN
    ):
        print(f"[{current_time}] ~ Continuing note: {note} "
              f"(Instrument {instrument_index}, Fingers {finger_count})")
        last_note_info["last_log_time"] = now
        return

    # If a note was playing before, compute its duration and log end
    prev_start = last_note_info.get("start_time")
    if last_note_info["note"] is not None and isinstance(prev_start, (int, float)):
        duration = now - prev_start
        if duration >= NOTE_STABILITY_WINDOW:
            print(f"[{current_time}] :: Ended note: {last_note_info['note']} "
                  f"(Instrument {last_note_info['instrument']}, Fingers {last_note_info['finger_count']}) "
                  f"-> Duration: {duration:.2f}s")

    # Log the new note
    print(f"[{current_time}] <> Playing note: {note} "
          f"(Instrument {instrument_index}, Fingers {finger_count})")

    # Update state explicitly for type-safety
    last_note_info["note"] = note
    last_note_info["start_time"] = now
    last_note_info["finger_count"] = finger_count
    last_note_info["instrument"] = instrument_index
    last_note_info["last_log_time"] = now
