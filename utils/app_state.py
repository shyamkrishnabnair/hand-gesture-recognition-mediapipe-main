# utils/app_state.py
from collections import deque
class AppState:
    def __init__(self):
        # For volume control
        self.last_volume_level = 100
        self.pinch_mode = False
        self.pinch_start_x = 0
        self.is_muted = False

        # For mute/unmute (left hand pinch drag)
        self.left_hand_pinch_state = False

        # For instrument scroll
        self.instrument_scroll_mode = False
        self.instrument_scroll_start_x = 0
        self.instrument_ids = [0, 1, 4, 6, 16, 24, 29, 33, 40, 48, 56, 65, 73, 80, 88]
        self.instrument_names = [
            "Acoustic Grand Piano 🎹", "Bright Piano 🎹", "Electric Piano 🎹", "Harpsichord 🎹", "Drawbar Organ 🎹", "Acoustic Guitar 🎸", "Overdriven Guitar 🎸", "Bass Guitar 🎸", "Violin 🎻", "String Ensemble 🎻", "Trumpet 🎺", "Saxophone 🎷", "Flute 🎶", "Synth Lead 🎹", "Synth Pad 🎹"
        ]
        self.current_instrument_index = 0
        self.notes_mapping = {
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

        # For gesture logic
        self.frame_id = 0
        self.last_finger_count = 0
        self.last_note_time = 0

        # Point history and gesture history
        self.point_history = []
        self.finger_gesture_history = []

        # Live control
        self.running = False
        self.start_time = None

        # MIDI player reference
        self.player = None  # Assign from outside

        # GUI elements (assign after initializing GUI)
        self.instrument_label = None
        self.status_label = None
        self.video_label = None
        self.mute_btn = None

        # Logging (assign externally)
        self.app_logger = None

        # Classifier models (assign externally)
        self.keypoint_classifier = None
        self.point_history_classifier = None

        # Gesture config
        self.note_mapping = {
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
        self.note_cooldown = 0.5  # Seconds
