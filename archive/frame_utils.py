import cv2 as cv
import copy
from collections import Counter
from utils.calculate import calc_bounding_rect, calc_landmark_list
from utils.draw import draw_info_text, draw_bounding_rect, draw_landmarks
from utils.pre_process import pre_process_landmark, pre_process_point_history
from utils.app_state import AppState
# from main import app_logger, instrument_label, player --- this lead to cyclic import line 31, 32, 145-149

def left_hand_label(is_pinch, debug_image, center_px, drag_threshold, player, app_state: AppState):
    if is_pinch:
        cv.putText(debug_image, f"Instrument: {app_state.instrument_names[app_state.current_instrument_index]}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

        cv.circle(debug_image, center_px, 15, (255, 255, 0), 3)  # Yellow for left hand
        if not app_state.instrument_scroll_mode:
            app_state.instrument_scroll_mode = True
            app_state.instrument_scroll_start_x = center_px[0]
        else:
            print(f"Instrument scroll mode active, start X: {app_state.instrument_scroll_start_x}, center X: {center_px[0]}")
            delta_x = center_px[0] - app_state.instrument_scroll_start_x

            if abs(delta_x) >= drag_threshold:
                if delta_x > 0:
                    app_state.current_instrument_index = (app_state.current_instrument_index + 1) % len(app_state.instrument_ids)
                else:
                    app_state.current_instrument_index = (app_state.current_instrument_index - 1) % len(app_state.instrument_ids)

                new_instrument = app_state.instrument_ids[app_state.current_instrument_index]
                player.set_instrument(new_instrument)
                player.play_note(60, duration=0.9)  # C4 note for 0.9s
                # app_logger.info(f"Instrument: {app_state.instrument_names[app_state.current_instrument_index]}")
                # instrument_label.configure(text=f"Instrument: {app_state.instrument_names[app_state.current_instrument_index]}")
                app_state.instrument_scroll_start_x = center_px[0]
    else:
        app_state.instrument_scroll_mode = False

def right_hand_label(is_pinch, center_px, debug_image, app_state, drag_threshold, screen_width):
    if is_pinch:
        cv.circle(debug_image, center_px, 15, (0, 255, 0), 3) # Green circle

        # Ensure last_volume_level always exists (reflect current player volume)
        last_volume_level = int(app_state.player.volume * 100)

        if not app_state.pinch_mode:
            app_state.pinch_mode = True
            app_state.pinch_start_x = center_px[0] # Record start X for horizontal movement
        else:
            if not app_state.player.muted:
                delta_x = center_px[0] - app_state.pinch_start_x
                if abs(delta_x) > drag_threshold/20:

                    # Volume sensitivity (drag 1% screen = 1 step)
                    volume_delta = delta_x / (screen_width * 0.01)
                    volume_delta = max(-10, min(10, volume_delta))  # Cap delta for control
                    # Apply new volume
                    new_volume = app_state.player.volume + (volume_delta / 100)
                    app_state.player.volume = max(0.0, min(1.0, new_volume))

                    # Sync last_volume_level with current MIDI player volume
                    last_volume_level = int(app_state.player.volume * 100)

                    app_state.app_logger.debug(f"Volume: {last_volume_level}%")

            # Always update start position for next drag, even if muted
            app_state.pinch_start_x = center_px[0]

        # Draw volume bar on debug image
        draw_volume_bar(debug_image, last_volume_level)
    else:
        cv.circle(debug_image, center_px, 15, (0, 0, 255), 2) # Blue circle when not pinching
        app_state.pinch_mode = False # Reset pinch mode

def draw_volume_bar(debug_image, last_volume_level):
    bar_x, bar_y = 10, 85
    bar_width, bar_height = 150, 10
    filled_width = int(bar_width * (last_volume_level / 100))
    cv.rectangle(debug_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1) # Background
    cv.rectangle(debug_image, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1) # Foreground
    cv.putText(debug_image, f"Volume: {last_volume_level}%", (bar_x, bar_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

def preprocess_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None, None, "Camera read failed"
    
    frame = cv.flip(frame, 1)
    debug_image = copy.deepcopy(frame)
    rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    return rgb_image, debug_image, ""

def detect_hands_and_classify(rgb_image, hands, debug_image, app_state):
    results = hands.process(rgb_image) # Process the image for hand landmarks
    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            pre_processed_point_history_list = pre_process_point_history(debug_image, app_state.point_history)

            # Classify hand sign (static pose)
            hand_sign_id = app_state.keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id == 2: # Assuming hand_sign_id 2 corresponds to a specific gesture for history
                app_state.point_history.append(landmark_list[8]) # Append index finger tip (landmark 8)
            else:
                app_state.point_history.append([0, 0]) # Append dummy point if not the specific gesture
            # Classify finger gesture (dynamic movement)
            finger_gesture_id = 0
            if len(pre_processed_point_history_list) == 32: # Check if history is full for classification
                finger_gesture_id = app_state.point_history_classifier(pre_processed_point_history_list)

            app_state.finger_gesture_history.append(finger_gesture_id)
            most_common_fg_id = Counter(app_state.finger_gesture_history).most_common()
            hand_label = handedness.classification[0].label # "Left" or "Right"
            hand_landmarks_xy = [[lm.x, lm.y] for lm in hand_landmarks.landmark] # Normalized coordinates

            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            h, w, _ = debug_image.shape
            thumb_px = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_px = int(index_tip.x * w), int(index_tip.y * h)

            # Draw line and small circle between thumb and index finger tips
            center_px = ((thumb_px[0] + index_px[0]) // 2, (thumb_px[1] + index_px[1]) // 2)
            cv.line(debug_image, thumb_px, index_px, (255, 255, 255), 2)
            cv.circle(debug_image, center_px, 10, (0, 255, 0), 2)

            # Calculate Euclidean distance between thumb and index tips
            dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

            # Detect pinch based on distance threshold
            is_pinch = dist < 0.07 # Threshold for pinch detection
            screen_width = debug_image.shape[1]
            drag_threshold = screen_width * 0.2  # 20% of screen width
            # Left Hand: Mute/Unmute Toggle

            if 'pinch_mode' not in globals():
                app_state.pinch_mode = False
                app_state.pinch_start_x = 0
                
            cv.putText(debug_image, f"Instrument: {app_state.instrument_names[app_state.current_instrument_index]}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

            if hand_label == "Left":
                # left_hand_label(is_pinch, debug_image, center_px, drag_threshold, player, app_state)
            # elif hand_label == "Right":
            #     right_hand_label(is_pinch, center_px, debug_image, app_state, drag_threshold, screen_width)

            # if app_state.player.volume < 0.01 or app_state.player.muted:
                cv.putText(debug_image, "Muted", (debug_image.shape[1] - 50, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 0, cv.LINE_AA)
                app_state.mute_btn.configure(text="Unmute" if app_state.player.muted else "Mute")
                app_state.app_logger.info("Muted" if app_state.player.muted else "Unmuted")

            draw_debug_overlays(brect, landmark_list, hand_sign_id, most_common_fg_id, count_fingers(hand_label, hand_landmarks_xy, hand_sign_id, most_common_fg_id, app_state), handedness, debug_image)

            return hand_label, hand_sign_id, most_common_fg_id, landmark_list, hand_landmarks_xy
    else:
        app_state.point_history.append([0, 0]) # Append dummy point if no hands detected
        app_state.app_logger.debug("No hands detected in frame.")
        app_state.player.note_cooldowns.clear()  # Reset if no hands detected
        # return None, None, None, [], []
        
def count_fingers(hand_label, hand_landmarks_xy, hand_sign_id, most_common_fg_id, app_state):
    # Finger Counting Logic
    finger_count = 0
    # Thumb (landmark 4 vs 3): Check horizontal position relative to its base
    if (hand_label == "Left" and hand_landmarks_xy[4][0] > hand_landmarks_xy[3][0]) or \
        (hand_label == "Right" and hand_landmarks_xy[4][0] < hand_landmarks_xy[3][0]):
        finger_count += 1
    # Other fingers (tips vs PIP joints): Check vertical position

    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if hand_landmarks_xy[tip][1] < hand_landmarks_xy[pip][1]:
            finger_count += 1
    app_state.total_finger_count += finger_count # Accumulate finger count from all hands

    # Log detected hand info to the UI
    app_state.app_logger.info(f"Hand: {hand_label}, Fingers: {finger_count}")
    return finger_count

def draw_debug_overlays(brect, landmark_list, hand_sign_id, most_common_fg_id, finger_count, handedness, debug_image):
    debug_image = draw_bounding_rect(True, debug_image, brect)
    debug_image = draw_landmarks(debug_image, landmark_list)
    debug_image = draw_info_text(
        debug_image,
        brect,
        handedness,
        finger_count,
    )