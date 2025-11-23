import cv2 as cv

import cv2 as cv

def draw_landmarks(image, landmark_point):
    if len(landmark_point) == 0:
        return image

    # Define connections (bones) and their colors
    fingers = {
        "thumb": ([2, 3, 4], (0, 102, 255)),        # Orange
        "index": ([5, 6, 7, 8], (0, 255, 255)),      # Yellow
        "middle": ([9, 10, 11, 12], (0, 255, 0)),    # Green
        "ring": ([13, 14, 15, 16], (255, 0, 255)),   # Purple
        "pinky": ([17, 18, 19, 20], (255, 0, 0)),    # Blue
    }

    palm_connections = [
        (0, 1), (1, 2), (2, 5), (5, 9),
        (9, 13), (13, 17), (17, 0)
    ]

    # Draw fingers
    for finger_name, (points, color) in fingers.items():
        for i in range(len(points) - 1):
            pt1 = tuple(landmark_point[points[i]])
            pt2 = tuple(landmark_point[points[i + 1]])
            cv.line(image, pt1, pt2, (0, 0, 0), 6)      # black shadow
            cv.line(image, pt1, pt2, color, 2)

    # Draw palm connections
    for (start, end) in palm_connections:
        pt1 = tuple(landmark_point[start])
        pt2 = tuple(landmark_point[end])
        cv.line(image, pt1, pt2, (0, 0, 0), 6)
        cv.line(image, pt1, pt2, (255, 255, 255), 2)

    # Draw joints (dots)
    for i, (x, y) in enumerate(landmark_point):
        radius = 8 if i in [4, 8, 12, 16, 20] else 5
        color = (255, 255, 255)
        outline = (0, 0, 0)
        cv.circle(image, (x, y), radius, color, -1)
        cv.circle(image, (x, y), radius, outline, 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, finger_count):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    shadow = 2
    # Draw background rectangle for hand label
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    # Combine handedness and hand sign
    info_text = handedness.classification[0].label

    # Draw hand label
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 5),
               font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)

    # Finger count (draw near hand)
    cv.putText(image, f"Count: {finger_count}", (brect[0], brect[3] + 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            alpha = int(255 * (index / len(point_history)))  # Trail fade effect
            color = (152, 251, 152)
            cv.circle(image, (point[0], point[1]), 2 + index // 4, color, -1)
    return image


def draw_info(image, fps, mode, number):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    shadow = 2

    cv.putText(image, f"FPS: {fps}", (10, 25), font, font_scale, (0, 0, 0), shadow, cv.LINE_AA)
    cv.putText(image, f"FPS: {fps}", (10, 25), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, f"MODE: {mode_string[mode - 1]}", (10, 55),
                   font, font_scale - 0.1, (255, 255, 255), thickness, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, f"NUM: {number}", (10, 75),
                       font, font_scale - 0.1, (255, 255, 255), thickness, cv.LINE_AA)

    return image