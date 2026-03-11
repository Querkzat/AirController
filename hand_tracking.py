import cv2
import mediapipe as mp
import pyautogui
import math
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

pyautogui.FAILSAFE = False

# screen size
screen_w, screen_h = pyautogui.size()

# mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# webcam
cap = cv2.VideoCapture(0)

# cursor variables
prev_x, prev_y = None, None
sensitivity = 4
smooth_factor = 0.3

# click cooldown
click_delay = 0.5
last_click = 0


def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark

        thumb = lm[4]
        index = lm[8]
        middle = lm[12]
        ring = lm[16]

        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        x = index.x
        y = index.y
        z = index.z

        if prev_x is not None:

            # depth scaling (important for distance use)
            depth_scale = 1 + abs(z) * 4

            dx = (x - prev_x) * screen_w * sensitivity * depth_scale
            dy = (y - prev_y) * screen_h * sensitivity * depth_scale

            # smoothing
            dx *= smooth_factor
            dy *= smooth_factor

            # remove tiny jitters
            if abs(dx) < 2:
                dx = 0
            if abs(dy) < 2:
                dy = 0

            pyautogui.moveRel(dx, dy)

        prev_x = x
        prev_y = y

        # gesture distances
        thumb_middle = distance(thumb, middle)
        thumb_ring = distance(thumb, ring)

        current_time = time.time()

        # left click
        if thumb_middle < 0.05:
            if current_time - last_click > click_delay:
                pyautogui.click()
                last_click = current_time
                cv2.putText(img, "LEFT CLICK", (20,60),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        # right click
        if thumb_ring < 0.05:
            if current_time - last_click > click_delay:
                pyautogui.rightClick()
                last_click = current_time
                cv2.putText(img, "RIGHT CLICK", (20,60),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("Gesture Controller", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()