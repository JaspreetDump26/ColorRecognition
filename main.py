import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Function to adjust brightness
def adjust_brightness(hand_y):
    screen_height = pyautogui.size().height
    # Normalize hand_y to a brightness level (0-100)
    brightness_level = int((hand_y / screen_height) * 100)
    brightness_level = max(0, min(100, brightness_level))  # Clamp between 0 and 100
    # Set brightness using system command (Windows example)
    pyautogui.press('f2')  # Assuming F2 is the brightness down key
    for _ in range(100 - brightness_level):
        pyautogui.press('f2')  # Decrease brightness
    for _ in range(brightness_level):
        pyautogui.press('f3')  # Increase brightness

# Start capturing video
cap = cv2.VideoCapture(0)  # Change to 1 if using an external camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks and adjust brightness
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the y-coordinate of the wrist (landmark 0)
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            # Convert to pixel coordinates
            h, w, _ = frame.shape
            hand_y = int(wrist_y * h)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Adjust brightness based on hand position
            adjust_brightness(hand_y)

    # Display the frame
    cv2.imshow('Hand Detection and Brightness Control', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()