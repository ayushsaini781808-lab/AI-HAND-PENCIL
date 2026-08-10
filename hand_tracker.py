import cv2
import mediapipe as mp

from config import (
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # model_complexity=0 uses the 'Lite' model which is dramatically faster on weak CPUs (like Streamlit Cloud)
        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            model_complexity=0, 
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.detector.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None, None

        hand = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        landmarks = []

        # Store landmarks
        for lm in hand.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))

        # Draw skeleton using built-in drawing utils
        self.mp_draw.draw_landmarks(
            frame, 
            hand, 
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=5),
            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

        # Handedness (Left/Right)
        hand_type = results.multi_handedness[0].classification[0].label

        return landmarks, hand_type

    def close(self):
        self.detector.close()