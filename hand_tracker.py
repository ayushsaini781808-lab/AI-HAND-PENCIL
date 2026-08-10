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
        self.mp_draw = mp.solutions.drawing_utils
        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            model_complexity=0,  # 0 = Lite (fastest), 1 = Full (heavy)
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

    def detect(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.detector.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None, None

        hand = results.multi_hand_landmarks[0]
        h, w = frame.shape[:2]
        landmarks = []

        # Store landmarks scaled to image size
        for lm in hand.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))

        mp_draw = self.mp_draw
        mp_hands = self.mp_hands

        mp_draw.draw_landmarks(
            frame, 
            hand, 
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=5),
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

        # Handedness
        hand_type = results.multi_handedness[0].classification[0].label

        return landmarks, hand_type

    def close(self):
        self.detector.close()