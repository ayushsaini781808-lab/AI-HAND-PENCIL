import cv2
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

from config import (
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)

class HandTracker:
    def __init__(self):
        # Using the Legacy API allows us to select model_complexity=0 (Lite model). 
        # This is strictly required for Streamlit Cloud CPUs to achieve 30 FPS without lag.
        self.detector = mp_hands.Hands(
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

        # Draw skeleton connections naturally
        mp_drawing.draw_landmarks(
            frame, 
            hand, 
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=5),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

        # Handedness
        hand_type = results.multi_handedness[0].classification[0].label

        return landmarks, hand_type

    def close(self):
        self.detector.close()