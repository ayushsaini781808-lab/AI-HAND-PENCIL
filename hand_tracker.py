import os
import cv2
import mediapipe as mp
import urllib.request

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)

# Define connections for drawing the skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # index
    (5, 9), (9, 10), (10, 11), (11, 12), # middle
    (9, 13), (13, 14), (14, 15), (15, 16),# ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                              # palm
]

class HandTracker:
    def __init__(self):
        # Auto-download the task file if it doesn't exist
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):
        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect
        results = self.detector.detect(mp_image)

        if not results.hand_landmarks:
            return None, None

        hand = results.hand_landmarks[0]
        h, w, _ = frame.shape
        landmarks = []

        # Store landmarks
        for lm in hand:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))

        # Draw skeleton
        for idx1, idx2 in HAND_CONNECTIONS:
            pt1 = landmarks[idx1]
            pt2 = landmarks[idx2]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        for pt in landmarks:
            cv2.circle(frame, pt, 5, (0, 0, 255), cv2.FILLED)

        # Handedness
        # In mediapipe Tasks API, results.handedness is a list of lists of categories.
        hand_type = results.handedness[0][0].category_name

        return landmarks, hand_type

    def close(self):
        self.detector.close()