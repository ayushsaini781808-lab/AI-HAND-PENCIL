import os
import cv2
import mediapipe as mp
import urllib.request
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)

MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

# Hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


class HandTracker:
    def __init__(self):
        # Download model if missing
        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = 0

        # ── PERFORMANCE: frame skipping ───────────────────────────────────
        # Running full MediaPipe inference on every incoming frame is the
        # single most expensive thing in this app, especially on a shared
        # cloud CPU. We only run detection every DETECT_EVERY_N frames and
        # reuse the last known landmarks in between. The hand moves little
        # between two consecutive frames, so this is visually smooth while
        # cutting inference workload dramatically (e.g. 1/2 or 1/3 the cost).
        self.DETECT_EVERY_N = 2
        self._frame_count = 0
        self._last_landmarks = None
        self._last_hand_type = None

    def detect(self, frame):
        h, w = frame.shape[:2]
        self._frame_count += 1

        run_detection = (self._frame_count % self.DETECT_EVERY_N == 0) \
            or self._last_landmarks is None

        if run_detection:
            # ── PERFORMANCE: Process at reduced resolution ────────────────
            small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Monotonically increasing timestamp in ms
            ts_ms = int(time.time() * 1000)
            if ts_ms <= self._last_ts_ms:
                ts_ms = self._last_ts_ms + 1
            self._last_ts_ms = ts_ms

            results = self.detector.detect_for_video(mp_image, ts_ms)

            if not results.hand_landmarks:
                self._last_landmarks = None
                self._last_hand_type = None
                return None, None

            hand = results.hand_landmarks[0]

            # Scale landmarks back to original frame size
            landmarks = []
            for lm in hand:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append((x, y))

            self._last_landmarks = landmarks
            self._last_hand_type = results.handedness[0][0].category_name

        landmarks = self._last_landmarks
        hand_type = self._last_hand_type

        if landmarks is None:
            return None, None

        # Draw skeleton on original frame
        for idx1, idx2 in HAND_CONNECTIONS:
            cv2.line(frame, landmarks[idx1], landmarks[idx2], (0, 255, 0), 2)
        for pt in landmarks:
            cv2.circle(frame, pt, 5, (0, 0, 255), cv2.FILLED)

        return landmarks, hand_type

    def close(self):
        self.detector.close()