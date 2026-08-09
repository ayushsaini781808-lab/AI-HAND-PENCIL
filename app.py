import os
import subprocess
import sys

# Hack for Streamlit Cloud to remove the non-headless OpenCV installed by mediapipe
# which causes libGL.so.1 and libgthread-2.0.so.0 errors
try:
    import cv2
except ImportError as e:
    if "libgthread" in str(e) or "libGL" in str(e) or "libSM" in str(e):
        print("Detected headless environment missing UI libraries. Fixing OpenCV dependencies...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-contrib-python", "opencv-python"])
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-contrib-python-headless", "opencv-python-headless"])
        print("Done. Restarting Streamlit...", flush=True)
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run"] + sys.argv)
    else:
        raise e

import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

from config import CURRENT_COLOR, TOOLBAR_HEIGHT
from hand_tracker import HandTracker
from gesture import get_finger_states, detect_gesture, GestureStabilizer
from painter import Painter
from utils import CoordinateSmoother
from ui import draw_toolbar, select_color, draw_mode, draw_cursor, draw_eraser_cursor, draw_coordinates

st.set_page_config(
    page_title="AI Hand Pencil",
    page_icon="✏️",
    layout="wide"
)

st.title("🖐️ AI Hand Pencil")
st.markdown(
    "Draw in the air using your hand gestures! "
    "☝️ **1 finger** = Draw  |  ✌️ **2 fingers** = Select Color  |  🖐️ **Open hand** = Erase  |  ✊ **Fist** = Idle"
)

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
        ]
    }
)


class AIPencilProcessor(VideoProcessorBase):
    def __init__(self):
        self.hand_tracker = HandTracker()
        self.gesture_stabilizer = GestureStabilizer()
        self.smoother = CoordinateSmoother()
        self.painter = Painter()
        self.current_color = CURRENT_COLOR

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        self.painter.create_canvas(img)
        landmarks, hand_type = self.hand_tracker.detect(img)
        draw_toolbar(img, self.current_color)

        if landmarks is not None:
            fingers = get_finger_states(landmarks, hand_type)
            raw_gesture = detect_gesture(fingers)
            stable_gesture = self.gesture_stabilizer.update(raw_gesture)

            raw_x, raw_y = landmarks[8]
            x, y = self.smoother.smooth(raw_x, raw_y)
            current_point = (x, y)

            draw_mode(img, stable_gesture)
            draw_coordinates(img, current_point)

            if stable_gesture == "DRAW":
                draw_cursor(img, current_point, self.current_color)
                if y > TOOLBAR_HEIGHT:
                    self.painter.draw(current_point, self.current_color)
                else:
                    self.painter.reset_previous()

            elif stable_gesture == "SELECT":
                self.painter.reset_previous()
                self.current_color = select_color(current_point, self.current_color)

            elif stable_gesture == "ERASE":
                self.painter.reset_previous()
                draw_eraser_cursor(img, current_point)
                if y > TOOLBAR_HEIGHT:
                    self.painter.erase(current_point)

            else:
                self.painter.reset_previous()
        else:
            self.painter.reset_previous()
            self.smoother.reset()

        draw_toolbar(img, self.current_color)
        output = self.painter.combine(img)

        return av.VideoFrame.from_ndarray(output, format="bgr24")


ctx = webrtc_streamer(
    key="ai-pencil",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=AIPencilProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
if st.button("🗑️ Clear Canvas"):
    if ctx.video_processor:
        ctx.video_processor.painter.clear()
        st.success("Canvas cleared!")

st.markdown("### 📖 How to use")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("☝️ **1 Finger**\nDraw mode")
with col2:
    st.info("✌️ **2 Fingers**\nSelect Color")
with col3:
    st.warning("🖐️ **Open Hand**\nErase mode")
with col4:
    st.error("✊ **Fist**\nIdle / Rest")
