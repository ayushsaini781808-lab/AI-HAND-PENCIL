import cv2
import numpy as np
import gradio as gr
from config import CURRENT_COLOR, TOOLBAR_HEIGHT
from hand_tracker import HandTracker
from gesture import get_finger_states, detect_gesture, GestureStabilizer
from painter import Painter
from utils import CoordinateSmoother
from ui import draw_toolbar, select_color, draw_mode, draw_cursor, draw_eraser_cursor, draw_coordinates

# Initialize global components (for a simple single-user HF space demo)
# In a robust multi-user app, you'd wrap this in a state object per session.
hand_tracker = HandTracker()
gesture_stabilizer = GestureStabilizer()
smoother = CoordinateSmoother()
painter = Painter()
current_color = CURRENT_COLOR

def process_frame(frame):
    global current_color
    
    if frame is None:
        return None
        
    # Flip the frame natively (just in case Gradio doesn't)
    frame = cv2.flip(frame, 1)

    painter.create_canvas(frame)
    landmarks, hand_type = hand_tracker.detect(frame)
    draw_toolbar(frame, current_color)

    if landmarks is not None:
        fingers = get_finger_states(landmarks, hand_type)
        raw_gesture = detect_gesture(fingers)
        stable_gesture = gesture_stabilizer.update(raw_gesture)

        raw_x, raw_y = landmarks[8]
        x, y = smoother.smooth(raw_x, raw_y)
        current_point = (x, y)

        draw_mode(frame, stable_gesture)
        draw_coordinates(frame, current_point)

        if stable_gesture == "DRAW":
            draw_cursor(frame, current_point, current_color)
            if y > TOOLBAR_HEIGHT:
                painter.draw(current_point, current_color)
            else:
                painter.reset_previous()
                
        elif stable_gesture == "SELECT":
            painter.reset_previous()
            current_color = select_color(current_point, current_color)
            
        elif stable_gesture == "ERASE":
            painter.reset_previous()
            draw_eraser_cursor(frame, current_point)
            if y > TOOLBAR_HEIGHT:
                painter.erase(current_point)
                
        else:
            painter.reset_previous()
    else:
        painter.reset_previous()
        smoother.reset()

    # Redraw toolbar to update selected color
    draw_toolbar(frame, current_color)
    output = painter.combine(frame)
    
    return output

def clear_canvas():
    painter.clear()
    return "Canvas Cleared!"

# Build the Gradio App
with gr.Blocks(title="AI Hand Pencil") as app:
    gr.Markdown("# 🖐️ AI Hand Pencil")
    gr.Markdown("Control the pencil with your hand gestures. **DRAW** (1 finger), **ERASE** (whole hand). Draw with your index finger.")
    
    with gr.Row():
        webcam = gr.Image(sources=["webcam"], streaming=True)
        img_out = gr.Image()
        
    webcam.stream(fn=process_frame, inputs=[webcam], outputs=[img_out])
    
    clear_btn = gr.Button("Clear Canvas")
    clear_msg = gr.Markdown("")
    clear_btn.click(fn=clear_canvas, outputs=[clear_msg])

if __name__ == "__main__":
    app.launch()
