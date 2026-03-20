import cv2
import time

prev_time = 0

def process_frame(frame):
    global prev_time

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

    # Convert to RGB 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    #Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return frame, frame_rgb