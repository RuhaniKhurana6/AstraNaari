import cv2

def start_video_capture():
    # Try multiple indices just in case
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Successfully opened camera {idx}")
            return cap
    return cv2.VideoCapture(0) # Fallback to 0 if none opened