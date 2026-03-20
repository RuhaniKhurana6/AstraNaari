import cv2

def start_video_capture():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    return cap