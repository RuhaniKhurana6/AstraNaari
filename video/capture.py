import cv2
import threading
import time

class ThreadedCamera:
    """
    Multithreaded camera capture to prevent the UI from blocking 
    while waiting for the webcam or YOLO inference.
    """
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        self.FPS = 1/30
        self.status, self.frame = self.capture.read()
        self.stopped = False
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(self.FPS)

    def read(self):
        return self.status, self.frame

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        self.capture.release()

def start_video_capture():
    # Backward compatibility for main.py
    return ThreadedCamera().start()