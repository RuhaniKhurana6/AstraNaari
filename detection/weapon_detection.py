from ultralytics import YOLO
from config import get_model_path, IMG_SIZE, CONF_THRESHOLD, IOU_THRESHOLD, MAX_DET

class WeaponDetector:
    """
    Weapon detector class featuring:
    1. Hysteresis (Sticky Bounding Boxes) to stop UI flickering.
    2. Configurable temporal thresholding.
    3. Graceful model error fallbacks.
    """
    def __init__(self, required_consecutive_frames=1, cooldown_frames=5):
        self.detection_count = 0
        self.required_frames = required_consecutive_frames
        
        # Hysteresis fields to stop UI flickering
        self.cooldown_frames = cooldown_frames
        self.missed_frames = 0
        self.last_valid_boxes = []
        self.last_best_conf = 0.0
        
        try:
            self.model = YOLO(get_model_path())
        except Exception as e:
            print(f"CRITICAL ERROR - Failed to load YOLO model: {e}")
            self.model = None

    def detect_weapon(self, frame):
        if self.model is None or frame is None:
            return False, [], 0
            
        try:
            results = self.model(frame, imgsz=IMG_SIZE, iou=IOU_THRESHOLD, max_det=MAX_DET, verbose=False)
        except Exception as e:
            print(f"WARNING - Inference failed on frame: {e}")
            return False, [], 0

        valid_boxes = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf)
                cls = int(box.cls)

                if cls not in [0, 1]:
                    continue

                if conf > CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    valid_boxes.append(((x1, y1, x2, y2), conf))

        if valid_boxes:
            # We actively see a weapon! Reset the miss timer and cache the boxes.
            self.missed_frames = 0
            self.last_valid_boxes = valid_boxes
            self.last_best_conf = max([conf for (_, conf) in valid_boxes])
            
            self.detection_count = min(self.detection_count + 1, self.required_frames + 5)

            if self.detection_count >= self.required_frames:
                return True, self.last_valid_boxes, self.last_best_conf
        else:
            # We DO NOT see a weapon on this exact frame
            if self.detection_count >= self.required_frames:
                # We were in extreme DANGER state. Trigger sticky visual cooldown (hysteresis)
                self.missed_frames += 1
                
                if self.missed_frames <= self.cooldown_frames:
                    # Return the last known boxes so the UI doesn't flicker violently
                    return True, self.last_valid_boxes, self.last_best_conf
                else:
                    # Object has truly been gone for too long. Reset state.
                    self.detection_count = 0
            else:
                self.detection_count = max(0, self.detection_count - 1)

        return False, [], 0