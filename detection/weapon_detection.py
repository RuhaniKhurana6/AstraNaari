from ultralytics import YOLO
from config import get_model_path

# Load model correctly
model = YOLO(get_model_path())
detection_count = 0

def detect_weapon(frame):
    global detection_count

    results = model(frame, imgsz=320, verbose=False)

    # safety check
    if not results or results[0].boxes is None:
        return False, None, 0

    best_box = None
    best_conf = 0

    # Find best detection
    for box in results[0].boxes:
        conf = float(box.conf)
        cls = int(box.cls)   # 🔥 get class id

        if cls != 0:
            continue

        if conf > best_conf:
            best_conf = conf
            best_box = box

    # threshold
    if best_conf > 0.65 and best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        width = x2 - x1
        height = y2 - y1
        area = width * height

        # filtering
        if area < 3500:
            return False, 

        # stable increment
        detection_count = min(detection_count + 1, 5)

        if detection_count >= 2:
            return True, (x1, y1, x2, y2), best_conf

    else:
        # smooth decay
        if detection_count > 0:
            detection_count -= 1

    return False, None, 0