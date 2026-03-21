import os
from ultralytics import YOLO

MODEL_PATH = "runs/detect/train7/weights/best.pt"
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"[AstraNaari] Loaded custom model: {MODEL_PATH}")
else:
    print(f"[AstraNaari] Custom model not found. Using yolov8n.pt (detects knives/scissors/bats).")
    model = YOLO("yolov8n.pt")

# COCO class IDs for sharp / impact objects
# 43 = knife, 76 = scissors, 73 = baseball bat
WEAPON_CLASSES = {43: "Knife", 76: "Scissors", 73: "Baseball Bat"}

detection_count = 0

def detect_weapon(frame):
    global detection_count

    results = model(frame, imgsz=320, verbose=False)

    if not results or results[0].boxes is None:
        detection_count = max(detection_count - 1, 0)
        return False, None, 0

    best_box = None
    best_conf = 0

    for box in results[0].boxes:
        conf = float(box.conf)
        cls = int(box.cls[0]) if box.cls is not None else -1

        if os.path.exists(MODEL_PATH):
            # Custom model: class 0 = weapon
            is_weapon = (cls == 0)
        else:
            # Fallback: knife / scissors / bat
            is_weapon = cls in WEAPON_CLASSES

        if is_weapon and conf > best_conf:
            best_conf = conf
            best_box = box

    if best_conf > 0.20 and best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        detection_count = min(detection_count + 1, 5)
        if detection_count >= 2:   # require 2 consecutive frames
            return True, (x1, y1, x2, y2), best_conf
    else:
        detection_count = max(detection_count - 1, 0)

    return False, None, 0