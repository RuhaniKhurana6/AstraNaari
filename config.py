import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best.pt")

# Fallback
FALLBACK_MODEL = os.path.join(PROJECT_DIR, "yolov8n.pt")

# --- CAMERA CONFIG ---
CAMERA_SOURCE = 0
LOCATION_NAME = "Room 1"

# --- ML INFERENCE CONFIG ---
IMG_SIZE = 640
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
MAX_DET = 10
HIGH_RISK_THRESHOLD = 0.8
MEDIUM_RISK_THRESHOLD = 0.65
FRAME_SKIP = 2
VALID_CLASSES = [0, 1]


def get_model_path():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    else:
        print(f"⚠️ Trained model not found at {MODEL_PATH}, using fallback model")
        return FALLBACK_MODEL