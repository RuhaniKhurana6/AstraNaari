import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PROJECT_DIR)

# Model path
MODEL_PATH = os.path.join(ROOT_DIR, "runs", "detect", "weapon_final7", "weights", "best.pt")

# Fallback
FALLBACK_MODEL = os.path.join(PROJECT_DIR, "yolov8n.pt")

# --- ML INFERENCE CONFIG ---
IMG_SIZE = 640
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
MAX_DET = 10

def get_model_path():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    else:
        print(f"⚠️ Trained model not found at {MODEL_PATH}, using fallback model")
        return FALLBACK_MODEL