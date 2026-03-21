import os

# Model path (STRING ✅)
MODEL_PATH = "models/best.pt"

# Fallback
FALLBACK_MODEL = "yolov8n.pt"

def get_model_path():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    else:
        print("⚠️ Trained model not found, using fallback model")
        return FALLBACK_MODEL