from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="C:/Users/Asus/OneDrive/Desktop/Hackathon/weapon_detection_project/data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,   # GPU
        workers=2,
        name="weapon_final"
    )

if __name__ == "__main__":
    main()