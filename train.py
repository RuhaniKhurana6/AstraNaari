from ultralytics import YOLO

# load pretrained model
model = YOLO("yolov8n.pt")

# train model
model.train(
    data="data.yaml",
    epochs=15,
    imgsz=640,
    batch=8
)