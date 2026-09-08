from ultralytics  import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data = ".github/agents/data/drone-detection-new.v5-new-train.yolov8/data.yaml",
    epochs=10
)