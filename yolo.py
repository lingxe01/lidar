from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.predict(source="point/000008/000008.png",save=True,show=True,conf=0.6)