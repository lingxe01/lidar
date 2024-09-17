from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model('point\\000008\\000008.png',show=True,save=False)
for result in results:
    print(result.boxes.xywhn)