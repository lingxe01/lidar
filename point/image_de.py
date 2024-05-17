from ultralytics import YOLO

model = YOLO('D:\lidar\yolov5mu.pt')

model.predict('D:\lidar\point\\000008\\000008.png',conf=0.65,show=True,save=True,save_txt=True,save_conf=True,show_conf=False,show_labels=False,name='D:\lidar\point\\000008\save',exist_ok=True)