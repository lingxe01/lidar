import cv2

def draw_bounding_boxes(image_path, label_file):
    # 读取图像
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    with open(label_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        label = parts[0]
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height

        # 计算检测框的左上角和右下角坐标
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # 绘制检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 在检测框上绘制标签
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例用法
image_path = 'D:\lidar\point\\000008\save\\000008.png'
label_file = 'D:\lidar\point\\000008_yolo_label.txt'

draw_bounding_boxes(image_path, label_file)
