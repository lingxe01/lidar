import cv2

# IP摄像头的URL，假设是MJPEG流
url = "rtsp://admin:123@192.168.1.37:554/mainstream"
# 打开视频流
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# 检查视频流是否成功打开
if not cap.isOpened():
    print("无法打开视频流")
    exit()

# 设置视频捕获的缓冲区大小为0，降低延迟
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

# 获取视频的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义编解码器并创建 VideoWriter 对象
output_filename = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

# 循环读取和显示视频帧
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧（可能是流结束了）")
        break

    # 写入当前帧到视频文件
    out.write(frame)

    # 显示帧
    cv2.imshow('Video Stream', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕捉对象和视频写入对象，并关闭所有OpenCV窗口
cap.release()
out.release()
cv2.destroyAllWindows()
