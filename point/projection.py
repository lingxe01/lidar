import numpy as np
import open3d as o3d
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import open3d as o3d
from ultralytics import YOLO
import shutil
def yolo_txt(path,name):

    model = YOLO("yolov5m.pt")
    model.predict(source=f'{path}/{name}/{name}.png',save=True,name=f'{path}/{name}/save',save_txt=True,conf=0.6,exist_ok=True)

def projection(path,name):
    # path = 'D:\lidar\point\\000008'
    # name = '000008'
    # yolo_txt(name,path)
    # 读取原始点云数据
    # name = '000008'
    binary = f'{path}/{name}/{name}.bin'
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1, 4))
    original_pts = scan[:, :3]  # 提取 (x, y, z) 坐标
    original_pts = np.array(original_pts)[original_pts[:, 0] > 0]
    pcd_selected = o3d.geometry.PointCloud()
    pcd_selected.points = o3d.utility.Vector3dVector(original_pts)

    # 显示筛选后的点云
    o3d.visualization.draw_geometries([pcd_selected])

    # 读取相机参数
    with open(f'{path}/{name}/{name}.txt','r') as f:
        calib = f.readlines()

    # P2 (3 x 4) for left eye
    P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
    R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

    # 读取图像尺寸
    img_path = f'{path}/{name}/save/{name}.png'
    img = mpimg.imread(img_path)
    IMG_H, IMG_W, _ = img.shape

    # 读取检测框信息
    with open(f'{path}/{name}/save/labels/{name}.txt', 'r') as f:
        detections = [line.strip().split() for line in f.readlines()]

    detections = [[float(x) for x in det] for det in detections]

    # 投影点云到图像
    velo = np.insert(original_pts, 3, 1, axis=1).T
    cam = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(velo)))
    cam = np.delete(cam, np.where(cam[2, :] < 0), axis=1)
    cam[:2] /= cam[2, :]

    # 获取落在检测框中的投影点云索引
    projected_indices = []
    for det in detections:
        _, u_center, v_center, width, height = det
        u_min = (u_center - width / 2) * IMG_W
        u_max = (u_center + width / 2) * IMG_W
        v_min = (v_center - height / 2) * IMG_H
        v_max = (v_center + height / 2) * IMG_H
        u, v = cam[:2]
        u_in = np.logical_and(u >= u_min, u <= u_max)
        v_in = np.logical_and(v >= v_min, v <= v_max)
        in_box = np.logical_and(u_in, v_in)
        projected_indices.append(np.where(in_box)[0])

    # 合并所有投影点云索引
    projected_indices = np.concatenate(projected_indices)

    # 提取落在检测框中的投影点云数据
    u, v, z = cam
    projected_pts = np.stack([u[projected_indices], v[projected_indices]], axis=1)

    # 显示投影点云和原始图像
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.scatter(projected_pts[:, 0], projected_pts[:, 1], c='r', s=1)
    plt.title(f'Projected Point Cloud on {name}_detect.png')
    plt.savefig(f'{path}/{name}_projection_detect_filter.png',bbox_inches='tight')
    plt.show()

if __name__=='__main__':
    # yolo_txt('D:\lidar\point','000008')
    projection('D:\lidar\point','000008')