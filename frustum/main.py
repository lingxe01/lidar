import matplotlib.image as mpimg
import numpy as np
import open3d as o3d
from function import *

calib_path = 'D:\lidar\point\\000008\\000008.txt'
img_path = 'D:\lidar\point\\000008\save\\000008.png'
binary = 'D:\lidar\point\\000008\point_seg\pcd_nongrond_seg.pcd'
label_path = 'D:\lidar\point\\000008\save\labels\\000008.txt'
img = mpimg.imread(img_path)
IMG_H, IMG_W, _ = img.shape

P2, K, Tr_velo_to_cam, Tr, R0_rect = get_calib(calib_path)

pcd = o3d.io.read_point_cloud(binary)
original_colors = np.asarray(pcd.colors)
original_pts = np.asarray(pcd.points)

# 筛选有效点云
valid_indices = original_pts[:, 0] > 0
valid_pts = original_pts[valid_indices]
gray_color = np.array([0.7, 0.7, 0.7])  # 灰色
valid_colors = np.tile(gray_color, (len(valid_pts), 1))

# 创建筛选后的点云
pcd_selected = o3d.geometry.PointCloud()
pcd_selected.points = o3d.utility.Vector3dVector(valid_pts)
pcd_selected.colors = o3d.utility.Vector3dVector(valid_colors)

with open(label_path, 'r') as f:
    detections = [line.strip().split() for line in f.readlines()]

detections = [[float(x) for x in det] for det in detections]

# 相机内参
fx, fy = P2[0, 0], P2[1, 1]
cx, cy = P2[0, 2], P2[1, 2]

# 远近平面
Z_near, Z_far = 1, 10

scale_factor=1.05

inside_points=[]
for det in detections:
    _, u_center, v_center, width, height, _ = det
    left_top = ((u_center - width / 2) * IMG_W, (v_center - height / 2) * IMG_H)
    right_top = ((u_center + width / 2) * IMG_W, (v_center - height / 2) * IMG_H)
    left_bottom = ((u_center - width / 2) * IMG_W, (v_center + height / 2) * IMG_H)
    right_bottom = ((u_center + width / 2) * IMG_W, (v_center + height / 2) * IMG_H)
    bbox = [left_top, right_top, left_bottom, right_bottom]

    # 判断点云中的点是否在视锥体内
    # 使用这个函数
    bbox = [left_top[0], left_top[1], right_bottom[0], right_bottom[1]]
    z_range = [Z_near, Z_far]
    inside_frustum_indices = filter_points_in_frustum(valid_pts, P2, Tr_velo_to_cam, R0_rect, bbox, z_range, scale_factor)
    inside_points.extend(inside_frustum_indices)

# 更新点云颜色
valid_colors[inside_points] = [1, 0, 0]  # 将视锥体内的点设置为红色

pcd_selected.colors = o3d.utility.Vector3dVector(valid_colors)

print(f"Number of points inside frustum: {len(inside_points)}")

geometry_list = [pcd_selected]

o3d.visualization.draw_geometries(geometry_list)