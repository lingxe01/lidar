import matplotlib.image as mpimg
import numpy as np
import open3d as o3d
from function import *

calib_path = 'D:\lidar\point\\000008\\000008.txt'
# 读取图像尺寸
img_path = f'D:\lidar\point\\000008\save\\000008.png'
img = mpimg.imread(img_path)
IMG_H, IMG_W, _ = img.shape

P2, K, Tr_velo_to_cam, Tr, R0_rect = get_calib(calib_path)

binary = f'D:\lidar\point\\000008\point_seg\pcd_nongrond_seg.pcd'
pcd = o3d.io.read_point_cloud(binary)
original_colors = np.asarray(pcd.colors)
original_pts = np.asarray(pcd.points)

# print("Color value range:", np.min(original_colors), np.max(original_colors))

# 检查原始点云数据
# print(f"Original point cloud shape: {original_pts.shape}")

# 筛选有效点云
valid_indices = original_pts[:, 0] > 0
valid_pts = original_pts[valid_indices]
# valid_colors = original_colors[valid_indices]
gray_color = np.array([0.7, 0.7, 0.7])  # 灰色
valid_colors = np.tile(gray_color, (len(valid_pts), 1))

# print(f"Valid point cloud shape: {valid_pts.shape}")

# 创建筛选后的点云
pcd_selected = o3d.geometry.PointCloud()
pcd_selected.points = o3d.utility.Vector3dVector(valid_pts)
# pcd_selected.paint_uniform_color([1, 0, 0])
# pcd_selected.colors = o3d.utility.Vector3dVector(valid_colors)

# 可视化带颜色的点云
# o3d.visualization.draw_geometries([pcd_selected])


with open(f'D:\lidar\point\\000008\save\labels\\000008.txt', 'r') as f:
    detections = [line.strip().split() for line in f.readlines()]

detections = [[float(x) for x in det] for det in detections]

# 相机内参
fx, fy = P2[0, 0], P2[1, 1]
cx, cy = P2[0, 2], P2[1, 2]

# 远近平面
Z_near, Z_far = 0, 5

# 用于存储所有视锥线条的列表
all_lines = []
all_frustum_points = []
all_planes = []
select_num=0
for det in detections:
    _, u_center, v_center, width, height, _ = det
    left_top = ((u_center - width / 2) * IMG_W, (v_center - height / 2) * IMG_H)
    right_top = ((u_center + width / 2) * IMG_W, (v_center - height / 2) * IMG_H)
    left_bottom = ((u_center - width / 2) * IMG_W, (v_center + height / 2) * IMG_H)
    right_bottom = ((u_center + width / 2) * IMG_W, (v_center + height / 2) * IMG_H)
    bbox = [left_top, right_top, left_bottom, right_bottom]
    frustum_points = []
    for (x, y) in bbox:
        x_norm, y_norm = pixel_to_normalized(x, y, cx, cy, fx, fy)
        for Z in [Z_near, Z_far]:
            X_cam, Y_cam, Z_cam = normalized_to_camera(x_norm, y_norm, Z)
            X_lidar, Y_lidar, Z_lidar = camera_to_lidar(X_cam, Y_cam, Z_cam, Tr_velo_to_cam, R0_rect)
            frustum_points.append(np.array([X_lidar, Y_lidar, Z_lidar]))

    frustum_points = np.array(frustum_points)
    base_index = len(all_frustum_points)
    all_frustum_points.extend(frustum_points)
    # 绘制视锥
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # 近平面
        [4, 5], [5, 7], [7, 6], [6, 4],  # 远平面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 连接线
    ]
    all_lines.extend([[base_index + line[0], base_index + line[1]] for line in lines])

    # Compute the plane equations for the frustum
    near_plane = compute_plane_equation(frustum_points[0], frustum_points[1], frustum_points[2])
    far_plane = compute_plane_equation(frustum_points[4], frustum_points[5], frustum_points[6])
    left_plane = compute_plane_equation(frustum_points[0], frustum_points[2], frustum_points[4])
    right_plane = compute_plane_equation(frustum_points[1], frustum_points[3], frustum_points[5])
    top_plane = compute_plane_equation(frustum_points[0], frustum_points[1], frustum_points[4])
    bottom_plane = compute_plane_equation(frustum_points[2], frustum_points[3], frustum_points[6])
    
    planes = [near_plane, far_plane, left_plane, right_plane, top_plane, bottom_plane]
    all_planes.append(planes)

# 判断点云中的点是否在视锥体内
# 使用这个函数
    bbox = [left_top[0], left_top[1], right_bottom[0], right_bottom[1]]
    z_range = [Z_near, Z_far]
    scale_factor=1.05
    inside_frustum_indices = filter_points_in_frustum(valid_pts, P2, Tr_velo_to_cam, R0_rect, bbox, z_range, scale_factor)

    # 更新点云颜色
    for i in inside_frustum_indices:
        valid_colors[i] = [1, 0, 0]  # 将视锥体内的点设置为红色
    select_num+=len(inside_frustum_indices)
    break

pcd_selected.colors = o3d.utility.Vector3dVector(valid_colors)

# # 创建一个新的点云对象，只包含视锥体内的点
# pcd_inside_frustum = pcd_selected.select_by_index(inside_frustum_indices)

print(f"Number of points inside frustum: {select_num}")

# 创建LineSet对象
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(np.vstack(all_frustum_points)),
    lines=o3d.utility.Vector2iVector(all_lines)
)
colors = [[0, 1, 0] for _ in range(len(all_lines))]
line_set.colors = o3d.utility.Vector3dVector(colors)
geometry_list = [pcd_selected,line_set]

# plane_polygons = []
# plane_polygons.append(frustum_points[[0, 1, 3, 2]])  # near plane
# plane_polygons.append(frustum_points[[4, 5, 7, 6]])  # far plane
# plane_polygons.append(frustum_points[[0, 2, 6, 4]])  # left plane
# plane_polygons.append(frustum_points[[1, 3, 7, 5]])  # right plane
# plane_polygons.append(frustum_points[[0, 1, 5, 4]])  # top plane
# plane_polygons.append(frustum_points[[2, 3, 7, 6]])  # bottom plane

# for poly in plane_polygons:
#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(poly)
#     mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
#     mesh.paint_uniform_color([0, 1, 0])  # Set the color to green
#     geometry_list.append(mesh)
# 显示点云和视锥
o3d.visualization.draw_geometries(geometry_list)
