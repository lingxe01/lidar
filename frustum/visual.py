import matplotlib.image as mpimg
import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import MeanShift,estimate_bandwidth
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN,KMeans

def pixel_to_normalized(x, y, cx, cy, fx, fy):
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy
    return x_norm, y_norm


def normalized_to_camera(x_norm, y_norm, Z):
    X = x_norm * Z
    Y = y_norm * Z
    return X, Y, Z


def camera_to_lidar(X, Y, Z, Tr_velo_to_cam, R0_rect):
    camera_coords = np.array([X, Y, Z, 1.0]).reshape(4, 1)
    R0_rect_4x4 = np.eye(4)
    R0_rect_4x4[:3, :3] = R0_rect

    # 首先应用R0_rect
    rectified_coords = R0_rect_4x4 @ camera_coords
    # 然后应用Tr_velo_to_cam的逆
    transform_matrix_inv = np.linalg.inv(Tr_velo_to_cam)
    lidar_coords = transform_matrix_inv @ rectified_coords
    return lidar_coords.flatten()[:3]


# 计算平面方程：Ax + By + Cz + D = 0
def compute_plane_equation(p1, p2, p3):
    normal = np.cross(p2 - p1, p3 - p1)
    normal = normal / np.linalg.norm(normal)
    D = -np.dot(normal, p1)
    return np.append(normal, D)


def get_calib(calib_path):
    with open(calib_path, 'r') as f:
        calib = f.readlines()

    P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
    K = np.array(P2[:12]).reshape(3, 4)[:3, :3]

    # 从字符串解析出从激光雷达到相机的变换矩阵
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)
    Tr = np.array([Tr_velo_to_cam[0, 3], Tr_velo_to_cam[1, 3], Tr_velo_to_cam[2, 3]])

    R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
    return P2, K, Tr_velo_to_cam, Tr, R0_rect

def lidar_to_camera(point, Tr_velo_to_cam, R0_rect):
    point_hom = np.append(point, 1).reshape(4, 1)
    R0_rect_4x4 = np.eye(4)
    R0_rect_4x4[:3, :3] = R0_rect
    R0_rect = R0_rect_4x4
    point_camera = R0_rect @ Tr_velo_to_cam @ point_hom
    return point_camera[:3].flatten()

def project_to_image(point_camera, P):
    point_2d = P @ np.append(point_camera, 1)
    point_2d = point_2d[:2] / point_2d[2]
    return point_2d

def is_point_in_box(point_2d, bbox):
    x, y = point_2d
    left, top, right, bottom = bbox
    return left <= x <= right and top <= y <= bottom

# def filter_points_in_frustum(points, P, Tr_velo_to_cam, R0_rect, bbox, z_range, scale_factor=1.05):
#     inside_frustum = []
    
#     # 扩大2D边界框
#     left, top, right, bottom = bbox
#     center_x, center_y = (left + right) / 2, (top + bottom) / 2
#     width, height = right - left, bottom - top
    
#     scaled_width, scaled_height = width * scale_factor, height * scale_factor
#     scaled_left = center_x - scaled_width / 2
#     scaled_right = center_x + scaled_width / 2
#     scaled_top = center_y - scaled_height / 2
#     scaled_bottom = center_y + scaled_height / 2
    
#     scaled_bbox = [scaled_left, scaled_top, scaled_right, scaled_bottom]
    
#     # 扩大深度范围
#     z_near, z_far = z_range
#     z_center = (z_near + z_far) / 2
#     z_range = (z_center - (z_center - z_near) * scale_factor,
#                z_center + (z_far - z_center) * scale_factor)

#     for i, point in enumerate(points):
#         # 转换到相机坐标系
#         point_camera = lidar_to_camera(point, Tr_velo_to_cam, R0_rect)
        
#         # 检查深度
#         if not (z_range[0] <= point_camera[2] <= z_range[1]):
#             continue
        
#         # 投影到图像平面
#         point_2d = project_to_image(point_camera, P)
        
#         # 检查是否在扩大后的2D边界框内
#         if is_point_in_box(point_2d, scaled_bbox):
#             inside_frustum.append(i)
    
#     return inside_frustum

def filter_points_in_frustum(points, P, Tr_velo_to_cam, R0_rect, bbox, z_range, scale_factor=1.05, neighborhood_radius=0.8, threshold_count=5):
    inside_frustum = []
    
    # 扩大2D边界框
    left, top, right, bottom = bbox
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    width, height = right - left, bottom - top
    
    scaled_width, scaled_height = width * scale_factor, height * scale_factor
    scaled_left = center_x - scaled_width / 2
    scaled_right = center_x + scaled_width / 2
    scaled_top = center_y - scaled_height / 2
    scaled_bottom = center_y + scaled_height / 2
    
    scaled_bbox = [scaled_left, scaled_top, scaled_right, scaled_bottom]
    
    # 扩大深度范围
    z_near, z_far = z_range
    z_center = (z_near + z_far) / 2
    z_range = (z_center - (z_center - z_near) * scale_factor,
               z_center + (z_far - z_center) * scale_factor)

    for i, point in enumerate(points):
        # 转换到相机坐标系
        point_camera = lidar_to_camera(point, Tr_velo_to_cam, R0_rect)
        
        # 检查深度
        if not (z_range[0] <= point_camera[2] <= z_range[1]):
            continue
        
        # 投影到图像平面
        point_2d = project_to_image(point_camera, P)
        
        # 检查是否在扩大后的2D边界框内
        if is_point_in_box(point_2d, scaled_bbox):
            inside_frustum.append(i)

    # 扩散步骤
    points_array = np.array(points)
    tree = cKDTree(points_array)
    
    added_points = set(inside_frustum)
    new_points = set(inside_frustum)
    
    while new_points:
        current_points = new_points
        new_points = set()
        
        for idx in current_points:
            neighbors = tree.query_ball_point(points_array[idx], r=neighborhood_radius)
            
            for neighbor in neighbors:
                if neighbor not in added_points:
                    neighbor_count = len(set(tree.query_ball_point(points_array[neighbor], r=neighborhood_radius)) & added_points)
                    
                    if neighbor_count >= threshold_count:
                        new_points.add(neighbor)
                        added_points.add(neighbor)
    
    return list(added_points)

def add_bounding_boxes(pcd, labels, max_dimensions=(4.0, 3.0, 2.0),min_dimensions=(0.5, 0.5, 0.5), orientation='z', angle=355):
    aabbs = []
    for i in range(max(labels) + 1):
        # 获取属于当前聚类的所有点
        in_cluster = np.where(labels == i)[0]
        if len(in_cluster) > 0:
            cluster_points = np.asarray(pcd.points)[in_cluster]
            # 计算边界盒
            min_bound = cluster_points.min(axis=0)
            max_bound = cluster_points.max(axis=0)
            # 检查边界盒的尺寸
            dimensions = max_bound - min_bound
            if np.any(dimensions > np.array(max_dimensions)) or np.any(dimensions < np.array(min_dimensions)):
                continue
            else:
                # 创建包围盒
                center = (min_bound + max_bound) / 2
                if orientation == 'x':
                    # 对齐X轴
                    R = np.eye(3)
                elif orientation == 'y':
                    # 对齐Y轴
                    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                else:
                    # 固定角度旋转
                    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, angle))
                obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=dimensions)
                aabbs.append(obb)
                
    return aabbs

def db_cluster_points(points, colors,inside_points, epsilon=0.3, min_samples=10):
    # 创建一个点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 提取红色点的位置和颜色
    red_points = np.array(pcd.points)[inside_points]
    red_colors = np.array(pcd.colors)[inside_points]

    # 对红色的点执行DBSCAN聚类
    db = DBSCAN(eps=epsilon, min_samples=min_samples)
    db.fit(red_points)
    labels = db.labels_

    # 创建一个新的点云用于存储聚类结果
    red_pcd = o3d.geometry.PointCloud()
    red_pcd.points = o3d.utility.Vector3dVector(red_points)
    red_pcd.colors = o3d.utility.Vector3dVector(red_colors)
    max_label = labels.max()
    print(f"red point cloud has {max_label + 1} clusters")

    # 根据聚类标签为红色点分配颜色
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # 将噪声点设置为黑色
    cluster_colors = colors[:, :3]

    # 使用已知的红色点索引来更新原始点云的颜色
    for i, index in enumerate(inside_points):
        pcd.colors[index] = cluster_colors[i]

    return pcd,red_pcd,labels


calib_path = 'point\\000008\\000008.txt'
# 读取图像尺寸
img_path = f'point\\000008\save\\000008.png'
img = cv2.imread(img_path)
IMG_H, IMG_W, _ = img.shape
cv2.imshow("origin image", img)

P2, K, Tr_velo_to_cam, Tr, R0_rect = get_calib(calib_path)

binary = f'point\\000008\point_seg\pcd_nongrond_seg.pcd'
pcd = o3d.io.read_point_cloud(binary)
print("原始点云点数:", len(pcd.points))
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd.select_by_index(ind)
print("滤波后点云点数:", len(pcd.points))
original_colors = np.asarray(pcd.colors)
original_pts = np.asarray(pcd.points)

# print("Color value range:", np.min(original_colors), np.max(original_colors))

# 检查原始点云数据
# print(f"Original point cloud shape: {original_pts.shape}")

# 筛选有效点云
valid_indices = original_pts[:, 0]>0
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


# with open(f'point\\000008\save\labels\\000008.txt', 'r') as f:
with open(f'frustum/000008.txt', 'r') as f:
    detections = [line.strip().split() for line in f.readlines()]

detections = [[float(x) for x in det] for det in detections]

# 相机内参
fx, fy = P2[0, 0], P2[1, 1]
cx, cy = P2[0, 2], P2[1, 2]

car_lenth = 3
# 远近平面
# Z_near, Z_far = 1,15

# 用于存储所有视锥线条的列表
all_lines = []
all_frustum_points = []
all_planes = []
select_num=0
inside_points=[]
scale_factor = 1.1
for det in detections:
    category, u_center, v_center, width, height, conf,distance = det
    if distance>30 or distance<=0:
        continue
    Z_near = 1
    Z_far = distance+car_lenth
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
    inside_frustum_indices = filter_points_in_frustum(valid_pts, P2, Tr_velo_to_cam, R0_rect, bbox, z_range, scale_factor)
    inside_points.extend(inside_frustum_indices)

    # 更新点云颜色
    for i in inside_frustum_indices:
        valid_colors[i] = [1, 0, 0]  # 将视锥体内的点设置为红色
    select_num+=len(inside_frustum_indices)
    # break

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
inside_pcd = pcd_selected.select_by_index(inside_points)
# geometry_list = [pcd_selected,line_set]

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

all_clustered_pcd,clustered_pcd,labels = db_cluster_points(pcd_selected.points,pcd_selected.colors,inside_points,0.8,10)
aabb_list=add_bounding_boxes(clustered_pcd,labels,(4,3,2))
geometry_list = [all_clustered_pcd,line_set]
# 显示点云和视锥
o3d.visualization.draw_geometries(geometry_list+aabb_list)
