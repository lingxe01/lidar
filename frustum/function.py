from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN,KMeans
import open3d as o3d
from scipy.spatial import cKDTree
from ultralytics import YOLO

def get_yolo_result_boxes(image_path):
    model = YOLO('yolov8n.pt')
    results = model(image_path,save=False)
    return results.boxes.xywhn


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

def filter_points_in_frustum(points, P, Tr_velo_to_cam, R0_rect, bbox, z_range, scale_factor=1.05, neighborhood_radius=0.5, threshold_count=5):
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
# dbscan聚类
def db_cluster_points(points, epsilon=0.3, min_samples=10):
    
    db = DBSCAN(eps=epsilon, min_samples=min_samples)
    db.fit(points)
    labels = db.labels_
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 根据聚类标签为点分配颜色
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # 将噪声点设置为黑色
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd


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
# k-means聚类
def k_means_cluster_points(point_cloud,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(point_cloud)

    # 获取聚类标签
    labels = kmeans.labels_

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 根据聚类标签为点分配颜色
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # 将噪声点设置为黑色
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd


if __name__=="__main__":
    print(0)