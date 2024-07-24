import open3d as o3d
import numpy as np
import cv2


def preprocess_point_cloud(pcd, voxel_size):
    """
    Args:
        pcd: 待采样点云
        voxel_size: 体素大小

    Returns:采样后的点云
    """
    # 对点云进行下采样
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 计算法向量
    pcd_down.estimate_normals()

    return pcd_down


def get_slope(normal_1, normal_2):
    cos_theta = np.dot(normal_1, normal_2) / (
            np.linalg.norm(normal_1) * np.linalg.norm(normal_2))
    angle = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle)
    return angle_degrees


def segment_planes(pcd, distance_threshold=0.05, ransac_n=5,
                   num_iterations=3000):
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    [a, b, c, d] = plane_model
    normals = np.asarray([a, b, c])
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud, normals


# 进行均值滤波处理
def mean_filter(point_cloud, radius=0.05):
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    points = np.asarray(point_cloud.points)
    filtered_points = np.zeros_like(points)

    for i in range(points.shape[0]):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point_cloud.points[i], radius)
        filtered_points[i] = np.mean(points[idx], axis=0)

    point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    return point_cloud


# 中值滤波
def median_filter(point_cloud, radius=0.05):
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    points = np.asarray(point_cloud.points)
    filtered_points = np.zeros_like(points)

    for i in range(points.shape[0]):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point_cloud.points[i], radius)
        filtered_points[i] = np.median(points[idx], axis=0)

    point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    return point_cloud


def bilateral_filter(pcd, points, sigma_s=0.05, sigma_r=0.05, radius=0.1):

    def gaussian(x, sigma):
        return np.exp(-0.5 * (x / sigma) ** 2)

    filtered_points = np.zeros_like(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    for i in range(points.shape[0]):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(points[i], radius)
        neighbors = points[idx, :]

        spatial_weights = gaussian(np.linalg.norm(neighbors - points[i], axis=1), sigma_s)
        intensity_weights = gaussian(np.linalg.norm(np.linalg.norm(points[i]) - np.linalg.norm(neighbors, axis=1)),
                                     sigma_r)

        bilateral_weights = spatial_weights * intensity_weights
        bilateral_weights /= np.sum(bilateral_weights)

        filtered_points[i] = np.sum(neighbors * bilateral_weights[:, np.newaxis], axis=0)

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_pcd


def calculate_z_mean(point_cloud):
    """Calculate the mean of the Z-axis values of a point cloud."""
    points = np.asarray(point_cloud.points)
    return np.mean(points[:, 2])


def classify_point_clouds(pcds, normals):
    """Classify point clouds based on the mean Z value."""
    clusters = []
    for pc, normal in zip(pcds, normals):
        z_mean = calculate_z_mean(pc)
        clusters.append((z_mean, pc, normal))

    # Sort clusters based on the Z mean value
    clusters_sorted = sorted(clusters, key=lambda x: x[0], reverse=True)

    return clusters_sorted


def add_noise_to_point_cloud(pcd, num_noise_points=100, noise_range=1):
    """
    在点云数据中添加噪声离群点

    参数:
    - pcd: open3d.geometry.PointCloud 对象
    - num_noise_points: 添加的噪声点数量
    - noise_range: 噪声点的坐标范围 (相对于点云数据的范围)

    返回:
    - 含有噪声的点云数据
    """
    points = np.asarray(pcd.points)

    # 点云数据的边界范围
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # 随机生成噪声点
    noise_points = np.random.uniform(min_bound - noise_range, max_bound + noise_range, size=(num_noise_points, 3))

    # 将噪声点添加到点云数据中
    noisy_points = np.vstack((points, noise_points))

    # 创建新的点云对象
    noisy_pcd = o3d.geometry.PointCloud()
    noisy_pcd.points = o3d.utility.Vector3dVector(noisy_points)

    return noisy_pcd


def remove_outliers_combined(pcd, nb_neighbors=20, std_ratio=2.0, radius=0.1, min_points=10):
    """
    结合多种方法剔除点云数据中的离群点

    参数:
    - pcd: open3d.geometry.PointCloud 对象
    - nb_neighbors: 统计滤波中计算每个点的邻居点数量
    - std_ratio: 统计滤波中标准差乘数，决定滤波的严格程度
    - radius: 半径滤波中固定半径
    - min_points: 半径滤波中固定半径内最少的邻居点数量

    返回:
    - 剔除离群点后的点云数据
    """
    # 统计滤波
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)

    # 半径滤波
    cl, ind = inlier_cloud.remove_radius_outlier(nb_points=min_points, radius=radius)
    inlier_cloud = inlier_cloud.select_by_index(ind)

    return inlier_cloud