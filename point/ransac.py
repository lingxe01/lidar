import open3d as o3d
import numpy as np
from PIL import Image
from fps import farthest_point_sample


def read_kitti_bin(bin_path):
    # 读取KITTI bin格式点云数据
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, 0:3]  # 提取x,y,z坐标值

    # 将点云数据转换为Open3D格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    #对点云进行下采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    #计算法向量
    pcd_down.estimate_normals()
    
    return pcd_down


def ransac_plane_segmentation(pcd, distance_threshold, ransac_n, num_iterations):
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    
    # 使用 Open3D 内置索引机制获取内群和离群点云
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    return inlier_cloud, outlier_cloud

# def region_growing_plane_segmentation(pcd, distance_threshold, curvature_threshold):
#     # 基于法向量和曲率做区域生长,细分割地面
#     outlier_cloud = pcd
#     cluster_labels = np.array(outlier_cloud.cluster_dbscan(eps=distance_threshold, 
#                                                             min_points=20, 
#                                                             print_progress=True))

#     # 获取每个聚类的点的索引列表
#     unique_labels = np.unique(cluster_labels)
#     plane_clouds = []
#     for label in unique_labels:
#         if label == -1:  # 跳过噪声点
#             continue
#         indices = np.where(cluster_labels == label)[0]
#         curr_plane = outlier_cloud.select_by_index(indices.tolist())
#         curr_plane, plane_inliers = curr_plane.segment_plane(distance_threshold, 
#                                                               ransac_n=3, 
#                                                               num_iterations=1000)
#         if isinstance(curr_plane, o3d.geometry.PointCloud):
#             plane_points = np.asarray(curr_plane.points)
#         else:
#     # 处理只剩一个点的情况或跳过该情况
#             continue
#         curvatures = curr_plane.estimate_curvature()
#         curvature = np.mean(curvatures)
#         if curvature < curvature_threshold:
#             plane_clouds.append(curr_plane)
            
#     return plane_clouds

def region_growing_plane_segmentation(pcd, distance_threshold, curvature_threshold, min_cluster_size=10, max_z_range=0.5):
    # 基于法向量和曲率做区域生长,细分割地面
    outlier_cloud = pcd
    cluster_labels = np.array(outlier_cloud.cluster_dbscan(eps=distance_threshold, min_points=20, print_progress=True))
    # 获取每个聚类的点的索引列表
    unique_labels = np.unique(cluster_labels)
    plane_clouds = []
    for label in unique_labels:
        if label == -1:
            # 跳过噪声点
            continue
        indices = np.where(cluster_labels == label)[0]
        curr_plane = outlier_cloud.select_by_index(indices.tolist())
        # 如果当前聚类的点数量小于阈值,视为噪声点,跳过
        if len(curr_plane.points) < min_cluster_size:
            continue
        curr_plane, plane_inliers = curr_plane.segment_plane(distance_threshold, ransac_n=3, num_iterations=1000)
        if isinstance(curr_plane, o3d.geometry.PointCloud):
            plane_points = np.asarray(curr_plane.points)
        else:
            # 处理只剩一个点的情况或跳过该情况
            continue
        curvatures = curr_plane.estimate_curvature()
        curvature = np.mean(curvatures)
        if curvature < curvature_threshold:
            # 计算当前平面点云的Z坐标范围
            z_values = plane_points[:, 2]
            z_max = np.max(z_values)
            z_min = np.min(z_values)
            z_range = z_max - z_min
            # 如果Z坐标范围小于阈值,认为是地面点云
            if z_range < max_z_range:
                plane_clouds.append(curr_plane)
    return plane_clouds

def ground_plane_segmentation(pcd, voxel_size=0.05):
    #预处理 - 下采样和法向量估计
    pcd_down = preprocess_point_cloud(pcd, voxel_size)
    
    #RANSAC粗分割
    inlier_cloud, outlier_cloud = ransac_plane_segmentation(pcd_down,
                                                           0.07, 3, 5000)
    
    #区域生长细分割
    plane_clouds = region_growing_plane_segmentation(outlier_cloud, 0.07, 0.05)
    
    #合并地面点云
    ground_cloud = inlier_cloud
    for plane in plane_clouds:
        ground_cloud += plane
    
    #着色显示结果
    ground_cloud.paint_uniform_color([1.0, 0, 0])
    non_ground_cloud = outlier_cloud
    non_ground_cloud.paint_uniform_color([0, 1.0, 0])
    
    o3d.visualization.draw_geometries([ground_cloud, non_ground_cloud])
    # o3d.visualization.draw_geometries([non_ground_cloud])
    return ground_cloud,non_ground_cloud


if __name__ == '__main__':
    # 假设KITTI bin文件是000000.bin
    pcd = read_kitti_bin('/home/ling/mmdetection3d/data/kitti/training/velodyne/000008.bin')
    # o3d.visualization.draw_geometries([pcd])
    points = np.asarray(pcd.points)
    points_num = len(points)
    points = points.reshape([1,points_num,3])

    centroids = farthest_point_sample(points,50000)
    # print(sample_cloud.shape)
    sample_cloud = points[0,centroids[0].cpu().numpy()]
    pcd_sampled = o3d.geometry.PointCloud()
    pcd_sampled.points = o3d.utility.Vector3dVector(sample_cloud)
    ground_cloud,non_ground_cloud = ground_plane_segmentation(pcd_sampled)
    ground_plane_segmentation(pcd)
    print('点云中点的数量：',len(pcd.points))
    print('地面上的点：',len(non_ground_cloud.points))
    print('地面的点：',len(ground_cloud.points))
