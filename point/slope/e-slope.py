import open3d as o3d
import numpy as np


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


def segment_planes(pcd, distance_threshold=0.01, ransac_n=3,
                   num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    [a, b, c, d] = plane_model
    normals = np.asarray([a, b, c])
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud, normals


if __name__ == '__main__':
    # 读取点云文件
    pcd1 = o3d.io.read_point_cloud('Plane.1.pcd')
    pcd2 = o3d.io.read_point_cloud('Plane.2.pcd')
    pcd3 = o3d.io.read_point_cloud('Plane.3.pcd')
    # pcd = o3d.io.read_point_cloud("Plane.pcd")
    pcd = pcd1 + pcd2 + pcd3
    pcd = preprocess_point_cloud(pcd, voxel_size=0.03)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])
    points = np.asarray(pcd.points)
    o3d.visualization.draw_geometries([pcd], window_name="Original Point")

    # 定义平面分割参数
    distance_threshold = 0.01
    ransac_n = 3
    num_iterations = 1000

    inlier_cloud_1, outlier_cloud_1, normal_vector_1 = segment_planes(pcd)
    inlier_cloud_2, outlier_cloud_2, normal_vector_2 = segment_planes(outlier_cloud_1)
    inlier_cloud_3, remaining_cloud, normal_vector_3 = segment_planes(outlier_cloud_2)

    # 显示分割结果
    inlier_cloud_1.paint_uniform_color([1.0, 0, 0])  # 将第一个平面内的点涂成红色
    inlier_cloud_2.paint_uniform_color([0, 1.0, 0])  # 将第二个平面内的点涂成绿色
    inlier_cloud_3.paint_uniform_color([1, 1, 0])  # 将第三个平面内的点涂成黄色
    remaining_cloud.paint_uniform_color([0, 0, 1.0])  # 将剩余点涂成蓝色
    o3d.visualization.draw_geometries([inlier_cloud_1, inlier_cloud_2, inlier_cloud_3, remaining_cloud],
                                      window_name="Point Segmentation")
    # 输出两个平面的法向量
    print(f"First plane normal vector: {normal_vector_1}")
    print(f"Second plane normal vector: {normal_vector_2}")
    print(f"Third plane normal vector: {normal_vector_3}")

    angle_pred12 = get_slope(normal_vector_1, normal_vector_2)
    angle_pred23 = get_slope(normal_vector_2, normal_vector_3)
    print(f"Angle_pred between the 1-2 planes: {angle_pred12} degrees")
    print(f"Angle_pred between the 2-3 planes: {angle_pred23} degrees")

    normal_1 = np.array([-0.0152804, 0.182808, 0.98303])
    normal_2 = np.array([-0.020418, -0.583249, 0.812037])
    normal_3 = np.array([0, 0, 1])

    angle_truth12 = get_slope(normal_1, normal_2)
    angle_truth23 = get_slope(normal_2, normal_3)
    print(f"Angle_truth between the 1-2 planes: {angle_truth12} degrees")
    print(f"Angle_truth between the 2-3 planes: {angle_truth23} degrees")
    angle_error12 = abs(angle_truth12 - angle_pred12) / angle_truth12 * 100
    angle_error23 = abs(angle_truth23 - angle_pred23) / angle_truth23 * 100
    print(f"Error between 1-2 planes: {angle_error12} %")
    print(f"Error between 2-3 planes: {angle_error23} %")
