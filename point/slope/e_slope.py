import open3d as o3d
import numpy as np
import cv2
from function import *

if __name__ == '__main__':
    # 读取点云文件
    pcd1 = o3d.io.read_point_cloud('Plane.1.pcd')
    pcd2 = o3d.io.read_point_cloud('Plane.2.pcd')
    pcd3 = o3d.io.read_point_cloud('Plane.3.pcd')
    # pcd = o3d.io.read_point_cloud("Plane.pcd")
    box1 = o3d.io.read_point_cloud('Box.pcd')
    box2 = o3d.io.read_point_cloud('Box.sampled.pcd')
    box = box1 + box2
    pcd = pcd1 + pcd2 + pcd3
    # pcd
    pcd = preprocess_point_cloud(pcd, voxel_size=0.04)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])
    points = np.asarray(pcd.points)
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    z_range = points[:, 2].max() - points[:, 2].min()
    # o3d.visualization.draw_geometries([pcd], window_name="Original Point")  # plane

    # 随机生成每个点在每个轴上的偏移量，范围为1%到10%
    x_offsets = np.random.uniform(-0.03 * x_range, 0.03 * x_range, size=points.shape[0])
    y_offsets = np.random.uniform(-0.03 * y_range, 0.03 * y_range, size=points.shape[0])
    z_offsets = np.random.uniform(-0.03 * z_range, 0.03 * z_range, size=points.shape[0])
    # 将偏移量加到原始点云坐标上
    points[:, 0] += x_offsets
    points[:, 1] += y_offsets
    points[:, 2] += z_offsets
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd = add_noise_to_point_cloud(pcd, 500)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([pcd], window_name="Updated Point Cloud")

    distance_threshold, ransac_n, num_iterations = 0.02, 3, 5000
    inlier_cloud_1, outlier_cloud_1, normal_vector_1 = segment_planes(pcd, distance_threshold, ransac_n,
                                                                      num_iterations)  # plane
    inlier_cloud_2, outlier_cloud_2, normal_vector_2 = segment_planes(outlier_cloud_1, distance_threshold, ransac_n,
                                                                      num_iterations)
    inlier_cloud_3, remaining_cloud, normal_vector_3 = segment_planes(outlier_cloud_2, distance_threshold, ransac_n,
                                                                      num_iterations)

    pcds = [inlier_cloud_1, inlier_cloud_2, inlier_cloud_3]
    normal_vectors = [normal_vector_1, normal_vector_2, normal_vector_3]

    #
    classified_clusters = classify_point_clouds(pcds, normal_vectors)

    # 动态创建变量名的字典
    clusters = {}
    normal_vectors_dict = {}
    # Output classification results and assign clusters and normal vectors
    for i, (z_mean, cluster, normal_vector) in enumerate(classified_clusters, start=1):
        # Visualize point cloud (optional)
        # o3d.visualization.draw_geometries([cluster])
        clusters[f'cluster_{i}'] = cluster
        normal_vectors_dict[f'normal_vector_{i}'] = normal_vector

    inlier_cloud_1 = clusters['cluster_1']
    inlier_cloud_2 = clusters['cluster_2']
    inlier_cloud_3 = clusters['cluster_3']

    normal_vector_1 = normal_vectors_dict['normal_vector_1']
    normal_vector_2 = normal_vectors_dict['normal_vector_2']
    normal_vector_3 = normal_vectors_dict['normal_vector_3']
    inlier_cloud_1.paint_uniform_color([1.0, 0, 0])  # 将第一个平面内的点涂成红色
    inlier_cloud_2.paint_uniform_color([0, 1.0, 0])  # 将第二个平面内的点涂成绿色
    inlier_cloud_3.paint_uniform_color([1, 1, 0])  # 将第三个平面内的点涂成黄色
    remaining_cloud.paint_uniform_color([0, 0, 1.0])  # 将剩余点涂成蓝色

    o3d.visualization.draw_geometries([inlier_cloud_1, inlier_cloud_2, inlier_cloud_3,remaining_cloud],
                                      window_name="Updated Point Cloud")

    # 滤波操作
    pcd_filter = pcd
    # pcd = mean_filter(pcd)
    # pcd = median_filter(pcd)
    pcd_filter = remove_outliers_combined(pcd_filter)
    points_filter = np.asarray(pcd_filter.points)
    pcd_filter = bilateral_filter(pcd_filter, points_filter, sigma_s=0.05, sigma_r=0.05, radius=0.1)
    # pcd_filter.points = o3d.utility.Vector3dVector(points_filter)
    pcd_filter.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([pcd_filter], window_name="Filtered Point Cloud")

    filter_inlier_cloud_1, filter_outlier_cloud_1, filter_normal_vector_1 = segment_planes(pcd_filter,
                                                                                           distance_threshold, ransac_n,
                                                                                           num_iterations)
    filter_inlier_cloud_2, filter_outlier_cloud_2, filter_normal_vector_2 = segment_planes(filter_outlier_cloud_1,
                                                                                           distance_threshold, ransac_n,
                                                                                           num_iterations)
    filter_inlier_cloud_3, filter_remaining_cloud, filter_normal_vector_3 = segment_planes(filter_outlier_cloud_2,
                                                                                           distance_threshold, ransac_n,
                                                                                           num_iterations)
    filter_pcds = [filter_inlier_cloud_1, filter_inlier_cloud_2, filter_inlier_cloud_3]
    filter_normals = [filter_normal_vector_1, filter_normal_vector_2, filter_normal_vector_3]

    filter_classified_clusters = classify_point_clouds(filter_pcds, filter_normals)

    # 动态创建变量名的字典
    filter_clusters = {}
    filter_normal_vectors_dict = {}
    # Output classification results and assign clusters and normal vectors
    for i, (z_mean, cluster, normal_vector) in enumerate(filter_classified_clusters, start=1):
        filter_clusters[f'cluster_{i}'] = cluster
        filter_normal_vectors_dict[f'normal_vector_{i}'] = normal_vector

    filter_inlier_cloud_1 = filter_clusters['cluster_1']
    filter_inlier_cloud_2 = filter_clusters['cluster_2']
    filter_inlier_cloud_3 = filter_clusters['cluster_3']

    filter_normal_vector_1 = filter_normal_vectors_dict['normal_vector_1']
    filter_normal_vector_2 = filter_normal_vectors_dict['normal_vector_2']
    filter_normal_vector_3 = filter_normal_vectors_dict['normal_vector_3']

    # # 显示分割结果
    filter_inlier_cloud_1.paint_uniform_color([1.0, 0, 0])  # 将第一个平面内的点涂成红色
    filter_inlier_cloud_2.paint_uniform_color([0, 1.0, 0])  # 将第二个平面内的点涂成绿色
    filter_inlier_cloud_3.paint_uniform_color([1, 1, 0])  # 将第三个平面内的点涂成黄色
    filter_remaining_cloud.paint_uniform_color([0, 0, 1.0])  # 将剩余点涂成蓝色
    o3d.visualization.draw_geometries(
        [filter_inlier_cloud_1, filter_inlier_cloud_2, filter_inlier_cloud_3, filter_remaining_cloud],
        window_name="Point Segmentation")  # plane
    print(
        len(filter_inlier_cloud_1.points) + len(filter_inlier_cloud_2.points) + len(filter_inlier_cloud_3.points) + len(
            filter_remaining_cloud.points))
    # # 输出两个平面的法向量
    print(f"First plane normal vector: {normal_vector_1}")
    print(f"Second plane normal vector: {normal_vector_2}")
    print(f"Third plane normal vector: {normal_vector_3}")
    #
    angle_pred12 = get_slope(normal_vector_1, normal_vector_2)
    angle_pred23 = get_slope(normal_vector_2, normal_vector_3)
    print(f"Angle_pred between the 1-2 planes: {angle_pred12} degrees")
    print(f"Angle_pred between the 2-3 planes: {angle_pred23} degrees")

    filter_angle_pred12 = get_slope(filter_normal_vector_1, filter_normal_vector_2)
    filter_angle_pred23 = get_slope(filter_normal_vector_2, filter_normal_vector_3)
    print(f"Angle_pred between filtered the 1-2 planes: {filter_angle_pred12} degrees")
    print(f"Angle_pred between filtered the 2-3 planes: {filter_angle_pred23} degrees")

    normal_1 = np.array([-0.0152804, 0.182808, 0.98303])  # plane
    normal_2 = np.array([-0.020418, -0.583249, 0.812037])
    normal_3 = np.array([0, 0, 1])  # plane

    angle_truth12 = get_slope(normal_1, normal_2)
    angle_truth23 = get_slope(normal_2, normal_3)
    print(f"Angle_truth between the 1-2 planes: {angle_truth12} degrees")
    print(f"Angle_truth between the 2-3 planes: {angle_truth23} degrees")
    angle_error12 = abs(angle_truth12 - angle_pred12) / angle_truth12 * 100
    angle_error23 = abs(angle_truth23 - angle_pred23) / angle_truth23 * 100
    filter_angle_error12 = abs(filter_angle_pred12 - angle_truth12) / angle_truth12 * 100
    filter_angle_error23 = abs(filter_angle_pred23 - angle_truth23) / angle_truth23 * 100
    print(f"Error between 1-2 planes: {angle_error12} %")
    print(f"Error between 2-3 planes: {angle_error23} %")
    print(f"Error between filtered  1-2 planes: {filter_angle_error12} %")
    print(f"Error between filtered  2-3 planes: {filter_angle_error23} %")

    # box
    # box = preprocess_point_cloud(box, voxel_size=0.03)
    # box.paint_uniform_color([0.0, 1.0, 1.0])
    # # o3d.visualization.draw_geometries([box], window_name="Original Point")  # box
    # points = np.asarray(box.points)
    # x_range = points[:, 0].max() - points[:, 0].min()
    # y_range = points[:, 1].max() - points[:, 1].min()
    # z_range = points[:, 2].max() - points[:, 2].min()
    # print(f'点的x轴范围：{points[:, 0].min()}~{points[:, 0].max()}')
    # print(f'点的y轴范围：{points[:, 1].min()}~{points[:, 1].max()}')
    # print(f'点的z轴范围：{points[:, 2].min()}~{points[:, 2].max()}')
    # # 随机生成每个点在每个轴上的偏移量，范围为1%到10%
    # x_offsets = np.random.uniform(-0.05 * x_range, 0.05 * x_range, size=points.shape[0])
    # y_offsets = np.random.uniform(-0.05 * y_range, 0.05 * y_range, size=points.shape[0])
    # z_offsets = np.random.uniform(-0.05 * z_range, 0.05 * z_range, size=points.shape[0])
    # # 将偏移量加到原始点云坐标上
    # points[:, 0] += x_offsets
    # points[:, 1] += y_offsets
    # points[:, 2] += z_offsets
    # print(f'随机点后的x轴范围：{points[:, 0].min()}~{points[:, 0].max()}')
    # print(f'随机点后的y轴范围：{points[:, 1].min()}~{points[:, 1].max()}')
    # print(f'随机点后的z轴范围：{points[:, 2].min()}~{points[:, 2].max()}')
    # # o3d.visualization.draw_geometries([box], window_name="Filtered Point Cloud")
    # # 滤波处理
    # # box = mean_filter(box)
    # points = bilateral_filter(box)
    # box.points = o3d.utility.Vector3dVector(points)
    # print(f'滤波后的x轴范围：{points[:, 0].min()}~{points[:, 0].max()}')
    # print(f'滤波点后的y轴范围：{points[:, 1].min()}~{points[:, 1].max()}')
    # print(f'滤波点后的z轴范围：{points[:, 2].min()}~{points[:, 2].max()}')
    # # o3d.visualization.draw_geometries([box], window_name="Updated Point Cloud")
    #
    # inlier_cloud_1, outlier_cloud_1, normal_vector_1 = segment_planes(box)
    # inlier_cloud_2, outlier_cloud_2, normal_vector_2 = segment_planes(outlier_cloud_1)
    # inlier_cloud_1.paint_uniform_color([1.0, 0, 0])  # 将第一个平面内的点涂成红色
    # inlier_cloud_2.paint_uniform_color([0, 1.0, 0])  # 将第二个平面内的点涂成绿色
    # o3d.visualization.draw_geometries([inlier_cloud_1, inlier_cloud_2, outlier_cloud_2],
    #                                   window_name="Point Segmentation")  # box
    # normal_1 = np.array([-0.00324307, 0.964278, 0.264871])
    # normal_2 = np.array([0, 0, 1])  # box
    # print(f"First plane normal vector: {normal_vector_1}")
    # print(f"Second plane normal vector: {normal_vector_2}")
    # angle_pred12 = get_slope(normal_vector_1, normal_vector_2)
    # print(f"Angle_pred between the 1-2 planes: {angle_pred12} degrees")
    # angle_truth12 = get_slope(normal_1, normal_2)
    # print(f"Angle_truth between the 1-2 planes: {angle_truth12} degrees")
    # angle_error12 = abs(angle_truth12 - angle_pred12) / angle_truth12 * 100
    # print(f"Error between 1-2 planes: {angle_error12} %")
