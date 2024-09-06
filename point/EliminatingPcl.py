import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


class eliminatingPcl():
    def __init__(self):
        self.boom_angle = np.radians(10.0)
        self.arm_angle = np.radians(150.0)
        self.bucket_angle = np.radians(135.0)
        self.nb_neighbors = 10  # 去噪邻居数
        self.std_ratio = 2.0  # 去噪标准差比率

        # 设置挖掘机参数
        self.boom_length = 6.5  # 单位米
        self.arm_length = 3.3  # 单位米
        self.bucket_size = 1.5  # 单位米
        self.boom_width = 1.5  # 单位米
        self.arm_width = 1.5  # 单位米
        self.bucket_width = 1.5  # 单位米
        self.base_point = np.array([0, -0.5, 1.5])

    def denoise_pointcloud(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # 使用统计离群值去除
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)

        # 保留强度信息
        denoised_points = points[ind]
        return denoised_points
    def voxel_downsample(self, points, voxel_size):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        return np.asarray(downsampled_pcd.points)

    def filter_excavator_parts(self, points):
        boom_start = self.base_point
        boom_end = self.calculate_boom_end()
        arm_end = self.calculate_arm_end(boom_end)
        bucket_end = self.calculate_bucket_end(arm_end)

        # 使用KD-Tree进行空间查询
        tree = cKDTree(points)

        # 过滤大臂
        boom_points = self.query_cylinder(tree, boom_start, boom_end, self.boom_width / 2)

        # 过滤小臂
        arm_points = self.query_cylinder(tree, boom_end, arm_end, self.arm_width / 2)

        # 过滤铲斗（简化为长方体）
        bucket_points = self.query_box(tree, arm_end, bucket_end, self.bucket_width)

        # 合并所有需要过滤的点的索引
        filter_indices = set(boom_points + arm_points + bucket_points)

        # 创建一个布尔掩码数组
        mask = np.ones(len(points), dtype=bool)
        mask[list(filter_indices)] = False

        return points[mask]

    def query_cylinder(self, tree, start, end, radius):
        direction = end - start
        length = np.linalg.norm(direction)

        # 避免除零错误
        if length == 0:
            return []  # 如果长度为0，则直接返回空列表

        # 确保num_points为正整数
        num_points = max(int(length / radius) + 1, 1)

        # 在圆柱体轴上均匀取点
        query_points = start + np.outer(np.linspace(0, 1, num_points), direction)

        # 使用集合操作来合并查询结果
        indices = set()
        for point in query_points:
            indices.update(tree.query_ball_point(point, radius))

        return list(indices)

    def query_box(self, tree, start, end, width):
        # 确保所有计算使用浮点数
        center = (start + end) / 2.0
        half_size = np.abs(end - start) / 2.0 + width / 2.0

        # 计算最小和最大角
        # min_corner = center - half_size
        # max_corner = center + half_size

        # 确保half_size是一个有效的数组
        if not isinstance(half_size, np.ndarray):
            half_size = np.array([half_size])

        # 计算半径
        radius = np.linalg.norm(half_size)

        # 返回查询结果
        return tree.query_ball_point(center, radius)

    # def point_in_cylinder(self, point, cylinder_start, cylinder_end, radius):
    #     """
    #     判断点是否在圆柱体内。
    #
    #     参数:
    #     - point: 待判断的点坐标，numpy数组。
    #     - cylinder_start: 圆柱体起始点坐标，numpy数组。
    #     - cylinder_end: 圆柱体结束点坐标，numpy数组。
    #     - radius: 圆柱体半径。
    #
    #     返回:
    #     - 布尔值，点是否在圆柱体内。
    #     """
    #     vec = cylinder_end - cylinder_start
    #     length = np.linalg.norm(vec)
    #     unit_vec = vec / length
    #
    #     proj = np.dot(point - cylinder_start, unit_vec)
    #     if proj < 0 or proj > length:
    #         return False
    #
    #     closest_point = cylinder_start + proj * unit_vec
    #     distance = np.linalg.norm(point - closest_point)
    #
    #     return distance <= radius
    #
    # def point_in_box(self, point, box_start, box_end, width):
    #     """
    #     判断点是否在长方体内。
    #
    #     参数:
    #     - point: 待判断的点坐标，numpy数组。
    #     - box_start: 长方体起始点坐标，numpy数组。
    #     - box_end: 长方体结束点坐标，numpy数组。
    #     - width: 长方体宽度（长方体为平行于坐标轴的长方体）。
    #
    #     返回:
    #     - 布尔值，点是否在长方体内。
    #     """
    #     vec = box_end - box_start
    #     length = np.linalg.norm(vec)
    #     unit_vec = vec / length
    #
    #     # 计算垂直于主轴的两个单位向量
    #     perp1 = np.array([-unit_vec[1], unit_vec[0], 0])
    #     perp2 = np.cross(unit_vec, perp1)
    #
    #     # 将点变换到以box_start为原点，unit_vec、perp1、perp2为坐标轴的坐标系
    #     local_point = point - box_start
    #     x = np.dot(local_point, unit_vec)
    #     y = np.dot(local_point, perp1)
    #     z = np.dot(local_point, perp2)
    #
    #     return (0 <= x <= length and
    #             -width / 2 <= y <= width / 2 and
    #             -width / 2 <= z <= width / 2)

    def calculate_boom_end(self):
        """
        计算挖掘机大臂末端点坐标。

        返回:
        - 大臂末端点坐标，numpy数组。
        """
        x = self.boom_length * np.cos(self.boom_angle)
        y = 0
        z = self.boom_length * np.sin(self.boom_angle)
        return np.array([x, y, z])+self.base_point

    def calculate_arm_end(self, boom_end):
        """
        计算挖掘机小臂末端点坐标。

        参数:
        - boom_end: 大臂末端点坐标，numpy数组。

        返回:
        - 小臂末端点坐标，numpy数组。
        """
        x = boom_end[0] - self.arm_length * np.cos(self.boom_angle + self.arm_angle)
        y = boom_end[1]
        z = boom_end[2] - self.arm_length * np.sin(self.boom_angle + self.arm_angle)
        return np.array([x, y, z])

    def calculate_bucket_end(self, arm_end):
        """
        计算挖掘机铲斗末端点坐标。

        参数:
        - arm_end: 小臂末端点坐标，numpy数组。

        返回:
        - 铲斗末端点坐标，numpy数组。
        """
        x = arm_end[0] + self.bucket_size * np.cos(self.boom_angle + self.arm_angle + self.bucket_angle)
        y = arm_end[1]
        z = arm_end[2] + self.bucket_size * np.sin(self.boom_angle + self.arm_angle + self.bucket_angle)
        return np.array([x, y, z])

    def visualize_results(self, original_pcd, filtered_pcd):
        boom_line = o3d.geometry.LineSet()
        boom_line.points = o3d.utility.Vector3dVector([self.base_point, self.calculate_boom_end()])
        boom_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        boom_line.paint_uniform_color([1, 0, 0])  # Red for boom

        arm_end = self.calculate_arm_end(self.calculate_boom_end())
        arm_line = o3d.geometry.LineSet()
        arm_line.points = o3d.utility.Vector3dVector([self.calculate_boom_end(), arm_end])
        arm_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        arm_line.paint_uniform_color([0, 1, 0])  # Green for arm

        bucket_end = self.calculate_bucket_end(arm_end)
        bucket_line = o3d.geometry.LineSet()
        bucket_line.points = o3d.utility.Vector3dVector([arm_end, bucket_end])
        bucket_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        bucket_line.paint_uniform_color([0, 0, 1])  # Blue for bucket

        o3d.visualization.draw_geometries([original_pcd, boom_line, arm_line, bucket_line])
        o3d.visualization.draw_geometries([filtered_pcd, boom_line, arm_line, bucket_line])

e_pcl = eliminatingPcl()
# 指定点云数据的路径
pcd_path = 'D:\Desktop\data\\rosbag2_2024_07_31-09_38_40_pcds\\1722389978-834841031.pcd'
# 读取点云数据
pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)
points = e_pcl.voxel_downsample(points, voxel_size=0.05)

# 对点云进行过滤
filtered_pcd = o3d.geometry.PointCloud()
filtered_points = e_pcl.filter_excavator_parts(points=points)
# 对点云进行去噪处理
filtered_points = e_pcl.denoise_pointcloud(points=filtered_points)

filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

e_pcl.visualize_results(pcd, filtered_pcd)
