import open3d as o3d
import os

# 设置pcd文件所在文件夹路径
folder_path = "D:\lidar\point\point_seg"

# 获取文件夹中所有pcd文件路径
pcd_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pcd")]

# 遍历每个pcd文件并可视化
for pcd_file in pcd_files:
    # 读取pcd文件
    pcd = o3d.io.read_point_cloud(pcd_file)
    
    # 可视化
    o3d.visualization.draw_geometries([pcd])