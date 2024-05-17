import open3d as o3d
import numpy as np

def bin2pcd(bin_path,save_path):
    # 读取KITTI bin格式点云数据
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, 0:3]  # 提取x,y,z坐标值

    # 将点云数据转换为Open3D格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(save_path, pcd)
    return pcd

if __name__=='__main__':
    bin_path='D:\lidar\point\\000008\\000008.bin'
    save_path = 'D:\lidar\point\\000008\\000008.pcd'
    bin2pcd(bin_path,save_path)