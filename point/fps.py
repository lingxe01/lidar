import numpy as np
import open3d as o3d
import torch
def read_kitti_bin(bin_path):
    # 读取KITTI bin格式点云数据
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, 0:3]  # 提取x,y,z坐标值

    # 将点云数据转换为Open3D格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将NumPy数组转换为PyTorch张量
    xyz = torch.from_numpy(xyz).float().to(device)

    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids

def random_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        sampled_points: sampled pointcloud data, [npoint, 3]
    """
    N = xyz.shape[0]
    indices = np.random.choice(N, npoint, replace=False)
    sampled_points = xyz[indices]
    return sampled_points

if __name__ == '__main__':
    pcd = read_kitti_bin('/home/ling/mmdetection3d/data/kitti/training/velodyne/000008.bin')
    o3d.visualization.draw_geometries([pcd])
    points = np.asarray(pcd.points)
    points_num = len(points)
    points = points.reshape([1,points_num,3])

    sample_cloud = farthest_point_sample(points,60000)
    print(sample_cloud.shape)
    sample_cloud = points[0,sample_cloud[0].cpu().numpy()]

    pcd_sampled = o3d.geometry.PointCloud()
    pcd_sampled.points = o3d.utility.Vector3dVector(sample_cloud)
    o3d.visualization.draw_geometries([pcd_sampled])
    print(sample_cloud.shape)