from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d

def preprocess_point_cloud(pcd, voxel_size):
    #对点云进行下采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    #计算法向量
    pcd_down.estimate_normals()
    
    return pcd_down


def point_seg(bin_path,save_path):
    pcd_combined = o3d.io.read_point_cloud("D:\lidar\point\\000008\copy_of_fragment.pcd")
    # 读取点云数据
    pcd_grond = o3d.io.read_point_cloud("D:\lidar\point\\000008\grond.pcd")

    pcd_nongrond= o3d.io.read_point_cloud(bin_path)
    print(len(pcd_nongrond.points))

    # 对非地面点云进行统计滤波去除噪声
    cl, ind = pcd_nongrond.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 下采样
    pcd_nongrond = preprocess_point_cloud(cl,0.2)
    print(len(pcd_nongrond.points))

    pcd_grond = preprocess_point_cloud(pcd_grond,0.2)

    # 给点云着色
    pcd_grond.paint_uniform_color([1, 0, 0])
    # 创建一个可视化窗口
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd_nongrond.cluster_dbscan(eps=0.7, min_points=20, print_progress=True))
        

    # 给不同簇上不同的伪彩色
    max_label = labels.max()
    print(f"点云共分为 {max_label + 1} 个簇")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd_nongrond.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 分割保存
    # clusters = []
    # for label in np.unique(labels):
    #     if label == -1:  # 噪声点跳过
    #         continue
    #     cluster_idx = np.where(labels == label)[0]
    #     cluster = pcd_nongrond.select_by_index(cluster_idx)
    #     clusters.append(cluster)

    # for i, cluster in enumerate(clusters):
    #     o3d.io.write_point_cloud(f"./point/point_seg/cluster_{i}.pcd", cluster)

    # print(f"共分割出 {len(clusters)} 个簇")

    # 可视化彩色点云
    # o3d.visualization.draw_geometries([pcd_nongrond])

    o3d.io.write_point_cloud(save_path, pcd_nongrond)

    vis = o3d.visualization.Visualizer()

    vis.create_window()


    # 将点云添加到可视化窗口
    vis.add_geometry(pcd_nongrond)
    # vis.add_geometry(pcd_grond)
    # 设置渲染参数
    render_options = vis.get_render_option()
    render_options.point_size = 2

    # 运行可视化窗口
    vis.run()
    vis.destroy_window()

if __name__=='__main__':
    point_seg(bin_path='D:\lidar\point\\000008\\nongrond.pcd',save_path='D:\lidar\point\\000008\point_seg\pcd_nongrond_seg.pcd')
