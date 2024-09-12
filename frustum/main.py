import matplotlib.image as mpimg
import numpy as np
import open3d as o3d
from function import *

def main(calib_path,img_path,binary,label_path):
    img = mpimg.imread(img_path)
    IMG_H, IMG_W, _ = img.shape

    P2, K, Tr_velo_to_cam, Tr, R0_rect = get_calib(calib_path)

    pcd = o3d.io.read_point_cloud(binary)
    print("原始点云点数:", len(pcd.points))
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    print("滤波后点云点数:", len(pcd.points))
    original_colors = np.asarray(pcd.colors)
    original_pts = np.asarray(pcd.points)

    # 筛选有效点云
    valid_indices = original_pts[:, 0] > 0
    valid_pts = original_pts[valid_indices]
    gray_color = np.array([0.7, 0.7, 0.7])  # 灰色
    valid_colors = np.tile(gray_color, (len(valid_pts), 1))

    # 创建筛选后的点云
    pcd_selected = o3d.geometry.PointCloud()
    pcd_selected.points = o3d.utility.Vector3dVector(valid_pts)
    pcd_selected.colors = o3d.utility.Vector3dVector(valid_colors)

    with open(label_path, 'r') as f:
        detections = [line.strip().split() for line in f.readlines()]

    detections = [[float(x) for x in det] for det in detections]

    # 相机内参
    # fx, fy = P2[0, 0], P2[1, 1]
    # cx, cy = P2[0, 2], P2[1, 2]

    car_length = 3

    scale_factor=1.1

    inside_points=[]
    i = 0
    for det in detections:
        category, u_center, v_center, width, height, conf,distance = det
        if distance > 30: continue
        i+=1
        Z_near = 1
        Z_far = distance + car_length
        left_top = ((u_center - width / 2) * IMG_W, (v_center - height / 2) * IMG_H)
        right_top = ((u_center + width / 2) * IMG_W, (v_center - height / 2) * IMG_H)
        left_bottom = ((u_center - width / 2) * IMG_W, (v_center + height / 2) * IMG_H)
        right_bottom = ((u_center + width / 2) * IMG_W, (v_center + height / 2) * IMG_H)
        bbox = [left_top, right_top, left_bottom, right_bottom]

        # 判断点云中的点是否在视锥体内
        # 使用这个函数
        bbox = [left_top[0], left_top[1], right_bottom[0], right_bottom[1]]
        z_range = [Z_near, Z_far]
        inside_frustum_indices = filter_points_in_frustum(valid_pts, P2, Tr_velo_to_cam, R0_rect, bbox, z_range, scale_factor)
        inside_points.extend(inside_frustum_indices)
    # 更新点云颜色
    valid_colors[inside_points] = [1, 0, 0]  # 将视锥体内的点设置为红色

    pcd_selected.colors = o3d.utility.Vector3dVector(valid_colors)

    print(f"Number of points inside frustum: {len(inside_points)}")
    # 选择在视锥体内的点
    inside_pcd = pcd_selected.select_by_index(inside_points)

    # clustered_pcd = db_cluster_points(pcd_selected.points,0.8,10)
    all_clustered_pcd,clustered_pcd,labels = db_cluster_points(pcd_selected.points,pcd_selected.colors,inside_points,0.8,10)
    aabb_list=add_bounding_boxes(clustered_pcd,labels,(4,3,2))
    geometry_list = [all_clustered_pcd]
    o3d.visualization.draw_geometries(geometry_list+aabb_list)


if __name__ == "__main__":
    calib_path = 'point\\000008\\000008.txt'
    img_path = 'point\\000008\save\\000008.png'
    binary = 'point\\000008\point_seg\pcd_nongrond_seg.pcd'
    label_path = 'frustum\\000008.txt'
    main(calib_path,img_path,binary,label_path)