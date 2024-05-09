import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import open3d as o3d

binary = f'D:\\lidar\\point\\000008\point_seg\\pcd_nongrond_seg.pcd'
pcd = o3d.io.read_point_cloud(binary)
original_colors = np.asarray(pcd.colors)
original_pts = np.asarray(pcd.points)
vis = o3d.visualization.Visualizer()

vis.create_window()


# 将点云添加到可视化窗口
vis.add_geometry(pcd)
vis.run()

print("Color value range:", np.min(original_colors), np.max(original_colors))

# 检查原始点云数据
print(f"Original point cloud shape: {original_pts.shape}")

# 筛选有效点云
valid_indices = original_pts[:, 0] > 0
valid_pts = original_pts[valid_indices]
valid_colors = original_colors[valid_indices]
valid_pts_with_colors = np.concatenate((valid_pts, valid_colors), axis=1)


print(f"Valid point cloud shape: {valid_pts.shape}")
print((f"Valid point colors shape: {valid_colors.shape}"))
print(f"valid_pts_with_colors shape:{valid_pts_with_colors.shape}")
# 创建筛选后的点云
pcd_selected = o3d.geometry.PointCloud()
pcd_selected.points = o3d.utility.Vector3dVector(valid_pts)
pcd_selected.colors = o3d.utility.Vector3dVector(valid_colors)
original_pts = np.asarray(pcd_selected.points)

# 可视化带颜色的点云
o3d.visualization.draw_geometries([pcd_selected])

# 读取相机参数
with open(f'D:\lidar\point\\000008\\000008.txt','r') as f:
    calib = f.readlines()

# P2 (3 x 4) for left eye
P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

# 读取图像尺寸
img_path = f'D:\lidar\point\\000008\\000008.png'
img = mpimg.imread(img_path)
IMG_H, IMG_W, _ = img.shape


# 投影点云到图像
velo = np.insert(original_pts, 3, 1, axis=1).T
velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
cam = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(velo)))
# cam = np.delete(cam, np.where(cam[2, :] < 0), axis=1)
cam[:2] /= cam[2, :]


# generate color map from depth
u,v,z = cam
print(cam.shape)
# 显示投影点云和原始图像
plt.figure(figsize=(12, 8))
plt.axis([0,IMG_W,IMG_H,0])
plt.imshow(img)
plt.scatter([u],[v],c=valid_colors,alpha=0.5,s=2)
plt.title(f'Projected Point Cloud on detect.png')
plt.savefig(f'D:\lidar\point\\000008/000008_projection_colors_filter.png',bbox_inches='tight')
plt.show()

