import numpy as np

def pixel_to_normalized(x, y, cx, cy, fx, fy):
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy
    return x_norm, y_norm


def normalized_to_camera(x_norm, y_norm, Z):
    X = x_norm * Z
    Y = y_norm * Z
    return X, Y, Z


def camera_to_lidar(X, Y, Z, Tr_velo_to_cam, R0_rect):
    camera_coords = np.array([X, Y, Z, 1.0]).reshape(4, 1)
    R0_rect_4x4 = np.eye(4)
    R0_rect_4x4[:3, :3] = R0_rect

    # 首先应用R0_rect
    rectified_coords = R0_rect_4x4 @ camera_coords
    # 然后应用Tr_velo_to_cam的逆
    transform_matrix_inv = np.linalg.inv(Tr_velo_to_cam)
    lidar_coords = transform_matrix_inv @ rectified_coords
    return lidar_coords.flatten()[:3]


# 计算平面方程：Ax + By + Cz + D = 0
def compute_plane_equation(p1, p2, p3):
    normal = np.cross(p2 - p1, p3 - p1)
    normal = normal / np.linalg.norm(normal)
    D = -np.dot(normal, p1)
    return np.append(normal, D)


def get_calib(calib_path):
    with open(calib_path, 'r') as f:
        calib = f.readlines()

    P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
    K = np.array(P2[:12]).reshape(3, 4)[:3, :3]

    # 从字符串解析出从激光雷达到相机的变换矩阵
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)
    Tr = np.array([Tr_velo_to_cam[0, 3], Tr_velo_to_cam[1, 3], Tr_velo_to_cam[2, 3]])

    R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
    return P2, K, Tr_velo_to_cam, Tr, R0_rect

def lidar_to_camera(point, Tr_velo_to_cam, R0_rect):
    point_hom = np.append(point, 1).reshape(4, 1)
    R0_rect_4x4 = np.eye(4)
    R0_rect_4x4[:3, :3] = R0_rect
    R0_rect = R0_rect_4x4
    point_camera = R0_rect @ Tr_velo_to_cam @ point_hom
    return point_camera[:3].flatten()

def project_to_image(point_camera, P):
    point_2d = P @ np.append(point_camera, 1)
    point_2d = point_2d[:2] / point_2d[2]
    return point_2d

def is_point_in_box(point_2d, bbox):
    x, y = point_2d
    left, top, right, bottom = bbox
    return left <= x <= right and top <= y <= bottom

def filter_points_in_frustum(points, P, Tr_velo_to_cam, R0_rect, bbox, z_range, scale_factor=1.05):
    inside_frustum = []
    
    # 扩大2D边界框
    left, top, right, bottom = bbox
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    width, height = right - left, bottom - top
    
    scaled_width, scaled_height = width * scale_factor, height * scale_factor
    scaled_left = center_x - scaled_width / 2
    scaled_right = center_x + scaled_width / 2
    scaled_top = center_y - scaled_height / 2
    scaled_bottom = center_y + scaled_height / 2
    
    scaled_bbox = [scaled_left, scaled_top, scaled_right, scaled_bottom]
    
    # 扩大深度范围
    z_near, z_far = z_range
    z_center = (z_near + z_far) / 2
    z_range = (z_center - (z_center - z_near) * scale_factor,
               z_center + (z_far - z_center) * scale_factor)

    for i, point in enumerate(points):
        # 转换到相机坐标系
        point_camera = lidar_to_camera(point, Tr_velo_to_cam, R0_rect)
        
        # 检查深度
        if not (z_range[0] <= point_camera[2] <= z_range[1]):
            continue
        
        # 投影到图像平面
        point_2d = project_to_image(point_camera, P)
        
        # 检查是否在扩大后的2D边界框内
        if is_point_in_box(point_2d, scaled_bbox):
            inside_frustum.append(i)
    
    return inside_frustum

if __name__=="__main__":
    print(0)