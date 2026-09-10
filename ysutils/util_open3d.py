import open3d as o3d
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def rotation_matrix_to_euler_scipy(rot_mat, degrees=True):
    """
    3D高斯专用：旋转矩阵 → 欧拉角
    顺序：XYZ（3DGS 标准顺序：roll, pitch, yaw）

    参数：
        rot_mat: (3,3) 旋转矩阵（numpy 数组）
        degrees: True=角度，False=弧度
    返回：
        roll, pitch, yaw
    """
    # SciPy 直接构建旋转
    r = R.from_matrix(np.array(rot_mat))

    # 3DGS 标准欧拉角顺序：XYZ
    euler_angles = r.as_euler('xyz', degrees=degrees)

    return euler_angles  # [roll, pitch, yaw]


# ====================== 4. 核心：基于已知对应 + 带尺度 求解变换矩阵 ======================
def solve_similarity_from_correspondences(src_pts, tgt_pts):
    """
    输入：源点、目标点（一一对应）
    输出：4x4相似变换矩阵 T，满足 target = s*R@source + t
    【所有点都参与计算，不丢弃任何点】
    """
    # 中心化
    src_mean = np.mean(src_pts, axis=0)
    tgt_mean = np.mean(tgt_pts, axis=0)

    src_centered = src_pts - src_mean
    tgt_centered = tgt_pts - tgt_mean

    # 计算尺度 s
    s = np.sqrt(np.sum(tgt_centered ** 2) / np.sum(src_centered ** 2))
    src_scaled = s * src_centered

    # SVD求旋转 R
    H = src_scaled.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 计算平移 t
    t = tgt_mean - s * R @ src_mean

    # 构造4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t

    return T


def decompose_transform_matrix(mat4x4):
    """
    从 4×4 变换矩阵中分解出：
    scale (缩放) + rotation (旋转矩阵/欧拉角) + translation (平移)
    """
    mat = np.array(mat4x4, dtype=np.float32)

    # 取出 3x3 线性部分（旋转+缩放）
    mat3x3 = mat[:3, :3]

    # ======================
    # 1. 分离 缩放 Scale
    # ======================
    scale_x = np.linalg.norm(mat3x3[:, 0])  # 第一列长度
    scale_y = np.linalg.norm(mat3x3[:, 1])  # 第二列长度
    scale_z = np.linalg.norm(mat3x3[:, 2])  # 第三列长度
    scale = np.array([scale_x, scale_y, scale_z])

    # ======================
    # 2. 分离 旋转 Rotation
    # ======================
    # 每一列除以缩放 → 得到纯旋转矩阵
    rot_mat = mat3x3 / scale.reshape(1, 3)
    # 转欧拉角（3DGS标准：XYZ → roll, pitch, yaw）
    rotation = R.from_matrix(rot_mat).as_euler('xyz', degrees=True)

    # ======================
    # 3. 分离 平移 Translation
    # ======================
    translation = mat[:3, 3]

    return scale, rotation, translation, rot_mat

def icp_pcd(source_point, target_point):


    max_iterations = 50  # ICP迭代次数
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.asarray(source_point))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.asarray(target_point))

    T_final = np.eye(4)

    # 迭代优化
    for i in range(max_iterations):
        T = solve_similarity_from_correspondences(source_point, target_point)
        # 应用变换
        source.transform(T)
        T_final = T @ T_final
        # 更新点坐标
        source_point = np.asarray(source.points)


    # o3d.visualization.draw_geometries([source,target])

    source.paint_uniform_color([1, 0, 0])    # 红色
    target.paint_uniform_color([0, 1, 0])    # 绿色
    o3d.visualization.draw_geometries([source,target])

    # ---------------------- 2. 粗配准（必须先做） ----------------------
    # 这里先用单位矩阵（如果两片点云本来就差不远）
    trans_init = np.eye(4)

    # ---------------------- 3. ICP 精配准（核心） ----------------------
    # 点对距离阈值：超过这个值的点忽略
    threshold = 1.0

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # ---------------------- 4. 输出结果 ----------------------
    print("配准矩阵：")
    print(reg_p2p.transformation)
    print("适配分数：", reg_p2p.fitness)          # 内点比例（越高越好）
    print("均方误差：", reg_p2p.inlier_rmse)      # 误差（越低越好）

    # ---------------------- 5. 可视化 ----------------------
    source.transform(reg_p2p.transformation)
    o3d.visualization.draw_geometries([source, target])


def create_bbox_lineset(bbox):
    """
    将bounding box转换为线框网格，便于可视化

    Args:
        bbox (open3d.geometry.AxisAlignedBoundingBox or open3d.geometry.OrientedBoundingBox): bounding box

    Returns:
        open3d.geometry.LineSet: bounding box的线框表示
    """
    # 获取bounding box的角点
    corners = bbox.get_box_points()

    # 定义bounding box的边索引
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # 创建线框
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([bbox.color for _ in range(len(lines))])

    return line_set

def get_cam_frust(scale=1):
    p0 = np.asarray([0, 0, 0]) * scale
    p1 = np.asarray([-2,1.5,2.5])* scale
    p2 = np.asarray([-2,-1.5,2.5])* scale
    p3 = np.asarray([2,1.5,2.5])* scale
    p4 = np.asarray([2,-1.5, 2.5])* scale

    frust_point = np.stack([p0,p1,p2,p3,p4])
    frust = o3d.geometry.LineSet()
    frust_line_idx = [[0,1],[0,2],[0,3],[0,4],
                     [1,2],[1,3],[4,2],[4,3]]

    frust_line_idx = np.asarray(frust_line_idx)
    frust.points = o3d.utility.Vector3dVector(frust_point)
    frust.lines = o3d.utility.Vector2iVector(frust_line_idx)
    frust.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([[255,0,0]]),(frust_line_idx.shape[0],1)))
    return frust

def gen_cam_frust(poses,scale=0.05,color=[1, 0, 0]):
    poses = [np.asarray(p) for p in poses]
    cams = []
    for i in range(len(poses)):
        cam_mod = get_cam_frust(scale)
        cam_mod.paint_uniform_color(color)
        cams.append(cam_mod.transform(poses[i]))
    return cams


def create_bbox(size, color=[0, 1, 0], transform=None, xyz_quat=None, lineset=False):
    """
    根据中心点坐标和三边长创建Open3D的bounding box

    Args:
        size (list or numpy array): 三边长 [length, width, height]
        color (list, optional): bounding box的颜色，默认为绿色 [0, 1, 0]
        is_axis_aligned (bool, optional): 是否为轴对齐的bounding box，默认为True
        transform (numpy array, optional): 4x4变换矩阵，仅用于非轴对齐的bounding box
        xyz_quat (list or numpy array, optional): 位移及四元数表示的旋转 (x, y, z, qw, qx, qy, qz)，仅用于非轴对齐的bounding box
    Returns:
        open3d.geometry.AxisAlignedBoundingBox or open3d.geometry.OrientedBoundingBox: 创建的bounding box
    """
    # 创建可以定向的bounding box
    # 使用变换矩阵或默认的单位变换

    if xyz_quat is not None:
        # 从xyz_quat中提取位置和旋转信息
        x, y, z, qw, qx, qy, qz = xyz_quat

        # 使用scipy将四元数转换为旋转矩阵
        quat = [qx, qy, qz, qw]  # scipy使用[qx, qy, qz, qw]顺序
        rotation = R.from_quat(quat)
        rotation_matrix = rotation.as_matrix()

        # 创建OBB：xyz_quat中的(x,y,z)直接作为bbox的中心
        bbox = o3d.geometry.OrientedBoundingBox(
            center=[x, y, z],  # 使用xyz_quat中的位置作为中心点
            R=rotation_matrix,  # 使用四元数转换的旋转矩阵
            extent=size  # 三边长作为extent
        )
    elif transform is not None:
        # 从transform矩阵中提取位置和旋转信息
        # transform的平移部分是bbox的中心
        transform_center = transform[:3, 3]
        rotation_matrix = transform[:3, :3]

        # 创建OBB
        bbox = o3d.geometry.OrientedBoundingBox(
            center=transform_center,  # 使用transform的平移部分作为中心点
            R=rotation_matrix,  # 使用transform的旋转部分
            extent=size
        )
    else:
        raise ValueError("Either transform or xyz_quat must be provided.")
    # 设置颜色
    bbox.color = color
    if lineset:
        return create_bbox_lineset(bbox)
    else:
        return bbox

def create_bboxs(sizes, color=[0, 1, 0], transform=None, xyz_quats=None, lineset=False):
    creabboxs = []
    for i in range(len(sizes)):
        if transform is None:
            assert len(sizes) == len(xyz_quats), "sizes and xyz_quats must have the same length"
            te_bbox = create_bbox(sizes[i], color, None, xyz_quats[i], lineset)
        else:
            assert len(sizes) == len(transform), "sizes and xyz_quats must have the same length"
            te_bbox = create_bbox(sizes[i], color, transform[i], None, lineset)
        creabboxs.append(te_bbox)
    return creabboxs


class Gen_ray():
    def __init__(self,H,W,K):
        u, v = np.meshgrid(np.arange(0, W), np.arange(0, H))
        p = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)
        p = np.matmul(np.linalg.pinv(K), p.transpose())
        self.rays_v = p / np.linalg.norm(p, ord=2, axis=0, keepdims=True)
        self.K = K
    def gen_rays_at(self,pose):
        """
        Generate rays at world space from one camera.
        pose, not extrinsic!!!!!
        """
        rays_v = np.matmul(pose[:3, :3], self.rays_v).transpose()
        rays_o = np.tile(pose[:3, 3].reshape(1,3),(rays_v.shape[0],1))
        return np.concatenate((rays_o,rays_v),axis=1)

    def _get_principal_v(self,K,pose):
        u,v =K[0,2],K[1,2]
        p = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)
        p = np.matmul(np.linalg.pinv(K), p.transpose())
        rays_v = p / np.linalg.norm(p, ord=2, axis=0, keepdims=True)
        rays_v = np.matmul(pose[:3, :3], rays_v).transpose()
        rays_o = np.tile(pose[:3, 3].reshape(1,3),(rays_v.shape[0],1))
        return np.concatenate((rays_o,rays_v),axis=1)
    def get_principal_v(self,pose):
        K = self.K
        u,v =K[0,2],K[1,2]
        p = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)
        p = np.matmul(np.linalg.pinv(K), p.transpose())
        rays_v = p / np.linalg.norm(p, ord=2, axis=0, keepdims=True)
        rays_v = np.matmul(pose[:3, :3], rays_v).transpose()
        rays_o = np.tile(pose[:3, 3].reshape(1,3),(rays_v.shape[0],1))
        return np.concatenate((rays_o,rays_v),axis=1)


def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    if not direct:
        depth = 1.0 / (depth + 1e-6)
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
    if mask is not None:
        invalid_mask += np.logical_not(mask)
    if depth_min is None:
        depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color


def create_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    return pcd

class open3d_raycast_scene():
    def __init__(self):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.GR = None
        self.H = None
        self.W = None
        self.K = None
    def regist_mesh(self, mesh: o3d.geometry.TriangleMesh):
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh_id = self.scene.add_triangles(tmesh)
        return mesh_id
    def regist_camera(self,K,H,W):
        self.GR = Gen_ray(H,W,K[:3,:3])
        self.H=H
        self.W=W
        self.K = K
    def ray_cast(self,pose):
        if self.GR is None:
            raise ValueError('Raycasting camera has not been initialized.')
            return None
        rays = self.GR.gen_rays_at(pose)
        rays = rays.astype(np.float32)
        rays_v = rays[:, 3:6]
        original_ray = self.GR.get_principal_v(pose=pose)
        adj_cos = np.dot(rays_v, original_ray[0, 3:6])
        rays_o3d = o3d.core.Tensor(rays,
                                   dtype=o3d.core.Dtype.Float32)
        ans = self.scene.cast_rays(rays_o3d)
        norm = ans['primitive_normals'].numpy()
        norm = norm.reshape(self.H, self.W, 3)
        t_img = ans['t_hit'].numpy()
        depth = (t_img * adj_cos).reshape(self.H, self.W)
        mask = np.logical_not(np.logical_or(np.isinf(depth), np.isnan(depth))) * 255
        return {'depth': depth, 'mask': mask,'normal': norm}
