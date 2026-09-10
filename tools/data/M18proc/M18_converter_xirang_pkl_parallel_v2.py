import numpy as np
import os
import open3d as o3d
import pickle
import argparse
import shutil
import glob
import cv2
import json
from multiprocessing import Pool, cpu_count

from M18proc import (
    POSE_WAYMO2CV, POSE_WAYMO2GL,
    image_heights, image_widths,
    cameras_type, lidar_type, downsample_factors
)

VELOCITY_THRESHOLD = 0.01
PURE_2DOBJ = False
LIDAR_COLOR_FROM_CAM = [0, 1, 2, 3, 4, 9, 10]
MAX_FRAME_LIDAR_POINTS = 400000  # Set to positive value to enable random downsampling; -1 to disable
USE_LIDAR_TYPE = [0]
DEBUG = False

# PROC_CAMERA_IDX: only cameras whose index is in this list will have lidar_depth,
# intensity, image, and dynamic_mask output saved. Projection still runs for all
# cameras needed by LIDAR_COLOR_FROM_CAM, but file output is gated by this list.
# Set to None to process all cameras (backward-compatible default).
# Camera index mapping:
#   0:left_front_camera  1:right_front_camera  2:rear_camera
#   3:left_rear_camera   4:right_rear_camera   5:front_camera_fov200
#   6:rear_camera_fov200 7:left_camera_fov200  8:right_camera_fov200
#   9:center_camera_fov30  10:center_camera_fov120
PROC_CAMERA_IDX = [0, 1, 2, 3, 4, 9, 10] # None  # e.g. [0, 1, 2, 9, 10] for front-facing cameras only


class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，自动处理所有NumPy类型"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def dict_to_json_file(data, file_path, indent=4):
    """保存含NumPy类型的字典为JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=NumpyJSONEncoder, indent=indent, ensure_ascii=False)
        print(f"成功保存到 {file_path}")
    except Exception as e:
        print(f"保存失败：{e}")


def tensor_pcd_to_geometry_pcd(tensor_pcd):
    """
    将 o3d.t.geometry.PointCloud 转换为 o3d.geometry.PointCloud
    """
    geo_pcd = o3d.geometry.PointCloud()

    # 转换XYZ坐标
    points_tensor = tensor_pcd.point.positions
    points_np = points_tensor.cpu().numpy()
    geo_pcd.points = o3d.utility.Vector3dVector(points_np)

    # 转换自定义字段（如intensity、timestamp）
    pcd_attribute = {}
    for attr_name in dir(tensor_pcd.point):
        if attr_name == "positions":
            continue
        attr_tensor = tensor_pcd.point[attr_name]
        attr_np = attr_tensor.cpu().numpy()
        pcd_attribute[attr_name] = attr_np[:, 0]

    return geo_pcd, pcd_attribute


def resolve_path(path, data_root=None):
    """
    解析PKL中的路径（可能是绝对路径或相对路径）。
    1. 如果路径已存在，直接返回
    2. 如果提供了data_root且路径为相对路径，尝试拼接
    3. 否则返回原路径（让后续逻辑处理失败）
    """
    if not path:
        return path
    if os.path.exists(path):
        return path
    if data_root:
        # 尝试直接拼接
        full = os.path.join(data_root, path)
        if os.path.exists(full):
            return full
        # 尝试找到路径中的 'clip_' 或日期前缀，取其后部分拼接
        parts = path.split('/')
        for i, p in enumerate(parts):
            if p.startswith('clip_') or (p.startswith('20') and len(p) >= 8 and p[:8].isdigit()):
                relative = '/'.join(parts[i:])
                full = os.path.join(data_root, relative)
                if os.path.exists(full):
                    return full
    return path


def load_pkl(pkl_file):
    """加载PKL文件"""
    try:
        import numpy.core as _numpy_core
        import numpy.core.multiarray as _numpy_core_multiarray
        import sys as _sys
        _sys.modules.setdefault("numpy._core", _numpy_core)
        _sys.modules.setdefault("numpy._core.multiarray", _numpy_core_multiarray)
    except Exception:
        pass
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


def modify_mask_m18(mask, camtype_id):
    mask_list = []
    if camtype_id == 3: # h,w
        leftup = (0,0)
        rightdown = (250/1280,550/1920)
        mask_list.append((leftup,rightdown))
        leftup = (0,0)
        rightdown = (1,80/1920)
        mask_list.append((leftup,rightdown))

    if camtype_id == 4:
        leftup = (0,(1920-550)/1920)
        rightdown = (250/1280,1)
        mask_list.append((leftup,rightdown))
        leftup = (0,(1920-80)/1920)
        rightdown = (1,1)
        mask_list.append((leftup,rightdown))

    if camtype_id == 10:
        leftup = (1500/2160,0)
        rightdown = (1,1)
        mask_list = [(leftup,rightdown)]

    _,mask_h,mask_w = mask.shape
    for (leftup,rightdown) in mask_list:
        mask[:,int(leftup[0]*mask_h):int(rightdown[0]*mask_h),int(leftup[1]*mask_w):int(rightdown[1]*mask_w)] = False


# ============================================================
# Worker function for parallel frame processing
# ============================================================
def process_frame_chunk(chunk_infos, output_dir, data_root, proc_types,
                        img_heights, img_widths, ds_factors, downsample_enabled, worker_id):
    """
    Process a chunk of frames. This is the worker function called by multiprocessing Pool.

    Args:
        chunk_infos: list of (idx, info) tuples for frames to process
        output_dir: output root directory
        data_root: root for path resolution
        proc_types: which phases to run
        img_heights: image heights (possibly downsampled)
        img_widths: image widths (possibly downsampled)
        ds_factors: downsample factors
        downsample_enabled: bool
        worker_id: int identifier

    Returns:
        dict with aggregated results for this chunk:
            - timestamps: list of per-frame timestamp entries
            - timestamps_streetgs: dict with FRAME and camera timestamp entries
            - track_info_lines: list of track info rows
            - track_camera_vis: dict of track→frame→camera indices
            - worker_id: int identifier
            - errors: list of error messages
    """

    timestamps = []
    timestamps_streetgs = {'FRAME': {}}
    pt3d_all = {}
    pt2d_all = {}
    track_info_lines = []
    track_camera_vis = {}
    errors = []

    # Output subdirectories (pre-created by main process)
    ego_pose_dir = os.path.join(output_dir, 'ego_pose')
    extrinsics_dir = os.path.join(output_dir, 'extrinsics')
    intrinsics_dir = os.path.join(output_dir, 'intrinsics')
    images_dir = os.path.join(output_dir, 'images')
    lidar_depth_dir = os.path.join(output_dir, 'lidar_depth')
    intensity_dir = os.path.join(output_dir, 'intensity')
    dynamic_mask_dir2d = os.path.join(output_dir, 'dynamic_mask')
    track_dir = os.path.join(output_dir, 'track')

    for idx, info in chunk_infos:
        frame_name = f"{idx:06d}"

        # ============================================================
        # Phase: timestamp
        # ============================================================
        if 'timestamp' in proc_types:
            try:
                timestamp = info.get('timestamp', "")

                timestamp_entry = {
                    "FRAME_IDX": idx,
                    "FRAME_NAME": frame_name,
                    "timestamp": timestamp,
                    "cameras": {}
                }

                if len(str(timestamp)) == 19:
                    timestamps_streetgs['FRAME'][frame_name] = timestamp / 1000000000
                else:
                    if len(str(timestamp)) == 16:
                        timestamps_streetgs['FRAME'][frame_name] = timestamp / 1000000
                    else:
                        raise NotImplementedError(f"Unexpected timestamp length: {len(str(timestamp))}")

                if 'sensors' in info and 'cams' in info['sensors']:
                    cams = info['sensors']['cams']
                    for cam_name, cam_data in cams.items():
                        if cam_name in cameras_type:
                            cam_idx = cameras_type[cam_name]
                            cam_timestamp = cam_data.get('timestamp', '')

                            timestamp_entry["cameras"][cam_name] = {
                                "cam_idx": cam_idx,
                                "timestamp": cam_timestamp
                            }

                            if cam_name not in timestamps_streetgs:
                                timestamps_streetgs[cam_name] = {}

                            if len(str(cam_timestamp)) == 16:
                                timestamps_streetgs[cam_name][frame_name] = cam_timestamp / 1000000
                            elif len(str(cam_timestamp)) == 19:
                                timestamps_streetgs[cam_name][frame_name] = cam_timestamp / 1000000000
                            else:
                                raise NotImplementedError(
                                    f"Unexpected cam_timestamp length: {len(str(cam_timestamp))}")
                        else:
                            raise NotImplementedError(f"相机类型 {cam_name} 未在 cameras_type 中定义")

                timestamps.append(timestamp_entry)
            except Exception as e:
                errors.append(f"Frame {frame_name} timestamp error: {e}")

        # ============================================================
        # Phase: lidar
        # ============================================================
        if 'lidar' in proc_types:
            try:
                # 提取并保存 ego2global_transformation_matrix
                if 'ego2global_transformation_matrix' in info:
                    ego2global_transformation_matrix = info['ego2global_transformation_matrix']
                    ego2global_transformation_matrix_np = np.array(ego2global_transformation_matrix)

                    ego2global_path = os.path.join(ego_pose_dir, f"{frame_name}.txt")
                    np.savetxt(ego2global_path, ego2global_transformation_matrix_np)

                    # 处理 liDAR 数据
                    frame_lidar_points = o3d.geometry.PointCloud()
                    frame_lidar_intensity = np.array([])

                    if 'sensors' in info and 'lidar' in info['sensors']:
                        lidars = info['sensors']['lidar']
                        for lidar_name, lidar_data in lidars.items():
                            if lidar_name in lidar_type:
                                lidar_idx = lidar_type[lidar_name]
                                lidar_path = lidar_data.get('aws_path', '')

                                # 路径解析
                                lidar_path = resolve_path(lidar_path, data_root)

                                if not lidar_path or not os.path.exists(lidar_path):
                                    # 尝试用 data_path 作为备选
                                    data_path = lidar_data.get('data_path', '')
                                    if data_path and data_path != 'N/A':
                                        data_path = resolve_path(data_path, data_root)
                                        if os.path.exists(data_path):
                                            lidar_path = data_path
                                            lidar_data['aws_path'] = data_path

                                if not lidar_path or not os.path.exists(lidar_path):
                                    print(f"警告: lidar {lidar_name} 路径不存在: {lidar_data.get('aws_path', '')}")
                                    continue

                                if 0 not in USE_LIDAR_TYPE:
                                    if lidar_type[lidar_name] in USE_LIDAR_TYPE:
                                        lidar_points, lidar_attr = tensor_pcd_to_geometry_pcd(
                                            o3d.t.io.read_point_cloud(lidar_path))
                                        if 'extrinsic' in lidar_data:
                                            # lidar的extrinsic为lidar2ego
                                            lidar_points_vehicle = np.array(lidar_points.points)
                                            lidar_points_vehicle = np.concatenate(
                                                [lidar_points_vehicle,
                                                 np.ones_like(lidar_points_vehicle[..., :1])], axis=-1
                                            )
                                            lidar_points_vehicle = lidar_data['extrinsic'] @ lidar_points_vehicle.T
                                            lidar_points.points = o3d.utility.Vector3dVector(
                                                lidar_points_vehicle[:3, :].T)
                                        frame_lidar_points = frame_lidar_points + lidar_points
                                        frame_lidar_intensity = np.concatenate(
                                            [frame_lidar_intensity, lidar_attr['intensity']])
                            else:
                                raise NotImplementedError(f"lidar类型 {lidar_name} 未在 lidar_type 中定义")

                        if 0 not in USE_LIDAR_TYPE:
                            lidar_perception = frame_lidar_points
                            lidar_perception_intensity = frame_lidar_intensity
                        else:
                            perception_path = resolve_path(lidars['perception']['aws_path'], data_root)
                            lidar_perception, lidar_perception_attr = tensor_pcd_to_geometry_pcd(
                                o3d.t.io.read_point_cloud(perception_path))
                            frame_lidar_intensity = lidar_perception_attr['intensity']

                    # # Random downsample if points exceed MAX_FRAME_LIDAR_POINTS
                    # if MAX_FRAME_LIDAR_POINTS > 0:
                    #     num_points = len(lidar_perception.points)
                    #     if num_points > MAX_FRAME_LIDAR_POINTS:
                    #         rng = np.random.default_rng()
                    #         indices = rng.choice(num_points, MAX_FRAME_LIDAR_POINTS, replace=False)
                    #         lidar_perception = lidar_perception.select_by_index(indices)
                    #         frame_lidar_intensity = frame_lidar_intensity[indices]

                    pt3d_all[idx] = np.array(lidar_perception.points)
                    pt2d_all[idx] = np.zeros([pt3d_all[idx].shape[0], 6], dtype=np.int16) - 1

                    # 处理相机投影
                    if 'sensors' in info and 'cams' in info['sensors']:
                        cams = info['sensors']['cams']

                        for cam_name, cam_data in cams.items():
                            print(f"Worker {worker_id}: 处理帧 {frame_name}, 处理相机 {cam_name}",
                                  end='\r', flush=True)
                            if cam_name in cameras_type:
                                cam_idx = cameras_type[cam_name]
                                cam_h, cam_w = img_heights[cam_idx], img_widths[cam_idx]

                                file_name = f"{frame_name}_{cam_idx:02d}.txt"

                                # 提取 intrinsic
                                if 'cam_intrinsic' in cam_data:
                                    intrinsic = cam_data['cam_intrinsic']
                                    if downsample_enabled:
                                        dscale = ds_factors[cam_idx]
                                        intrinsic = np.array(intrinsic)
                                        intrinsic[:2, :] = intrinsic[:2, :] / dscale
                                    intrinsic_np = np.array(intrinsic)
                                    intrinsic_path = os.path.join(intrinsics_dir, file_name)
                                    np.savetxt(intrinsic_path, intrinsic_np)
                                else:
                                    raise NotImplementedError(f"cam_intrinsic未定义")

                                # 提取 extrinsic
                                if 'extrinsic' in cam_data:
                                    extrinsic = cam_data['extrinsic']
                                    extrinsic_np = np.array(extrinsic)
                                    extrinsic_path = os.path.join(extrinsics_dir, file_name)
                                    np.savetxt(extrinsic_path, extrinsic_np)
                                else:
                                    raise NotImplementedError(f"extrinsic未定义")

                                # LiDAR → 相机投影
                                points_ego = np.array(lidar_perception.points)
                                points_ego_homogeneous = np.concatenate(
                                    [points_ego,
                                     np.ones_like(points_ego[..., :1])], axis=-1
                                )
                                # ego坐标系 → 相机坐标系（使用相机的extrinsic）
                                points_cam = extrinsic_np @ points_ego_homogeneous.T
                                # 相机坐标系 → 图像平面
                                points_image = intrinsic_np @ points_cam[:3, :]
                                points_image_normalized = points_image / points_image[2, :]

                                # 过滤相机后方的点
                                valid_points = points_cam[2, :] > 0
                                update_color_index = np.where(valid_points)[0]

                                if valid_points.any():
                                    points_image_valid = points_image_normalized[:, valid_points]
                                    depth = points_image[2, valid_points]
                                    points_intensity = frame_lidar_intensity[valid_points]

                                    # 筛选在图像边界内的点
                                    within_bounds = (
                                        (points_image_valid[0, :] >= 0) &
                                        (points_image_valid[0, :] < cam_w) &
                                        (points_image_valid[1, :] >= 0) &
                                        (points_image_valid[1, :] < cam_h)
                                    )

                                    points_image_final = points_image_valid[:, within_bounds]
                                    depth = depth[within_bounds]
                                    points_intensity = points_intensity[within_bounds]
                                    update_color_index = update_color_index[within_bounds]

                                    # 过滤掉 modify_mask_m18 指定区域内的点
                                    exclusion_mask = np.ones((1, cam_h, cam_w), dtype=bool)
                                    modify_mask_m18(exclusion_mask, cam_idx)
                                    xc = np.round(points_image_final[0, :]).astype(int)
                                    yc = np.round(points_image_final[1, :]).astype(int)
                                    xc = np.clip(xc, 0, cam_w - 1)
                                    yc = np.clip(yc, 0, cam_h - 1)
                                    keep = exclusion_mask[0, yc, xc]
                                    points_image_final = points_image_final[:, keep]
                                    depth = depth[keep]
                                    points_intensity = points_intensity[keep]
                                    update_color_index = update_color_index[keep]

                                else:
                                    print(f"相机 {cam_name} 没有有效的投影点")
                                    continue

                                # 生成 mask 和 intensity map
                                mask = np.zeros((cam_h, cam_w), dtype=float)
                                intensity_map = np.zeros((cam_h, cam_w), dtype=np.uint8)

                                if points_image_final.shape[1] > 0:
                                    x_coords = np.round(points_image_final[0, :]).astype(int)
                                    y_coords = np.round(points_image_final[1, :]).astype(int)

                                    x_coords = np.clip(x_coords, 0, cam_w - 1)
                                    y_coords = np.clip(y_coords, 0, cam_h - 1)

                                    # update color rec
                                    if cam_idx in LIDAR_COLOR_FROM_CAM:
                                        pt2d_all[idx][update_color_index, 0] = cam_idx
                                        pt2d_all[idx][update_color_index, 1] = x_coords
                                        pt2d_all[idx][update_color_index, 2] = y_coords

                                    coords_depth = np.array(
                                        list(zip(x_coords, y_coords, depth, points_intensity)),
                                        dtype=[('x', int), ('y', int), ('depth', float), ('intensity', float)]
                                    )

                                    sorted_indices = np.lexsort(
                                        (coords_depth['depth'], coords_depth['y'], coords_depth['x']))
                                    coords_depth_sorted = coords_depth[sorted_indices]

                                    _, unique_indices = np.unique(
                                        coords_depth_sorted[['x', 'y']], axis=0, return_index=True
                                    )

                                    unique_coords_data = coords_depth_sorted[unique_indices]
                                    if len(unique_coords_data) > 0:
                                        # Only save lidar_depth, intensity and ego_pose_cam
                                        # if this camera is in PROC_CAMERA_IDX
                                        if PROC_CAMERA_IDX is None or cam_idx in PROC_CAMERA_IDX:
                                            x_coords_unique = unique_coords_data['x']
                                            y_coords_unique = unique_coords_data['y']
                                            depth_unique = unique_coords_data['depth']
                                            # intensity_unique = unique_coords_data['intensity']

                                            mask[y_coords_unique, x_coords_unique] = depth_unique
                                            # intensity_map[y_coords_unique, x_coords_unique] = np.asarray(
                                            #     intensity_unique, dtype=np.uint8)

                                            v_indices, u_indices = np.where(mask)
                                            value = mask[v_indices, u_indices]

                                            mask_obj = np.array({'mask': mask > 0, 'value': value}, dtype=object)
                                            mask_path = os.path.join(
                                                lidar_depth_dir, f"{frame_name}_{cam_idx:02d}.npy")
                                            np.save(mask_path, mask_obj)

                                            # cv2.imwrite(
                                            #     os.path.join(intensity_dir,
                                            #                  f"{frame_name}_{cam_idx:02d}.png"),
                                            #     intensity_map)
                                    else:
                                        raise ValueError(f"相机 {cam_name} 没有有效的投影点")
                                else:
                                    print(f"相机 {cam_name} 没有有效的投影点")
                                    continue

                                # 提取并保存 ego_pose_cam 数据 (always for all cameras)
                                ego2sensor = extrinsic_np
                                senser2ego = np.linalg.pinv(ego2sensor)
                                senser_pose = ego2global_transformation_matrix_np @ senser2ego

                                ego_pose_path = os.path.join(ego_pose_dir, file_name)
                                np.savetxt(ego_pose_path, senser_pose)
                            else:
                                print(f"警告：相机类型 {cam_name} 不在cameras_type中")
            except Exception as e:
                errors.append(f"Frame {frame_name} lidar error: {e}")

        # ============================================================
        # Phase: image
        # ============================================================
        if 'image' in proc_types:
            try:
                if 'sensors' in info and 'cams' in info['sensors']:
                    cams = info['sensors']['cams']

                    for cam_name, cam_data in cams.items():
                        if cam_name in cameras_type:
                            cam_idx = cameras_type[cam_name]

                            # Skip image output for cameras not in PROC_CAMERA_IDX
                            if PROC_CAMERA_IDX is not None and cam_idx not in PROC_CAMERA_IDX:
                                continue

                            img_path = cam_data.get('aws_path', '')
                            # 路径解析
                            img_path = resolve_path(img_path, data_root)

                            if not img_path or not os.path.exists(img_path):
                                # 尝试 data_path 备选
                                data_path = cam_data.get('data_path', '')
                                if data_path and data_path != 'N/A':
                                    data_path = resolve_path(data_path, data_root)
                                    if os.path.exists(data_path):
                                        img_path = data_path
                                        cam_data['aws_path'] = data_path

                            if not img_path or not os.path.exists(img_path):
                                print(f"警告：相机 {cam_name} 的图片路径不存在: {cam_data.get('aws_path', '')}")
                                raise Exception(f"图片路径不存在: {cam_name}")

                            # 生成新文件名
                            ext = os.path.basename(img_path).split('.')[-1]
                            new_img_name = f"{frame_name}_{cam_idx:02d}.{ext}"
                            new_img_path = os.path.join(images_dir, new_img_name)

                            try:
                                if downsample_enabled:
                                    dscale = ds_factors[cam_idx]
                                    img = cv2.imread(img_path)
                                    img = cv2.resize(img, (img.shape[1] // dscale, img.shape[0] // dscale),
                                                     interpolation=cv2.INTER_LINEAR)
                                    cv2.imwrite(new_img_path, img)
                                else:
                                    shutil.copy2(img_path, new_img_path)
                            except Exception as e:
                                print(f"复制图片失败 {img_path} -> {new_img_path}: {e}")
                        else:
                            print(f"警告：相机类型 {cam_name} 不在cameras_type中")
            except Exception as e:
                errors.append(f"Frame {frame_name} image error: {e}")

        # ============================================================
        # Phase: dynamic
        # ============================================================
        if 'dynamic' in proc_types:
            try:
                timestamp = str(info.get('timestamp', ''))[:-3]

                if 'ego2global_transformation_matrix' in info:
                    ego2global = np.array(info['ego2global_transformation_matrix'])
                else:
                    raise Exception(f"未找到ego2global变换矩阵: {timestamp}")

                # 处理所有相机
                if 'sensors' in info and 'cams' in info['sensors']:
                    cams = info['sensors']['cams']

                    # 创建所有相机的掩码
                    camera_masks = {}
                    camera_info = {}

                    for cam_name, cam_data in cams.items():
                        if cam_name not in cameras_type:
                            print(f"警告：相机类型 {cam_name} 不在cameras_type中")
                            raise Exception(f"相机类型 {cam_name} 不在cameras_type中")
                        cam_idx = cameras_type[cam_name]

                        # Skip dynamic mask for cameras not in PROC_CAMERA_IDX
                        if PROC_CAMERA_IDX is not None and cam_idx not in PROC_CAMERA_IDX:
                            continue

                        print(f"Worker {worker_id}: 处理帧 {frame_name}, 处理相机 {cam_name}",
                              end='\r', flush=True)

                        if 'cam_intrinsic' in cam_data and 'extrinsic' in cam_data:
                            camera_intrinsic = np.array(cam_data['cam_intrinsic'])
                            if downsample_enabled:
                                dscale = ds_factors[cam_idx]
                                camera_intrinsic[:2, :] = camera_intrinsic[:2, :] / dscale
                            camera_extrinsic = np.array(cam_data['extrinsic'])

                            mask_filename = f"{frame_name}_{cam_idx:02d}.png"
                            mask_path = os.path.join(dynamic_mask_dir2d, mask_filename)

                            # 获取图像尺寸
                            image_size = None
                            img_path = os.path.join(images_dir, f"{frame_name}_{cam_idx:02d}.jpg")
                            if os.path.exists(img_path):
                                img = cv2.imread(img_path)
                                if img is not None:
                                    image_size = (img.shape[1], img.shape[0])

                            if not image_size:
                                raise Exception(f"未找到相机 {cam_name} 的图片: {timestamp}")

                            mask = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
                            if DEBUG:
                                mask = img.copy()

                            camera_masks[cam_name] = {
                                'mask': mask,
                                'mask_path': mask_path,
                                'image_size': image_size
                            }

                            camera_info[cam_name] = {
                                'cam_idx': cam_idx,
                                'intrinsic': camera_intrinsic,
                                'extrinsic': camera_extrinsic
                            }

                    # 绘制物体边界框
                    if 'objects' not in info:
                        raise Exception(f"未找到物体信息: {timestamp}")

                    if len(info['objects']) == 0:
                        print(f"警告: 帧 {frame_name} 没有物体信息")
                    else:
                        for obj in info['objects']:
                            obj_velocity = obj.get('velocity', None)
                            if obj_velocity is None:
                                obj_velocity = [1.0, 1.0, 1.0]
                            obj_velocity_mag = np.linalg.norm(obj_velocity)

                            # 记录 track info
                            rotation = obj.get('rotation', [0, 0, 0])
                            track_info_line = [
                                str(idx),
                                str(obj['id']),
                                str(obj['type']),
                                "-10",
                                str(obj['size'][2]),
                                str(obj['size'][1]),
                                str(obj['size'][0]),
                                str(obj['location'][0]),
                                str(obj['location'][1]),
                                str(obj['location'][2]),
                                str(rotation[2]),
                                str(obj_velocity_mag)
                            ]
                            track_info_lines.append(track_info_line)

                            if obj['id'] not in track_camera_vis:
                                track_camera_vis[obj['id']] = {}
                            if idx not in track_camera_vis[obj['id']]:
                                track_camera_vis[obj['id']][idx] = []

                            # 过滤静态物体
                            if obj_velocity_mag < VELOCITY_THRESHOLD:
                                continue

                            # 在对应相机上绘制2D边界框
                            for info2d in obj.get('info2d', []):
                                if 'camera' not in info2d or 'bbox' not in info2d:
                                    continue
                                cam_name = info2d['camera']
                                if cam_name not in camera_info:
                                    continue
                                cam_idx = camera_info[cam_name]['cam_idx']
                                bbox = info2d['bbox']

                                track_camera_vis[obj['id']][idx].append(cameras_type[cam_name])

                                cam_mask_info = camera_masks[cam_name]
                                mask = cam_mask_info['mask']
                                image_size = cam_mask_info['image_size']

                                anchor_x, anchor_y, width, height = bbox
                                if downsample_enabled:
                                    dscale = ds_factors[cam_idx]
                                    anchor_x, anchor_y, width, height = \
                                        anchor_x / dscale, anchor_y / dscale, width / dscale, height / dscale

                                xmin = max(0, int(anchor_x - width / 2))
                                ymin = max(0, int(anchor_y - height / 2))
                                xmax = min(image_size[0], int(anchor_x + width / 2))
                                ymax = min(image_size[1], int(anchor_y + height / 2))

                                if DEBUG:
                                    cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (0, 0, 255), 10)
                                else:
                                    cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

                    # 绘制 Pure2DObjects
                    if PURE_2DOBJ:
                        for obj in info.get('Pure2DObjects', []):
                            cam_name = obj['camera']
                            if cam_name not in camera_info:
                                continue
                            cam_idx = camera_info[cam_name]['cam_idx']
                            bbox = obj['bbox']

                            cam_mask_info = camera_masks[cam_name]
                            mask = cam_mask_info['mask']
                            image_size = cam_mask_info['image_size']

                            anchor_x, anchor_y, width, height = bbox
                            if downsample_enabled:
                                dscale = ds_factors[cam_idx]
                                anchor_x, anchor_y, width, height = \
                                    anchor_x / dscale, anchor_y / dscale, width / dscale, height / dscale

                            xmin = max(0, int(anchor_x - width / 2))
                            ymin = max(0, int(anchor_y - height / 2))
                            xmax = min(image_size[0], int(anchor_x + width / 2))
                            ymax = min(image_size[1], int(anchor_y + height / 2))

                            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

                    # 保存所有相机的掩码
                    for cam_name, cam_mask_info in camera_masks.items():
                        mask = cam_mask_info['mask']
                        mask_path = cam_mask_info['mask_path']
                        if DEBUG:
                            mask_path = mask_path.replace('.png', '_debug.png')
                        cv2.imwrite(mask_path, mask)
            except Exception as e:
                errors.append(f"Frame {frame_name} dynamic error: {e}")

    # Save partial pointcloud data for this worker
    partial_npz_path = os.path.join(output_dir, f"pointcloud_part_{worker_id:03d}.npz")
    np.savez_compressed(partial_npz_path,
                        pointcloud=pt3d_all,
                        camera_projection=pt2d_all)

    return {
        'timestamps': timestamps,
        'timestamps_streetgs': timestamps_streetgs,
        'track_info_lines': track_info_lines,
        'track_camera_vis': track_camera_vis,
        'worker_id': worker_id,
        'errors': errors
    }


# ============================================================
# Helper: process a single lidar_depth visualization pair
# ============================================================
def process_lidar_depth_pair(args_tuple):
    """Process a single (lidar_depth_npy, image) pair for visualization."""
    lidar_f, img_f = args_tuple
    try:
        lidar = np.load(lidar_f, allow_pickle=True).item()
        img = cv2.imread(img_f)
        mask = lidar['mask']
        depth = lidar['value']

        h, w = img.shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)
        y_coords, x_coords = np.where(mask)

        if len(depth) == len(x_coords):
            for i, (x, y, d) in enumerate(zip(x_coords, y_coords, depth)):
                x_min = max(0, x - 2)
                x_max = min(w - 1, x + 2)
                y_min = max(0, y - 2)
                y_max = min(h - 1, y + 2)
                depth_map[y_min:y_max + 1, x_min:x_max + 1] = d

            valid_depth = depth_map > 0
            if np.any(valid_depth):
                min_depth = np.min(depth[depth > 0])
                max_depth = np.max(depth)
                depth_normalized = (depth_map - min_depth) / (max_depth - min_depth + 1e-6)
                depth_normalized = (depth_normalized * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
                mask_3channel = np.repeat((depth_map > 0)[:, :, np.newaxis], 3, axis=2)
                img_with_depth = img.copy()
                img_with_depth[mask_3channel] = depth_color[mask_3channel]
                output_path = os.path.join(os.path.dirname(lidar_f), os.path.basename(img_f))
                cv2.imwrite(output_path, img_with_depth)
                return f"保存带深度的图像: {output_path}"
            else:
                return f"没有有效的深度值: {lidar_f}"
        else:
            return f"深度值数量与掩码坐标数量不匹配: {lidar_f}"
    except Exception as e:
        return f"处理 {lidar_f} 失败: {e}"


# ============================================================
# Helper: merge timestamps_streetgs from multiple workers
# ============================================================
def merge_timestamps_streetgs(results):
    """Merge timestamps_streetgs dicts from all workers."""
    merged = {}
    for r in results:
        for key, value in r['timestamps_streetgs'].items():
            if key not in merged:
                merged[key] = {}
            merged[key].update(value)
    return merged


# ============================================================
# Helper: merge pointcloud partial npz files
# ============================================================
def merge_pointcloud_npz(output_dir, num_workers):
    """Merge partial pointcloud npz files into a single pointcloud.npz."""
    pt3d_all = {}
    pt2d_all = {}

    for worker_id in range(num_workers):
        partial_path = os.path.join(output_dir, f"pointcloud_part_{worker_id:03d}.npz")
        if os.path.exists(partial_path):
            data = np.load(partial_path, allow_pickle=True)
            # pointcloud and camera_projection are stored as arrays of dicts
            pt3d_part = data['pointcloud'].item()
            pt2d_part = data['camera_projection'].item()
            pt3d_all.update(pt3d_part)
            pt2d_all.update(pt2d_part)

    np.savez_compressed(os.path.join(output_dir, 'pointcloud.npz'),
                        pointcloud=pt3d_all,
                        camera_projection=pt2d_all)

    # Clean up partial files
    for worker_id in range(num_workers):
        partial_path = os.path.join(output_dir, f"pointcloud_part_{worker_id:03d}.npz")
        if os.path.exists(partial_path):
            os.remove(partial_path)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='M18 PKL converter (parallel) - reads PKL file and processes BEV data')
    parser.add_argument('--input', type=str, required=True,
                        help='Input PKL file path (e.g., clip_xxx.pkl)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory path')
    parser.add_argument('--data_root', type=str, default='/mnt/l3-labeled-data/prd_data/ALL',
                        help='Root directory for resolving relative paths in PKL '
                             '(e.g., /mnt/l3-labeled-data/prd_data/ALL)')
    parser.add_argument('--proc_type', nargs='+', type=str,
                        default=['image', 'cam', 'timestamp', 'lidar', 'dynamic', 'lidar_depth'],
                        help='Process type: image, cam, timestamp, lidar, dynamic, lidar_depth')
    parser.add_argument('--downsample', action='store_true', default=False,
                        help='Downsample images')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Limit number of frames to process (for debugging). '
                             'Shorthand for --start_frame 0 --end_frame N.')
    parser.add_argument('--start_frame', type=int, default=None,
                        help='Start frame index (inclusive). Default: 0.')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='End frame index (exclusive). Default: total frames.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel worker processes (default: cpu_count)')
    parser.add_argument('--proc_camera_idx', nargs='+', type=int, default=None,
                        help='Only output lidar_depth/image/dynamic_mask for these camera '
                             'indices (e.g. --proc_camera_idx 0 1 2 9 10). '
                             'Default: all cameras. Overrides PROC_CAMERA_IDX in the script.')
    args = parser.parse_args()

    # Apply CLI override for PROC_CAMERA_IDX
    if args.proc_camera_idx is not None:
        PROC_CAMERA_IDX = args.proc_camera_idx
        print(f"PROC_CAMERA_IDX overridden from CLI: {PROC_CAMERA_IDX}")

    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print(f"使用 {num_workers} 个并行 worker 进程")

    os.makedirs(args.output, exist_ok=True)

    # 0. 创建输出文件夹结构
    output_folders = ['ego_pose', 'extrinsics', 'images', 'lidar_depth', 'intrinsics', 'intensity']
    for folder in output_folders:
        folder_path = os.path.join(args.output, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"创建文件夹: {folder_path}")

    # Create dynamic_mask and track dirs if needed
    if 'dynamic' in args.proc_type:
        for folder in ['dynamic_mask', 'track']:
            folder_path = os.path.join(args.output, folder)
            os.makedirs(folder_path, exist_ok=True)
            print(f"创建文件夹: {folder_path}")

    # Apply downsample factors to image dimensions
    img_heights = list(image_heights)
    img_widths = list(image_widths)
    ds_factors = list(downsample_factors)
    if args.downsample:
        for i in range(len(ds_factors)):
            img_heights[i] //= ds_factors[i]
            img_widths[i] //= ds_factors[i]

    # 1. 加载PKL文件
    print(f"加载PKL文件: {args.input}")
    pkl_data = load_pkl(args.input)
    infos = pkl_data['infos']
    print(f"PKL中包含 {len(infos)} 帧数据")

    if DEBUG:
        infos = infos[:100]

    # Apply frame range selection
    start_frame = args.start_frame if args.start_frame is not None else 0
    end_frame = args.end_frame if args.end_frame is not None else len(infos)
    if args.max_frames is not None:
        end_frame = min(end_frame, start_frame + args.max_frames)
    start_frame = max(0, start_frame)
    end_frame = min(len(infos), max(start_frame, end_frame))
    if start_frame > 0 or end_frame < len(infos):
        infos = infos[start_frame:end_frame]
        print(f"Frame range: [{start_frame}, {end_frame}), total {len(infos)} frames")

    # 自动推断 data_root（如果未指定）
    if args.data_root is None:
        for info in infos:
            for cam_name, cam_data in info.get('sensors', {}).get('cams', {}).items():
                aws = cam_data.get('aws_path', '')
                if aws and os.path.isabs(aws) and os.path.exists(aws):
                    parts = aws.split('/')
                    for i, p in enumerate(parts):
                        if p.startswith('clip_') or (p.startswith('20') and len(p) >= 8 and p[:8].isdigit()):
                            args.data_root = '/'.join(parts[:i])
                            break
                    if args.data_root:
                        break
            if args.data_root:
                break
        if args.data_root:
            print(f"自动推断 data_root: {args.data_root}")
        else:
            print("警告: 未指定 --data_root 且无法自动推断，相对路径可能无法解析")

    # 2. Split frames into chunks for parallel processing
    # Use original frame indices (offset by start_frame) for correct naming
    total_frames = len(infos)
    # Ensure we don't have more workers than frames
    actual_workers = min(num_workers, total_frames) if total_frames > 0 else 1
    chunk_size = max(1, total_frames // actual_workers)

    chunks = []
    for w in range(actual_workers):
        start = w * chunk_size
        if w == actual_workers - 1:
            # Last chunk takes the remainder
            end = total_frames
        else:
            end = start + chunk_size
        chunk_infos = [(start_frame + idx, infos[idx]) for idx in range(start, end)]
        if chunk_infos:
            chunks.append(chunk_infos)

    print(f"将 {total_frames} 帧分为 {len(chunks)} 个chunk，每个chunk约 {chunk_size} 帧")

    # 3. Prepare arguments for each worker
    worker_args = []
    for worker_id, chunk_infos in enumerate(chunks):
        worker_args.append((
            chunk_infos,
            args.output,
            args.data_root,
            args.proc_type,
            img_heights,
            img_widths,
            ds_factors,
            args.downsample,
            worker_id
        ))

    # 4. Run parallel processing
    print(f"开始并行处理...")
    results = []
    with Pool(processes=actual_workers) as pool:
        results = pool.starmap(process_frame_chunk, worker_args)

    # 5. Report errors
    all_errors = []
    for r in results:
        if r['errors']:
            all_errors.extend(r['errors'])
    if all_errors:
        print(f"\n警告: 处理过程中出现 {len(all_errors)} 个错误:")
        for err in all_errors:
            print(f"  - {err}")

    # 6. Merge and write timestamp files
    if 'timestamp' in args.proc_type:
        print("合并 timestamps...")
        all_timestamps = []
        for r in sorted(results, key=lambda x: x['worker_id']):
            all_timestamps.extend(r['timestamps'])

        # Sort timestamps by FRAME_IDX
        all_timestamps.sort(key=lambda x: x['FRAME_IDX'])

        # Save timestamps_specific.json
        timestamps_path = os.path.join(args.output, 'timestamps_specific.json')
        with open(timestamps_path, 'w', encoding='utf-8') as f:
            json.dump(all_timestamps, f, indent=4, ensure_ascii=False)

        # Merge and save timestamps.json
        merged_streetgs = merge_timestamps_streetgs(results)
        timestamps_path = os.path.join(args.output, 'timestamps.json')
        with open(timestamps_path, 'w', encoding='utf-8') as f:
            json.dump(merged_streetgs, f, indent=4, ensure_ascii=False)
        print(f"生成timestamps.json，共 {len(all_timestamps)} 条记录")

    # 7. Merge pointcloud npz files
    if 'lidar' in args.proc_type:
        print("合并 pointcloud 数据...")
        merge_pointcloud_npz(args.output, len(chunks))
        print("Processing LiDAR data done...")

    # 8. Merge and write track files
    if 'dynamic' in args.proc_type:
        print("合并 track 数据...")
        all_track_lines = []
        merged_track_camera_vis = {}

        for r in results:
            all_track_lines.extend(r['track_info_lines'])
            for track_id, frame_dict in r['track_camera_vis'].items():
                if track_id not in merged_track_camera_vis:
                    merged_track_camera_vis[track_id] = {}
                merged_track_camera_vis[track_id].update(frame_dict)

        # Save track results
        track_dir = os.path.join(args.output, 'track')
        print('保存track result...')
        dict_to_json_file(merged_track_camera_vis, os.path.join(track_dir, 'track_camera_vis.json'), indent=1)
        header = 'frame_id track_id object_class alpha box_height box_width box_length box_center_x box_center_y box_center_z box_heading speed\n'
        with open(os.path.join(track_dir, 'track_info.txt'), 'w') as f:
            f.write(header)
            for l in all_track_lines:
                f.writelines(' '.join(l) + '\n')
        print("动态掩码和track信息生成完成")

    print("帧处理阶段完成")

    # ============================================================
    # 9. LiDAR depth 可视化（在所有帧处理完成后执行）
    # ============================================================
    if 'lidar_depth' in args.proc_type:
        print("生成 LiDAR 深度可视化...")
        lidar_fs = glob.glob(os.path.join(args.output, 'lidar_depth', f"*_10.npy"))
        img_fs = glob.glob(os.path.join(args.output, 'images', f"*_10.jpg"))
        lidar_fs.sort()
        img_fs.sort()

        if lidar_fs and img_fs:
            pair_args = list(zip(lidar_fs, img_fs))
            with Pool(processes=min(num_workers, len(pair_args))) as pool:
                viz_results = pool.map(process_lidar_depth_pair, pair_args)
            for msg in viz_results:
                print(msg, end='\r', flush=True)
            print("\nLiDAR 深度可视化完成")
        else:
            print(f"未找到 lidar_depth 文件 (lidar: {len(lidar_fs)}, img: {len(img_fs)})")

    print("转换完成！")
