import glob

from lib.utils.m18_utils import generate_dataparser_outputs
from lib.utils.graphics_utils import focal2fov, BasicPointCloud
from lib.utils.data_utils import get_val_frames
from lib.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, get_PCA_Norm, get_Sphere_Norm
from lib.config import cfg
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
import cv2
import sys
import copy
import shutil
import glob

sys.path.append(os.getcwd())


def readM18Info(path, images='images', split_train=-1, split_test=-1, **kwargs):
    selected_frames = cfg.data.get('selected_frames', None)
    if cfg.debug:
        selected_frames = [0, 0]

    if cfg.data.get('load_pcd_from', False) and (cfg.mode == 'train'):
        load_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'input_ply')
        save_dir = os.path.join(cfg.model_path, 'input_ply')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(load_dir, save_dir)

        colmap_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'colmap')
        save_dir = os.path.join(cfg.model_path, 'colmap')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(colmap_dir, save_dir)

    bkgd_ply_path = os.path.join(cfg.model_path, 'input_ply/points3D_bkgd.ply')
    build_pointcloud = (cfg.mode == 'train') and (
                not os.path.exists(bkgd_ply_path) or cfg.data.get('regenerate_pcd', False))

    # dynamic mask
    dynamic_mask_dir = os.path.join(path, 'dynamic_mask')
    load_dynamic_mask = (cfg.mode == 'train') and os.path.exists(dynamic_mask_dir)

    # sky mask
    sky_mask_dir = os.path.join(path, 'sky_mask')
    load_sky_mask = (cfg.mode == 'train') and os.path.exists(sky_mask_dir)

    # lidar depth
    lidar_depth_dir = os.path.join(path, 'lidar_depth')
    load_lidar_depth = (cfg.mode == 'train') and os.path.exists(lidar_depth_dir)

    output = generate_dataparser_outputs(
        datadir=path,
        selected_frames=selected_frames,
        build_pointcloud=build_pointcloud,
        cameras=cfg.data.get('cameras', [0, 1, 2]),
    )

    # read external colmap data
    ext_COLMAP_ROOT = cfg.get('ext_colmap_data', None)
    if ext_COLMAP_ROOT is not None:
        import open3d as o3d
        from ysutils.util_colmap import read_model, colmap_points3d_to_ply
        from ysutils.util_open3d import icp_pcd, gen_cam_frust, solve_similarity_from_correspondences, decompose_transform_matrix

        proc_cam_id = cfg.data.cameras
        start_frame, end_frame = cfg.data.selected_frames[0], cfg.data.selected_frames[1]
        num_frames = end_frame - start_frame + 1

        cameras, images, points3D = read_model(os.path.join(ext_COLMAP_ROOT, '0'))

        rec_colmap_poses = {}
        rec_cam_intrin = {}
        for idx, image in images.items():
            cam_img = image.name
            cam_id = int(cam_img.split('/')[0].split('_')[-1])
            frame_id = int(cam_img.split('/')[-1].split('.')[0])
            if frame_id<start_frame or frame_id>end_frame or cam_id not in proc_cam_id: continue

            if frame_id not in rec_colmap_poses:
                rec_colmap_poses[frame_id] = {}
            if cam_id not in rec_colmap_poses[frame_id]:
                rec_colmap_poses[frame_id][cam_id] = {}
            if cam_id not in rec_cam_intrin:
                rec_cam_intrin[cam_id] = {}
                rec_cam_intrin[cam_id]['colmap_cam_idx'] = image.camera_id

            col_extrin = np.eye(4)
            col_extrin[:3, :3] = image.qvec2rotmat()
            col_extrin[:3, 3] = image.tvec
            col_pose = np.linalg.inv(col_extrin)
            rec_colmap_poses[frame_id][cam_id]['colmap_pose'] = col_pose

        for real_id, cam_data in rec_cam_intrin.items():
            if real_id not in proc_cam_id: continue
            cam_param = cameras[cam_data['colmap_cam_idx']].params
            intrin = np.eye(3)
            intrin[0, 0] = cam_param[0]
            intrin[1, 1] = cam_param[1]
            intrin[0, 2] = cam_param[2]
            intrin[1, 2] = cam_param[3]
            rec_cam_intrin[real_id]['intrin'] = intrin

        # align the colmap and the input data
        m18_poses = output['c2ws']
        colmap_poses = []
        frames, cams = output['frames'], output['cams']
        for frame_id, cam_id in zip(frames, cams):
            colmap_poses.append(rec_colmap_poses[frame_id][cam_id]['colmap_pose'])

        colmap_poses = np.array(colmap_poses)
        assert m18_poses.shape[0] == colmap_poses.shape[0]

        source_point = np.array([m[:3,3] for m in colmap_poses])
        target_point = np.array([m[:3,3] for m in m18_poses])

        max_iterations = 200  # ICP迭代次数
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
        # source.paint_uniform_color([1, 0, 0])  # 红色
        # target.paint_uniform_color([0, 1, 0])  # 绿色
        # o3d.visualization.draw_geometries([source,target])

        # redraw the frustums
        trans_poses_raw = [T_final @ pose for pose in colmap_poses]
        trans_poses = []
        for pose in trans_poses_raw:
            pose_scale, pose_rotation, pose_translation, pose_rot_mat =  decompose_transform_matrix(pose)
            new_pose = np.eye(4)
            new_pose[:3, :3] = pose_rot_mat
            new_pose[:3, 3] = pose_translation
            trans_poses.append(new_pose)

        # trans_all_frusts = gen_cam_frust(trans_poses, color=[0, 1, 0])
        # m18_all_frusts = gen_cam_frust(m18_poses, color=[1, 0, 0])
        # o3d.visualization.draw_geometries(m18_all_frusts+trans_all_frusts)

        # align the pointcloud
        colmap_pcd_path = os.path.join(ext_COLMAP_ROOT, '0/points3D.ply')
        if not os.path.exists(colmap_pcd_path):
            colmap_points3d_to_ply(os.path.join(ext_COLMAP_ROOT, '0/points3D.txt'),colmap_pcd_path)
        col_pcd = o3d.io.read_point_cloud(colmap_pcd_path)
        col_pcd_clean, ind = col_pcd.remove_statistical_outlier(nb_neighbors=5,
                                                                std_ratio=0.01)
        m18_statics_pcd = o3d.io.read_point_cloud(os.path.join(cfg.model_path, 'input_ply', 'points3D_bkgd.ply'))

        transform_col_pcd = o3d.geometry.PointCloud()
        points = []
        points_homogeneous = np.hstack(
            [np.asarray(col_pcd_clean.points), np.ones((np.asarray(col_pcd_clean.points).shape[0], 1))])
        # 执行矩阵乘法
        transformed_points = T_final @ points_homogeneous.T  # 结果是 4×n
        # 转换回笛卡尔坐标，得到 n×3 的结果
        result = transformed_points[:3, :].T  # 取前3行并转置，得到 n×3
        points.append(result)
        transform_col_pcd.points = o3d.utility.Vector3dVector(np.concatenate(points, axis=0))
        transform_col_pcd.colors = o3d.utility.Vector3dVector(col_pcd_clean.colors)

        # o3d.visualization.draw_geometries([m18_statics_pcd, transform_col_pcd])
        from lib.datasets.base_readers import storePly
        storePly(os.path.join(cfg.model_path, 'input_ply','points3D_colmap.ply'), np.asarray(transform_col_pcd.points), np.asarray(col_pcd_clean.colors))

        if os.path.exists(os.path.join(cfg.model_path, 'input_ply','points3D_raw_bkgd.ply')):
            pass
        else:
            shutil.move(os.path.join(cfg.model_path, 'input_ply','points3D_bkgd.ply'), os.path.join(cfg.model_path, 'input_ply','points3D_raw_bkgd.ply'))
            shutil.copy(os.path.join(cfg.model_path, 'input_ply','points3D_colmap.ply'),os.path.join(cfg.model_path, 'input_ply','points3D_bkgd.ply'))


        # o3d.visualization.draw_geometries([o3d.io.read_point_cloud(os.path.join(cfg.model_path, 'input_ply','points3D_colmap.ply'))])
        # o3d.io.write_point_cloud(os.path.join(cfg.model_path, 'input_ply','points3D_colmap.ply'), transform_col_pcd)


    exts = output['exts']
    ixts = output['ixts']
    poses = output['poses']
    c2ws = output['c2ws']
    image_filenames = output['image_filenames']
    obj_tracklets = output['obj_tracklets']
    obj_info = output['obj_info']
    frames, cams = output['frames'], output['cams']
    frames_idx = output['frames_idx']
    num_frames = output['num_frames']
    cams_timestamps = output['cams_timestamps']
    tracklet_timestamps = output['tracklet_timestamps']
    obj_bounds = output['obj_bounds']
    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    if ext_COLMAP_ROOT is not None:
        ixts = []
        exts = [np.linalg.pinv(ego2global)@ sensor2global for sensor2global,ego2global in zip(trans_poses, output['poses'])]
        # exts = [np.linalg.pinv(ego2global)@ sensor2global for sensor2global,ego2global in zip(output['c2ws'], output['poses'])]
        c2ws = trans_poses
        for frame_id, cam_id in zip(frames, cams):
            # exts.append(None)
            ixts.append(rec_cam_intrin[cam_id]['intrin'] )


    scene_metadata = dict()
    scene_metadata['obj_tracklets'] = obj_tracklets
    scene_metadata['tracklet_timestamps'] = tracklet_timestamps
    scene_metadata['obj_meta'] = obj_info
    scene_metadata['num_images'] = len(exts)
    scene_metadata['num_cams'] = len(cfg.data.cameras)
    scene_metadata['num_frames'] = num_frames

    camera_timestamps = dict()
    for cam in cfg.data.get('cameras', [0, 1, 2]):
        camera_timestamps[cam] = dict()
        camera_timestamps[cam]['train_timestamps'] = []
        camera_timestamps[cam]['test_timestamps'] = []

        ########################################################################################################################
    cam_infos = []
    for i in tqdm(range(len(exts))):
        # generate pose and image
        ext = exts[i]
        ixt = ixts[i]
        c2w = c2ws[i]
        pose = poses[i]
        image_path = image_filenames[i]
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)

        width, height = image.size
        fx, fy = ixt[0, 0], ixt[1, 1]
        FovY = focal2fov(fx, height)
        FovX = focal2fov(fy, width)

        RT = np.linalg.inv(c2w)
        R = RT[:3, :3].T
        T = RT[:3, 3]

        # print(T)
        K = ixt.copy()

        metadata = dict()
        metadata['frame'] = frames[i]
        metadata['cam'] = cams[i]
        metadata['frame_idx'] = frames_idx[i]
        metadata['ego_pose'] = pose
        metadata['extrinsic'] = ext
        metadata['timestamp'] = cams_timestamps[i]

        if frames_idx[i] in train_frames:
            metadata['is_val'] = False
            camera_timestamps[cams[i]]['train_timestamps'].append(cams_timestamps[i])
        else:
            metadata['is_val'] = True
            camera_timestamps[cams[i]]['test_timestamps'].append(cams_timestamps[i])

        guidance = dict()
        guidance['obj_bound'] = Image.fromarray(obj_bounds[i])
        # load dynamic mask
        if load_dynamic_mask:
            # bboxs = {
            #     3: [0, 0, 520, 200,1920,1280],
            #     4: [1500, 0, 1920, 135,1920,1280],
            #     10: [0, 750, 1920, 1080,1920,1080],
            # }
            # def draw_mask(bbox, mask):
            #     xmin, ymin, xmax, ymax, w, h = bbox
            #     H,W = mask.shape
            #     assert h==H and w==W, f'h={h}, w={w}, H={H}, W={W}'
            #     mask = np.asarray(mask, dtype=np.uint8)
            #     cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)
            #     mask = mask >0
            #     return mask
            #     # 绘制红色边界框
            #     # cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (0, 0, 255), 10)

            dynamic_mask_path = os.path.join(dynamic_mask_dir, f'{image_name}.png')
            obj_bound = (cv2.imread(dynamic_mask_path)[..., 0]) > 0.
            # real_cam_id = int(image_name.split('_')[-1])
            # if real_cam_id in bboxs:
            #     obj_bound = draw_mask(bboxs[real_cam_id],obj_bound)

            obj_bound = Image.fromarray(obj_bound)
            guidance['obj_bound'] = obj_bound
            # guidance['obj_bound'] = Image.fromarray(obj_bounds[i])

        # load lidar depth
        if load_lidar_depth:
            depth_path = os.path.join(lidar_depth_dir, f'{image_name}.npy')
            depth = np.load(depth_path, allow_pickle=True)
            depth = dict(depth.item())
            mask = depth['mask']
            value = depth['value']
            depth = np.zeros_like(mask).astype(np.float32)
            depth[mask] = value
            guidance['lidar_depth'] = depth


        #     # test
        #
        #     # Compute global coordinates from lidar depth
        #     # Get valid pixels
        #     valid_mask = depth > 0
        #     v_indices, u_indices = np.where(valid_mask)
        #     d_values = depth[valid_mask]
        #
        #     if len(d_values) > 0:
        #         # Camera intrinsic parameters
        #         cx = ixt[0, 2]
        #         cy = ixt[1, 2]
        #         fx = ixt[0, 0]
        #         fy = ixt[1, 1]
        #
        #         # Convert pixel coordinates and depth to camera space 3D points
        #         x_cam = (u_indices - cx) * d_values / fx
        #         y_cam = (v_indices - cy) * d_values / fy
        #         z_cam = d_values
        #
        #         cam_points = np.stack([x_cam, y_cam, z_cam], axis=1)
        #         cam_points_hom = np.hstack([cam_points, np.ones((cam_points.shape[0], 1))])
        #
        #         #
        #         import open3d as o3d
        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(cam_points)
        #         o3d.visualization.draw_geometries([pcd])
        #     # end

        # load sky mask
        if load_sky_mask:
            sky_mask_path = glob.glob(os.path.join(sky_mask_dir, f'{image_name}.*'))[0]
            # sky_mask_path = os.path.join(sky_mask_dir, f'{image_name}.png')
            sky_mask = (cv2.imread(sky_mask_path)[..., 0]) > 0.
            guidance['sky_mask'] = Image.fromarray(sky_mask)

        # mask = None

        cam_info = CameraInfo(
            uid=i, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
            image=image, image_path=image_path, image_name=image_name,
            width=width, height=height,
            metadata=metadata,
            guidance=guidance,
        )
        cam_infos.append(cam_info)

        # sys.stdout.write('\n')

    train_cam_infos = [cam_info for cam_info in cam_infos if not cam_info.metadata['is_val']]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata['is_val']]

    for cam in cfg.data.get('cameras', [0, 1, 2]):
        camera_timestamps[cam]['train_timestamps'] = sorted(camera_timestamps[cam]['train_timestamps'])
        camera_timestamps[cam]['test_timestamps'] = sorted(camera_timestamps[cam]['test_timestamps'])
    scene_metadata['camera_timestamps'] = camera_timestamps

    novel_view_cam_infos = []

    #######################################################################################################################3
    # Get scene extent
    # 1. Default nerf++ setting
    if cfg.mode == 'novel_view':
        nerf_normalization = getNerfppNorm(novel_view_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)

    # 2. The radius we obtain should not be too small (larger than 10 here)
    nerf_normalization['radius'] = max(nerf_normalization['radius'], 10)

    # 3. If we have extent set in config, we ignore previous setting
    if cfg.data.get('extent', False):
        nerf_normalization['radius'] = cfg.data.extent

    # 4. We write scene radius back to config
    cfg.data.extent = float(nerf_normalization['radius'])

    # 5. We write scene center and radius to scene metadata
    scene_metadata['scene_center'] = nerf_normalization['center']
    scene_metadata['scene_radius'] = nerf_normalization['radius']
    print(f'Scene extent: {nerf_normalization["radius"]}')

    # Get sphere center
    lidar_ply_path = os.path.join(cfg.model_path, 'input_ply/points3D_lidar.ply')
    if os.path.exists(lidar_ply_path):
        sphere_pcd: BasicPointCloud = fetchPly(lidar_ply_path)
    else:
        sphere_pcd: BasicPointCloud = fetchPly(bkgd_ply_path)

    sphere_normalization = get_Sphere_Norm(sphere_pcd.points)
    scene_metadata['sphere_center'] = sphere_normalization['center']
    scene_metadata['sphere_radius'] = sphere_normalization['radius']
    print(f'Sphere extent: {sphere_normalization["radius"]}')

    pcd: BasicPointCloud = fetchPly(bkgd_ply_path)
    if cfg.mode == 'train':
        point_cloud = pcd
    else:
        point_cloud = None
        bkgd_ply_path = None

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=bkgd_ply_path,
        metadata=scene_metadata,
        novel_view_cameras=novel_view_cam_infos,
    )

    return scene_info



