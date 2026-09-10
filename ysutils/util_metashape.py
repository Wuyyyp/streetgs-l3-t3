import open3d as o3d
import numpy as np
import xmltodict
from .util_megasurf import get_bbox_from_pcd
from .util_mvsnet import write_cam
import os
import tqdm
import glob
import cv2
import ysutils
import math
import json

metadict = {
  "scene_split": {
    "split_side_length": 100.0,
    "overlap_side_length": 25.0,
    "z_extend": 5.0
  }
}

def parse_groupdict(photos_dict):
    W = int(photos_dict['ImageDimensions']['Width'])
    H = int(photos_dict['ImageDimensions']['Height'])
    if 'FocalLengthPixels' in photos_dict.keys():
        FocalLengthPixels = float(photos_dict['FocalLengthPixels'])
    if 'FocalLength' in photos_dict.keys() and 'SensorSize' in photos_dict.keys():
        FocalLengthPixels_x = float(photos_dict['FocalLength']) / (float(photos_dict['SensorSize']) / W)
        FocalLengthPixels_y = float(photos_dict['FocalLength']) / (float(photos_dict['SensorSize']) / H)
        FocalLengthPixels = np.maximum(FocalLengthPixels_x, FocalLengthPixels_y)
        print(f'warning please check the focal length: {FocalLengthPixels}')
    cx, cy = float(photos_dict['PrincipalPoint']['x']), float(photos_dict['PrincipalPoint']['y'])
    k1, k2, k3, p1, p2 = float(photos_dict['Distortion']['K1']), float(photos_dict['Distortion']['K2']), float(
        photos_dict['Distortion']['K3']), float(photos_dict['Distortion']['P1']), float(photos_dict['Distortion']['P2'])
    n_image = photos_dict['Photo'].__len__()
    intrindict = {}
    intrindict['H'] = H
    intrindict['W'] = W
    intrindict['FocalLengthPixels'] = FocalLengthPixels
    intrindict['cx'] = cx
    intrindict['cy'] = cy
    intrindict['k1'] = k1
    intrindict['k2'] = k2
    intrindict['k3'] = k3
    intrindict['p1'] = p1
    intrindict['p2'] = p2
    intrindict['n_image'] = n_image

    photos_rec_dict = {}
    for photo_dict in photos_dict['Photo']:

        photo_name = os.path.basename(photo_dict['ImagePath'])
        id = int(photo_dict['Id'])
        valid_flag = int(photo_dict['Component']) != 0
        if not valid_flag:
            f'{photo_name} pose estimation has error!'
            continue
        rot_mat = np.zeros((3, 3))
        rot_mat[0, 0] = float(photo_dict['Pose']['Rotation']['M_00'])
        rot_mat[0, 1] = float(photo_dict['Pose']['Rotation']['M_01'])
        rot_mat[0, 2] = float(photo_dict['Pose']['Rotation']['M_02'])
        rot_mat[1, 0] = float(photo_dict['Pose']['Rotation']['M_10'])
        rot_mat[1, 1] = float(photo_dict['Pose']['Rotation']['M_11'])
        rot_mat[1, 2] = float(photo_dict['Pose']['Rotation']['M_12'])
        rot_mat[2, 0] = float(photo_dict['Pose']['Rotation']['M_20'])
        rot_mat[2, 1] = float(photo_dict['Pose']['Rotation']['M_21'])
        rot_mat[2, 2] = float(photo_dict['Pose']['Rotation']['M_22'])
        rot_mat = np.linalg.pinv(rot_mat)

        t_vec = np.zeros((3,))
        t_vec[0] = float(photo_dict['Pose']['Center']['x'])
        t_vec[1] = float(photo_dict['Pose']['Center']['y'])
        t_vec[2] = float(photo_dict['Pose']['Center']['z'])

        photo_rec_dict = {}
        photo_rec_dict['id'] = photo_dict['ImagePath']
        photo_rec_dict['img_path'] = photo_dict['ImagePath']
        photo_rec_dict['pose_rot'] = rot_mat
        photo_rec_dict['pose_t'] = t_vec
        photo_rec_dict['photo_name'] = photo_name
        photos_rec_dict[id] = photo_rec_dict
    return photos_rec_dict,intrindict
def readxml_urbanscene3d(fxml):
    with open(fxml, 'r') as xml:
        print(f'opening {fxml}')
        content = xml.read()

    # import time
    # print(time.ctime())
    camdict: dict = xmltodict.parse(content)
    # print(time.ctime())
    # camdict: dict = quick_xmltodict.parse(content)
    # print(time.ctime())

    # parse camera pose
    assert camdict['BlocksExchange']['Block']['Photogroups'].keys().__len__() == 1,'More than 1 block. Not implemented.'
    camgroups_dict={}
    if type(camdict['BlocksExchange']['Block']['Photogroups']['Photogroup']) == dict:
        ngroup = 1
        photos_rec_dict, intrindict = parse_groupdict(camdict['BlocksExchange']['Block']['Photogroups']['Photogroup'])
        camgroups_dict[0] = {'group_0_pose':photos_rec_dict,'group_0_intrin':intrindict}
    elif type(camdict['BlocksExchange']['Block']['Photogroups']['Photogroup']) == list:
        ngroup = len(camdict['BlocksExchange']['Block']['Photogroups']['Photogroup'])
        for idx_group in range(ngroup):
            photos_dict = camdict['BlocksExchange']['Block']['Photogroups']['Photogroup'][idx_group]
            photos_rec_dict, intrindict = parse_groupdict(photos_dict)
            camgroups_dict[idx_group] = {f'group_{idx_group}_pose': photos_rec_dict, f'group_{idx_group}_intrin': intrindict}
    else:
        raise NotImplementedError('Not implemented.')

    # parse tie points
    # nties = len(camdict['BlocksExchange']['Block']['TiePoints']['TiePoint'])
    # ties_list = camdict['BlocksExchange']['Block']['TiePoints']['TiePoint']
    # ties_position_list = []
    # ties_measurement_list = []
    # for i in range(nties):
    #     point = np.asarray([ties_list[i]['Position']['x'],ties_list[i]['Position']['y'],ties_list[i]['Position']['z']],dtype=np.float32)
    #     ties_position_list.append(point)
    #     ties_measurement_list.append(ties_list[i]['Measurement'])
    return camgroups_dict #, ties_position_list, ties_measurement_list

def readnvm_urbanscene3d(fxml):
    with open(fxml,'r') as xml:
        print(f'opening {fxml}')
        content = xml.readlines()
    n_cameras = int(content[2].strip())
    filenames = []
    tiepoints = []
    assert content[n_cameras + 3] == '\n', "nvm file format err"
    for f in content[3:n_cameras + 3]:
        filenames.append(f.split('\t')[0])

    n_tiepoints = int(content[n_cameras + 4].strip())
    ties_xyz = []
    ties_img = []

    for tie in tqdm.tqdm(content[n_cameras + 5 : n_cameras + 5 + n_tiepoints]):
        tie = tie.strip().split(' ')
        ties_xyz.append(np.asarray(tie[:3],dtype = np.float32))
        ties_img.append(np.asarray(tie[7:7+ int(tie[6]) * 4:4],dtype=np.int32))
    print("process tie points done")
    return n_cameras, n_tiepoints, ties_xyz, ties_img, filenames

def generate_pair_Fmat(Fmat):
    sorted_indices_row = np.argsort(Fmat,axis=1)
    print("generate matching according to the tiepoints")
    return sorted_indices_row[:,::-1]

def generate_img_tiepoints(n_cameras, ties_img_id,valid_mask=None):
    Fmat = np.zeros([n_cameras,n_cameras],dtype=np.int32)
    if valid_mask is not None:
        assert valid_mask.shape[0] == len(ties_img_id),'data doesnt match'
        for i in tqdm.tqdm(range(valid_mask.shape[0])):
            if valid_mask[i] == False:
                continue
            rec = ties_img_id[i]
            x, y = np.meshgrid(rec, rec)
            x = x.reshape(-1)
            y = y.reshape(-1)
            Fmat[x, y] += 1
    else:
        for rec in tqdm.tqdm(ties_img_id):
            x,y = np.meshgrid(rec,rec)
            x = x.reshape(-1)
            y = y.reshape(-1)
            Fmat[x,y] += 1
    return Fmat

def split_scene(args):

    os.makedirs(args.out,exist_ok=True)
    camxml_file = glob.glob(os.path.join(args.root,'cameras_local.xml'))[0]
    nvm_file = glob.glob(os.path.join(args.root,'*.nvm'))[0]
    tie_provide = False
    try:
        tie_file = glob.glob(os.path.join(args.root,'tiepoints_local.ply'))[0]
        tie_local = o3d.io.read_point_cloud(tie_file)
        tie_provide=True
    except:
        pass


    # read cameras xml
    camgroups_dict = readxml_urbanscene3d(camxml_file)
    # read nvm file
    n_cameras, n_tiepoints, ties_xyz, ties_img_id, nvm_filenames = readnvm_urbanscene3d(nvm_file)
    # check id consistency
    poses = []
    intrins = []
    meta = []

    for group_id in camgroups_dict:
        intrin_info = camgroups_dict[group_id][f'group_{group_id}_intrin']
        intrin = np.eye(3)
        intrin[0,0] = intrin_info['FocalLengthPixels']
        intrin[1,1] = intrin_info['FocalLengthPixels']
        intrin[0,2] = intrin_info['cx']
        intrin[1,2] = intrin_info['cy']
        for camid in camgroups_dict[group_id][f'group_{group_id}_pose']:
            camdict = camgroups_dict[group_id][f'group_{group_id}_pose'][camid]
            meta.append([camid,camdict['img_path'],intrin_info['H'],intrin_info['W']])
            pose = np.eye(4)
            pose[:3,:3] = camdict['pose_rot']
            pose[:3,3] = camdict['pose_t']
            poses.append(pose)
            intrins.append(intrin)
            assert camdict['photo_name'] == nvm_filenames[camid], 'nvm doesnt match the camera xml'

    # output cameras
    os.makedirs(os.path.join(args.out,'cams'),exist_ok=True)
    for i in range(len(nvm_filenames)):
        imgname,intrin,pose = nvm_filenames[i], intrins[i], poses[i]

        # ysutils.util_colmap.qvec2rotmat(np.asarray([-0.140364162659,0.373258306649,0.848121322598,0.348807053417]))
        # nvm qvec2rot R: extrin_R, t: pose_t
        extrin = np.linalg.pinv(pose)
        depthstat = np.zeros((4,),dtype=float)
        depthstat[0] = 0 # near
        depthstat[3] = 100 # far = min(median * 4, far)
        depthstat[1] = (depthstat[3] - depthstat[0])/192
        depthstat[2] = 192
        write_cam(os.path.join(args.out,'cams',f'{imgname.split(".")[0]}_cam.txt'),intrin,extrin,depthstat) # file, intrinsic, extrinsic, depth_stat
        meta[i].append(os.path.join(args.out,'cams',f'{imgname.split(".")[0]}_cam.txt'))
    print('output cameras')

    np.savetxt(os.path.join(args.out,f'meta.txt'),meta,fmt='%s')

    # cal matching score
    Fmat = generate_img_tiepoints(n_cameras, ties_img_id)
    np.save(os.path.join(args.out,'Fmat.npy'),Fmat)
    matching = generate_pair_Fmat(Fmat)
    np.save(os.path.join(args.out,'matching.npy'),matching)

    # generate bounding box for each splits
    # read all tiepoints


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(ties_xyz))
    print('cleaning the outliers')
    if os.path.exists(os.path.join(args.root,'tiepoints_local_clean.ply')):
        print('use user provided point cloud')
        pcd_clean = o3d.io.read_point_cloud(os.path.join(args.root,'tiepoints_local_clean.ply'))
    else:
        pcd_clean = pcd.remove_radius_outlier(20,5)[0]
    print('cleaning done')

    o3d.io.write_point_cloud(os.path.join(args.out,'clean_tiepoints.ply'),pcd_clean)
    # pcd_clean2 = pcd.remove_statistical_outlier(20, 0.1)
    meta_aabb :o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.PointCloud.get_axis_aligned_bounding_box(pcd_clean)

    block_length = metadict['scene_split']['split_side_length'] + metadict['scene_split']['overlap_side_length'] * 2

    nblocks_min = np.ceil(np.abs(meta_aabb.min_bound)/metadict['scene_split']['split_side_length'])
    nblocks_max = np.ceil(np.abs(meta_aabb.max_bound)/metadict['scene_split']['split_side_length'])

    x_coord = (np.arange(-nblocks_min[0], nblocks_max[0]) + 0.5) * metadict['scene_split']['split_side_length']
    y_coord = (np.arange(-nblocks_min[1], nblocks_max[1]) + 0.5) * metadict['scene_split']['split_side_length']

    splits_bbox = []
    for x_c in x_coord:
        for y_c in y_coord:
            split_bbox_min = [x_c - block_length/2, y_c - block_length/2, meta_aabb.min_bound[-1] - metadict['scene_split']['z_extend']]
            split_bbox_max = [x_c + block_length/2, y_c + block_length/2, meta_aabb.max_bound[-1] + metadict['scene_split']['z_extend']]
            splits_bbox.append(np.asarray(split_bbox_min + split_bbox_max))
    os.makedirs(os.path.join(args.out,"splits"), exist_ok=True)
    np.savetxt(os.path.join(args.out,"splits",f'splits_bbox.txt'),splits_bbox)

    # make output folders
    n_tiepoints_within = []
    os.makedirs(os.path.join(args.out, "splits", 'bbox_real'), exist_ok=True)
    for i in range(len(splits_bbox)):
        tempcd = o3d.geometry.PointCloud()
        tempcd.points = o3d.utility.Vector3dVector(splits_bbox[i].reshape(-1,3))
        cube, aabb = get_bbox_from_pcd(tempcd)
        # tie points within the bbox
        if tie_provide:
            croppcd = tie_local.crop(aabb)
        else:
            croppcd = pcd.crop(aabb)
        n_tiepoints_within.append(len(croppcd.points))
        if len(croppcd.points) < 5000:
            continue
        print(f"processing part {i}")
        print(f"contains {len(croppcd.points)} points")

        os.makedirs(os.path.join(args.out,"splits",f'{i}'), exist_ok=True)
        o3d.io.write_triangle_mesh(os.path.join(args.out,"splits",f'{i}',f'realworld_bbox.ply'),cube)
        o3d.io.write_triangle_mesh(os.path.join(args.out,"splits",'bbox_real',f'realworld_bbox_{i}.ply'),cube)


        o3d.io.write_point_cloud(os.path.join(args.out, "splits", f'{i}', f'tiepoints.ply'), croppcd)
        valid_mask = ((np.asarray(pcd.points) >= aabb.min_bound) * (np.asarray(pcd.points) <= aabb.max_bound)).sum(axis=-1) == 3

        Fmat_split = generate_img_tiepoints(n_cameras, ties_img_id, valid_mask=valid_mask)
        np.save(os.path.join(args.out, "splits", f'{i}', 'Fmat_split.npy'), Fmat_split)
        matching_split = generate_pair_Fmat(Fmat_split)
        np.save(os.path.join(args.out, "splits", f'{i}', 'matching_split.npy'), matching_split)

    np.savetxt(os.path.join(args.out,"splits",f'nties_splits.txt'),np.asarray(n_tiepoints_within),fmt='%i')
    print("ok")

def process_single(args):
    os.makedirs(args.out,exist_ok=True)
    camxml_file = glob.glob(os.path.join(args.root,'cameras_local.xml'))[0]
    nvm_file = glob.glob(os.path.join(args.root,'*.nvm'))[0]
    tie_file = glob.glob(os.path.join(args.root,'tiepoints_local.ply'))[0]

    # read cameras xml
    camgroups_dict = readxml_urbanscene3d(camxml_file)
    # read nvm file
    n_cameras, n_tiepoints, ties_xyz, ties_img_id, nvm_filenames = readnvm_urbanscene3d(nvm_file)
    # check id consistency
    poses = []
    intrins = []
    meta = []
    distort = []
    for group_id in camgroups_dict:
        intrin_info = camgroups_dict[group_id][f'group_{group_id}_intrin']
        intrin = np.eye(3)
        intrin[0,0] = intrin_info['FocalLengthPixels']
        intrin[1,1] = intrin_info['FocalLengthPixels']
        intrin[0,2] = intrin_info['cx']
        intrin[1,2] = intrin_info['cy']
        k1k2k3p1p2 = np.asarray([intrin_info['k1'],intrin_info['k2'],intrin_info['k3'],intrin_info['p1'],intrin_info['p2']])
        for camid in camgroups_dict[group_id][f'group_{group_id}_pose']:
            camdict = camgroups_dict[group_id][f'group_{group_id}_pose'][camid]
            meta.append([camid,camdict['img_path'],intrin_info['H'],intrin_info['W']])
            pose = np.eye(4)
            pose[:3,:3] = camdict['pose_rot']
            pose[:3,3] = camdict['pose_t']
            poses.append(pose)
            intrins.append(intrin)
            distort.append(k1k2k3p1p2)
            assert camdict['photo_name'] == nvm_filenames[camid], 'nvm doesnt match the camera xml'

    # read all tiepoints
    pcd = o3d.io.read_point_cloud(tie_file)
    meta_aabb: o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.PointCloud.get_axis_aligned_bounding_box(pcd)
    radius = meta_aabb.get_extent().min()/50
    pcd_clean = pcd.remove_radius_outlier(20,radius)[0]
    meta_aabb :o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.PointCloud.get_axis_aligned_bounding_box(pcd_clean)
    cube, aabb = get_bbox_from_pcd(pcd_clean)
    o3d.io.write_triangle_mesh(os.path.join(args.out, f'realworld_bbox.ply'), cube)
    o3d.io.write_point_cloud(os.path.join(args.out, f'clen_tiepoints_local.ply'), pcd_clean)

    # calculate scale mat
    scale_mat = np.eye(4)

    cube_cal, bbox_cal = ysutils.util_megasurf.get_bbox_from_pcd(pcd_clean)
    scale = bbox_cal.get_extent().max() / 2
    meanvec = bbox_cal.get_center()
    x_mean, y_mean, z_mean = meanvec
    scale_mat[:3, :3] = scale_mat[:3, :3] * scale
    scale_mat[:3, 3] = meanvec

    cube_cal.vertices = o3d.utility.Vector3dVector((cube_cal.vertices - meanvec) /scale)

    pcd_clean.points = o3d.utility.Vector3dVector((pcd_clean.points - meanvec) /scale)
    o3d.io.write_triangle_mesh(os.path.join(args.out, "normalized_bbox.ply"),cube_cal)
    o3d.io.write_point_cloud(os.path.join(args.out, "normalized_points.ply"),pcd_clean)


    #
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(cube)
    mesh_id = scene.add_triangles(mesh)
    os.makedirs(os.path.join(args.out, 'block_mask'), exist_ok=True)

    d_min_max = []
    masks_savepath = []
    for i in tqdm.tqdm(range(len(meta))):
        path_img = meta[i][1]
        basename = os.path.basename(path_img).split('.')[0]

        # import matplotlib.pyplot as plt
        # img = cv2.imread(path_img)
        # plt.imshow(img)
        # plt.show()
        # undstort = cv2.undistort(img,intrin, distCoeffs = np.asarray([distort[i][0], distort[i][1], distort[i][3], distort[i][4], distort[i][2]]))


        h, w = int(int(meta[i][2]) ), int(int(meta[i][3]) )
        pose = poses[i]
        K = intrins[i]
        mask_savepath = os.path.join(args.out,'block_mask', f'blockmask_{basename}.jpg')
        masks_savepath.append(mask_savepath)
        GR = ysutils.util_open3d.Gen_ray(H=h, W=w, K=K)
        original_ray = GR.get_principal_v(K=K, pose=pose)
        rays = GR.gen_rays_at(pose)
        rays = rays.astype(np.float32)
        rays_v = rays[:, 3:6]
        adj_cos = np.dot(rays_v, original_ray[0, 3:6])
        rays_o3d = o3d.core.Tensor(rays,
                                   dtype=o3d.core.Dtype.Float32)
        # ans = scene.cast_rays(rays_o3d)
        # norm = ans['primitive_normals'].numpy()
        # norm = norm.reshape(h, w, 3)
        # t_img = ans['t_hit'].numpy()
        # depth = (t_img * adj_cos).reshape(h, w)
        t_img = scene.cast_rays(rays_o3d)['t_hit'].numpy()
        depth = (t_img * adj_cos).reshape(h, w)
        valid_depth = np.logical_not(np.ma.masked_invalid(depth).mask)
        d_min_max.append((depth[valid_depth].min(),depth[valid_depth].max()))
        mask = np.logical_not(np.logical_or(np.isinf(depth), np.isnan(depth))) * 255
        cv2.imwrite(mask_savepath, mask.astype(np.uint8))



    # output cameras
    os.makedirs(os.path.join(args.out,'cams'),exist_ok=True)
    for i in range(len(nvm_filenames)):
        imgname,intrin,pose = nvm_filenames[i], intrins[i], poses[i]
        extrin = np.linalg.pinv(pose)
        depthstat = np.zeros((4,),dtype=float)
        depthstat[0] = d_min_max[i][0] * 0.5 # near
        depthstat[3] = d_min_max[i][1] * 2 # far = min(median * 4, far)
        depthstat[1] = (depthstat[3] - depthstat[0])/192
        depthstat[2] = 192
        write_cam(os.path.join(args.out,'cams',f'{imgname.split(".")[0]}_cam.txt'),intrin,extrin,depthstat) # file, intrinsic, extrinsic, depth_stat
        meta[i].append(os.path.join(args.out,'cams',f'{imgname.split(".")[0]}_cam.txt'))
    print('output cameras')

    np.savetxt(os.path.join(args.out,f'meta.txt'),meta,fmt='%s')

    # cal matching score
    Fmat = generate_img_tiepoints(n_cameras, ties_img_id)
    np.save(os.path.join(args.out,'Fmat.npy'),Fmat)
    matching = generate_pair_Fmat(Fmat)
    np.save(os.path.join(args.out,'matching.npy'),matching)

    block_img_ties = Fmat.diagonal()
    valid_img = block_img_ties >= 1

    block_meta_rec = {}
    block_img_collect = []
    poses = []
    n_valid_img = valid_img.sum()
    for i in tqdm.tqdm(range(len(valid_img))):
        if valid_img[i] == False:
            continue
        path_img = meta[i][1]
        path_cam = meta[i][4]
        basename = os.path.basename(path_img).split('.')[0]
        block_img_collect.append(os.path.basename(path_img))
        cam = ysutils.util_mvsnet.load_cam(path_cam)
        cam[1][:2, :] = cam[1][:2, :]
        h, w = int(int(meta[i][2])), int(int(meta[i][3]) )
        pose = np.linalg.pinv(cam[0])
        poses.append(pose)
        mask_savepath = masks_savepath[i]

        img_rec = {
            'image_path': path_img,
            'mask_path': mask_savepath,
            'intrinsic': cam[1][:3, :3],
            'extrinsic': cam[0],
            'pose': pose,
            'h': h,
            'w': w
        }
        block_meta_rec[basename] = img_rec
    with open(os.path.join(args.out,'block_imgid.txt'),'w') as f:
        for imgid in block_img_collect:
            f.write(imgid + '\n')
    # generate json
    h = block_meta_rec[list(block_meta_rec.keys())[0]]['h']
    w = block_meta_rec[list(block_meta_rec.keys())[0]]['w']
    fix_size_flag = True
    for rec in list(block_meta_rec):
        if block_meta_rec[rec]['h'] != h or block_meta_rec[rec]['w'] != w:
            fix_size_flag = False
            break

    out = {
        "coord_sys":  "OPENCV",
        "is_fisheye": False,
        "world_to_gt": scale_mat.tolist(),
        "n_frames": int(n_valid_img),
    }
    if fix_size_flag:
        out.update({
            "w": int(w),
            "h": int(h),
        })


    frames = []
    for idx, rec in enumerate(block_meta_rec):
        intrin = np.eye(4)
        intrin[:3, :3] = block_meta_rec[rec]['intrinsic']
        extrin = block_meta_rec[rec]['extrinsic']
        proj_mat = intrin @ extrin

        # scale and decompose
        P = proj_mat @ scale_mat
        P = P[:3, :4]
        intrinsic_param, c2w = ysutils.util_neuralangelo.load_K_Rt_from_P(None, P)
        c2w_gl = ysutils.util_neuralangelo._cv_to_gl(c2w)
        angle_x = math.atan(w / (intrinsic_param[0][0] * 2)) * 2
        angle_y = math.atan(h / (intrinsic_param[1][1] * 2)) * 2
        frame = {"file_path": os.path.normpath(block_meta_rec[rec]['image_path']),
                 "mask_path": os.path.normpath(block_meta_rec[rec]['mask_path']),
                 "transform_matrix": c2w.tolist(),
                 "fl_x": intrinsic_param[0][0],
                 "fl_y": intrinsic_param[1][1],
                 "cx": intrinsic_param[0][2],
                 "cy": intrinsic_param[1][2],
                 "camera_angle_x": angle_x,
                 "camera_angle_y": angle_y,
                 }

        frames.append(frame)
    out.update({
        "frames": frames
    })

    file_path = os.path.join(args.out, 'transforms.json')
    with open(file_path, "w") as outputfile:
        json.dump(out, outputfile, indent=2)



if __name__ == '__main__':
    pass