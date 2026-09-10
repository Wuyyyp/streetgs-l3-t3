import open3d as o3d
import numpy as np
import xmltodict
import os
import glob
import json
def get_distortion_params(
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
):
    """Returns a distortion parameters matrix.

    Args:
        k1: The first radial distortion parameter.
        k2: The second radial distortion parameter.
        k3: The third radial distortion parameter.
        k4: The fourth radial distortion parameter.
        p1: The first tangential distortion parameter.
        p2: The second tangential distortion parameter.
    Returns:
        torch.Tensor: A distortion parameters matrix.
    """
    return np.asarray([k1, k2, k3, k4, p1, p2])


def parse_json(json_path):
    with open(json_path, 'r') as f:
        meta = json.load(f)
    if "coord_sys" in meta:
        assert meta["coord_sys"] == "OPENCV", 'coordinate system not supported'
    image_filenames = []
    mask_filenames = []
    poses = []

    fx_fixed = "fl_x" in meta
    fy_fixed = "fl_y" in meta
    cx_fixed = "cx" in meta
    cy_fixed = "cy" in meta
    height_fixed = "h" in meta
    width_fixed = "w" in meta
    distort_fixed = False
    for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
        if distort_key in meta:
            distort_fixed = True
            break
    fx = []
    fy = []
    cx = []
    cy = []
    height = []
    width = []
    distort = []
    filepaths = []

    for frame in meta["frames"]:
        filepath = frame["file_path"]
        filepaths.append(filepath)
        if not fx_fixed:
            assert "fl_x" in frame, "fx not specified in frame"
            fx.append(float(frame["fl_x"]))
        if not fy_fixed:
            assert "fl_y" in frame, "fy not specified in frame"
            fy.append(float(frame["fl_y"]))
        if not cx_fixed:
            assert "cx" in frame, "cx not specified in frame"
            cx.append(float(frame["cx"]))
        if not cy_fixed:
            assert "cy" in frame, "cy not specified in frame"
            cy.append(float(frame["cy"]))
        if not height_fixed:
            assert "h" in frame, "height not specified in frame"
            height.append(int(frame["h"]))
        if not width_fixed:
            assert "w" in frame, "width not specified in frame"
            width.append(int(frame["w"]))
        if not distort_fixed:
            distort.append(
                get_distortion_params(
                    k1=float(meta["k1"]) if "k1" in meta else 0.0,
                    k2=float(meta["k2"]) if "k2" in meta else 0.0,
                    k3=float(meta["k3"]) if "k3" in meta else 0.0,
                    k4=float(meta["k4"]) if "k4" in meta else 0.0,
                    p1=float(meta["p1"]) if "p1" in meta else 0.0,
                    p2=float(meta["p2"]) if "p2" in meta else 0.0,
                )
            )

        image_filenames.append(os.path.basename(filepath))
        poses.append(np.array(frame["transform_matrix"]))
        if "mask_path" in frame:
            mask_filepath = frame["mask_path"]
            mask_filenames.append(os.path.basename(mask_filepath))

    fx_fixed = "fl_x" in meta
    fy_fixed = "fl_y" in meta
    cx_fixed = "cx" in meta
    cy_fixed = "cy" in meta
    height_fixed = "h" in meta
    width_fixed = "w" in meta
    if height_fixed:
        for i in range(len(image_filenames)):
            height.append(meta['h'])
            width.append(meta['w'])
    if fx_fixed:
        for i in range(len(image_filenames)):
            fx.append(meta['fx'])
            fy.append(meta['fy'])
            cx.append(meta['cx'])
            cy.append(meta['cy'])
    intrins = []
    extrins = []
    for i in range(len(image_filenames)):
        intrin = np.eye(3)
        intrin[0,0] = fx[i]
        intrin[1,1] = fy[i]
        intrin[0,2] = cx[i]
        intrin[1,2] = cy[i]
        intrins.append(intrin)
        extrins.append(np.linalg.pinv(poses[i]))
    return image_filenames, filepaths, height, width, intrins, extrins

def readcam_xml(camxml):
    with open(camxml,'r') as xml:
        content = xml.read()
    camdict = xmltodict.parse(content)
    assert camdict['BlocksExchange']['Block']['Photogroups'].keys().__len__() == 1,'More than 1 block. Not implemented.'
    photos_dict = camdict['BlocksExchange']['Block']['Photogroups']['Photogroup']
    W = int(photos_dict['ImageDimensions']['Width'])
    H = int(photos_dict['ImageDimensions']['Height'])
    CameraOrientation = photos_dict['CameraOrientation']
    FocalLengthPixels = float(photos_dict['FocalLengthPixels'])
    fx,fy = float(photos_dict['PrincipalPoint']['x']),float(photos_dict['PrincipalPoint']['y'])
    n_image = photos_dict['Photo'].__len__()
    intrindict = {}
    intrindict['H'] = H
    intrindict['W'] = W
    intrindict['CameraOrientation'] = CameraOrientation
    intrindict['FocalLengthPixels'] = FocalLengthPixels
    intrindict['fx'] = fx
    intrindict['fy'] = fy
    intrindict['n_image'] = n_image

    photos_rec_dict = {}
    for photo_dict in photos_dict['Photo']:

        photo_name = os.path.basename(photo_dict['ImagePath'])
        assert int(photo_dict['Component'])!=0,f'{photo_name} pose estimation has error!'

        rot_mat = np.zeros((3,3))
        rot_mat[0,0] = float(photo_dict['Pose']['Rotation']['M_00'])
        rot_mat[0,1] = float(photo_dict['Pose']['Rotation']['M_01'])
        rot_mat[0,2] = float(photo_dict['Pose']['Rotation']['M_02'])
        rot_mat[1,0] = float(photo_dict['Pose']['Rotation']['M_10'])
        rot_mat[1,1] = float(photo_dict['Pose']['Rotation']['M_11'])
        rot_mat[1,2] = float(photo_dict['Pose']['Rotation']['M_12'])
        rot_mat[2,0] = float(photo_dict['Pose']['Rotation']['M_20'])
        rot_mat[2,1] = float(photo_dict['Pose']['Rotation']['M_21'])
        rot_mat[2,2] = float(photo_dict['Pose']['Rotation']['M_22'])
        rot_mat = np.linalg.pinv(rot_mat)

        t_vec = np.zeros((3,))
        t_vec[0] = float(photo_dict['Pose']['Center']['x'])
        t_vec[1] = float(photo_dict['Pose']['Center']['y'])
        t_vec[2] = float(photo_dict['Pose']['Center']['z'])

        near_far = np.zeros((3,))
        near_far[0] = float(photo_dict['NearDepth'])
        near_far[1] = float(photo_dict['MedianDepth'])
        near_far[2] = float(photo_dict['FarDepth'])

        photo_rec_dict = {}
        photo_rec_dict['img_path'] = photo_dict['ImagePath']
        photo_rec_dict['pose_rot'] = rot_mat
        photo_rec_dict['pose_t'] = t_vec
        photo_rec_dict['near_far'] = near_far

        photos_rec_dict[photo_name] = photo_rec_dict

    return photos_rec_dict,intrindict

def merge_mesh(meshdir,suffix):
    folders = os.listdir(meshdir)
    meshs_f = []
    for folder in folders:
        meshs_f += glob.glob(os.path.join(meshdir,folder,f'*.{suffix}'))
    mesh = o3d.geometry.TriangleMesh()
    for mesh_f in meshs_f:
        mesh += o3d.io.read_triangle_mesh(mesh_f)
    return mesh
if __name__ == '__main__':

    pass