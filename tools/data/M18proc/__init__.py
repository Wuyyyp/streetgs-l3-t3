import numpy as np

cameras_type = {'left_front_camera':0,'right_front_camera':1,'rear_camera':2,\
                'left_rear_camera':3,'right_rear_camera':4,'front_camera_fov200':5,\
                'rear_camera_fov200':6,'left_camera_fov200':7,'right_camera_fov200':8,\
                'center_camera_fov30':9, 'center_camera_fov120':10}

lidar_type = {'perception':0,'front_lidar':1,'rear_lidar':2,\
                'left_lidar':3,'right_lidar':4,'top_center_lidar':5,\
                'front_left_lidar':6,'front_right_lidar':7}

additonal_extrinsics_mapper = {
    'front-camera-fov30.json':'center_camera_fov30', 'left-camera-fov200.json':'left_camera_fov200', 'front-camera-fov200.json':'front_camera_fov200', 
    'rear-camera.json':'rear_camera', 'right-camera-fov200.json':'right_camera_fov200', 'left-front-camera.json':'left_front_camera', 
    'right-rear-camera.json':'right_rear_camera', 'front-camera-fov120.json':'center_camera_fov120', 'rear-camera-fov200.json':'rear_camera_fov200', 
    'right-front-camera.json':'right_front_camera', 'left-rear-camera.json':'left_rear_camera',
    'rear-lidar.json':'rear_lidar', 'front-lidar.json':'front_lidar', 'top-right-lidar.json':'front_right_lidar', 
    'left-lidar.json':'left_lidar', 'top-center-lidar.json':'top_center_lidar', 'top-left-lidar.json':'front_left_lidar', 'right-lidar.json':'right_lidar'
}

gys_cameras_type = {'left_front_camera':0,'right_front_camera':1,'rear_camera':2,\
                'left_rear_camera':3,'right_rear_camera':4,'front_camera_fov200':5,\
                'rear_camera_fov200':6,'left_camera_fov200':7,'right_camera_fov200':8,\
                'front_camera_fov30':9, 'front_camera_fov120':10}

gys_lidar_type = {'perception':0,'front_lidar':1,'rear_lidar':2,\
                'left_lidar':3,'right_lidar':4,'top_center_lidar':5,\
                'top_left_lidar':6,'top_right_lidar':7}

gys_additonal_extrinsics_mapper = {
    'front-camera-fov30.json':'front_camera_fov30', 'left-camera-fov200.json':'left_camera_fov200', 'front-camera-fov200.json':'front_camera_fov200', 
    'rear-camera.json':'rear_camera', 'right-camera-fov200.json':'right_camera_fov200', 'left-front-camera.json':'left_front_camera', 
    'right-rear-camera.json':'right_rear_camera', 'front-camera-fov120.json':'front_camera_fov120', 'rear-camera-fov200.json':'rear_camera_fov200', 
    'right-front-camera.json':'right_front_camera', 'left-rear-camera.json':'left_rear_camera',
    'rear-lidar.json':'rear_lidar', 'front-lidar.json':'front_lidar', 'top-right-lidar.json':'top_right_lidar', 
    'left-lidar.json':'left_lidar', 'top-center-lidar.json':'top_center_lidar', 'top-left-lidar.json':'top_left_lidar', 'right-lidar.json':'right_lidar'
}



# left_lidar


# WAYMO [forward, left, up]
# ENU [east, north, up]
# OPENCV [right, down, forward]
# OPENGL [right, up, back]

# 保证translation不变的情况下对pose进行转换， pose_waymo @ POSE_WAYMO2CV = pose_cv
POSE_WAYMO2CV = np.array([[0., 0., 1., 0.],
                          [-1., 0., 0., 0.],
                          [0., -1., 0., 0.],
                          [0., 0., 0., 1.]])

# POSE_WAYMO2CV = np.array([[0., 0., 1., 0.],
#                           [-1., 0., 0., 0.],
#                           [0., -1., 0., 0.],
#                           [0., 0., 0., 1.]])

POSE_WAYMO2GL = np.array([[0., 0., 1., 0.],
                          [-1., 0., 0., 0.],
                          [0., -1., 0., 0.],
                          [0., 0., 0., 1.]])

POSE_ENU2CV = np.array([[0., 0., 1., 0.],
                          [-1., 0., 0., 0.],
                          [0., -1., 0., 0.],
                          [0., 0., 0., 1.]])

# def get_extrinsic(camera_calibration):
#     camera_extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4)  # camera to vehicle
#     extrinsic = np.matmul(camera_extrinsic, POSE_WAYMO2CV)  # [forward, left, up] to [right, down, forward]
#     return extrinsic


_camera2label = {'left_front_camera':0,'right_front_camera':1,'rear_camera':2,\
                'left_rear_camera':3,'right_rear_camera':4,'front_camera_fov200':5,\
                'rear_camera_fov200':6,'left_camera_fov200':7,'right_camera_fov200':8,\
                'center_camera_fov30':9, 'center_camera_fov120':10}


_label2camera = {v: k for k, v in _camera2label.items()}

image_heights = [1280, 1280, 1280, 1280, 1280, 1536, 1536, 1536, 1536, 2160, 2160]
image_widths = [1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 3840, 3840]
downsample_factors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
output_flag = [True, True, True, False, False, False, False, False, False, True, False]
image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
image_filename_to_frame = lambda x: int(x.split('.')[0][:6])

# bev2camera_type = {'left_front_camera':0,'right_front_camera':1,'rear_camera':2,\
#                 'left_rear_camera':3,'right_rear_camera':4,'front_camera_fov200':5,\
#                 'rear_camera_fov200':6,'left_camera_fov200':7,'right_camera_fov200':8,\
#                 'front_camera_fov30':9, 'front_camera_fov120':10}