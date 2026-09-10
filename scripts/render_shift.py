import sys
import os

# --- parse custom args before cfg import ---
_camera_types = None
_shift_meters = 3.0
_new_argv = []
_i = 0
_args = sys.argv
while _i < len(_args):
    if _args[_i] == '--cameras':
        _camera_types = []
        _i += 1
        while _i < len(_args) and not _args[_i].startswith('--'):
            _camera_types.append(int(_args[_i]))
            _i += 1
    elif _args[_i] == '--shift':
        _i += 1
        if _i < len(_args):
            _shift_meters = float(_args[_i])
            _i += 1
    else:
        _new_argv.append(_args[_i])
        _i += 1
sys.argv = _new_argv

import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.utils.camera_utils import Camera
from lib.utils.m18_utils import load_camera_info, _label2camera
from lib.config import cfg


def build_extra_cameras(datadir, selected_frames, cameras_build, all_camera_types):
    """Build Camera objects for camera types not in the original dataset."""
    import json
    from PIL import Image
    from lib.utils.graphics_utils import focal2fov

    start_frame, end_frame = selected_frames[0], selected_frames[1]
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info(datadir)
    image_dir = os.path.join(datadir, 'images')
    all_images = sorted(os.listdir(image_dir))

    timestamp_path = os.path.join(datadir, 'timestamps.json')
    with open(timestamp_path, 'r') as f:
        timestamps = json.load(f)

    extra_cameras = []
    for frame in range(start_frame, end_frame + 1):
        for cam in all_camera_types:
            if cam in cameras_build:
                continue  # already in the dataset

            # find image filename
            image_name = f'{frame:06d}_{cam:02d}'
            matches = [img for img in all_images if img.startswith(image_name)]
            if not matches:
                continue
            image_path = os.path.join(image_dir, matches[0])
            image = Image.open(image_path)

            cam_idx = frame * 11 + cam
            ixt = intrinsics[cam_idx]
            c2w = ego_cam_poses[cam_idx]
            pose = ego_frame_poses[frame]
            ext = extrinsics[cam_idx]

            width, height = image.size
            fx, fy = ixt[0, 0], ixt[1, 1]
            FovY = focal2fov(fx, height)
            FovX = focal2fov(fy, width)

            RT = np.linalg.inv(c2w)
            R = RT[:3, :3].T
            T = RT[:3, 3]
            K = ixt.copy()

            camera_name = _label2camera[cam]
            ts = timestamps[camera_name][f'{frame:06d}']
            timestamp_offset = 0  # doesn't matter for inference

            metadata = {
                'frame': frame,
                'cam': cam,
                'frame_idx': frame - start_frame,
                'ego_pose': pose,
                'extrinsic': ext,
                'timestamp': ts - timestamp_offset,
            }

            from lib.datasets.base_readers import CameraInfo
            from lib.utils.general_utils import PILtoTorch
            image_tensor = PILtoTorch(image, (width, height))

            cam_obj = Camera(
                id=len(extra_cameras),
                R=R, T=T,
                FoVx=FovX, FoVy=FovY, K=K,
                image=image_tensor,
                image_name=image_name,
                metadata=metadata,
            )
            extra_cameras.append(cam_obj)

    return extra_cameras


def revert_model_changes():
    """Revert changes made to model files."""
    pass  # we'll revert manually after running


def render_shifted(camera_types=None, shift_right_meters=3.0):
    # remember original training cameras
    train_cameras = list(cfg.data.cameras)

    # override for rendering
    cfg.mode = 'evaluate'
    cfg.eval.skip_train = False
    cfg.eval.skip_test = False
    cfg.render.save_image = True
    cfg.render.save_video = False

    if cfg.loaded_iter == -1:
        from lib.utils.system_utils import searchForMaxIteration
        cfg.loaded_iter = searchForMaxIteration(cfg.point_cloud_dir)

    print(f"Model path: {cfg.model_path}")
    print(f"Loaded iteration: {cfg.loaded_iter}")
    print(f"Training cameras: {train_cameras}")
    print(f"Render cameras: {camera_types}")
    print(f"Shift right: {shift_right_meters}m")

    with torch.no_grad():
        # Step 1: load data and model with ORIGINAL training cameras
        cfg.data.cameras = train_cameras
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        model_name = os.path.basename(cfg.model_path.rstrip('/'))
        cam_str = '_'.join(str(c) for c in camera_types)
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'test_shift_render',
            f"{model_name}_iter{scene.loaded_iter}_shift{shift_right_meters}m_cams{cam_str}"
        )
        os.makedirs(save_dir, exist_ok=True)
        print(f"Output: {save_dir}")

        # Step 2: build extra cameras for camera types not in training
        all_cameras = []
        if not cfg.eval.skip_train:
            all_cameras.extend(scene.getTrainCameras())
        if not cfg.eval.skip_test:
            all_cameras.extend(scene.getTestCameras())

        if camera_types is not None:
            extra_camera_types = [c for c in camera_types if c not in train_cameras]
            if extra_camera_types:
                print(f"Building extra cameras: {extra_camera_types}")
                extra_cameras = build_extra_cameras(
                    cfg.source_path,
                    cfg.data.selected_frames,
                    train_cameras,
                    extra_camera_types,
                )
                all_cameras.extend(extra_cameras)

        print(f"Total cameras to render: {len(all_cameras)}")

        for idx, camera in enumerate(tqdm(all_cameras, desc="Rendering shifted views")):
            c2w = camera.get_extrinsic()
            ego_pose = camera.ego_pose.cpu().numpy()

            c2w_in_ego = np.linalg.inv(ego_pose) @ c2w
            c2w_in_ego[:3, 3] += np.array([shift_right_meters, 0, 0])
            c2w_new = ego_pose @ c2w_in_ego

            camera.set_extrinsic(c2w_new)
            result = renderer.render(camera, gaussians)

            name = camera.image_name
            save_image(result['rgb'], os.path.join(save_dir, f'{name}_rgb_shift.png'))

            camera.set_extrinsic(c2w)

        print(f"Done. Results saved to {save_dir}")


if __name__ == "__main__":
    safe_state(True)
    render_shifted(camera_types=_camera_types, shift_right_meters=_shift_meters)
