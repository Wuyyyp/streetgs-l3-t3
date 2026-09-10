"""Render an unchanged StreetGS camera trajectory and compose matched comparison video."""
import sys
sys.path.insert(0, '/data/l3_data_test/street_gaussians-main-local-v2')
import os
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from lib.config import cfg
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state

out = Path(os.environ['COMPARISON_OUT'])
sense = Path(os.environ['SENSE_SAMPLE'])
variant = os.environ['STREETGS_VARIANT']
out.mkdir(parents=True, exist_ok=True)
images = out / 'streetgs_frames'
images.mkdir(exist_ok=True)
views = [(0, 'left_front_camera', 'Left front'),
         (10, 'center_camera_fov120', 'Front FOV120'),
         (1, 'right_front_camera', 'Right front')]
safe_state(cfg.eval.quiet)
with torch.no_grad():
    dataset = Dataset()
    model = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(model, dataset)
    renderer = StreetGaussianRenderer()
    cameras = scene.getTrainCameras()
    assert len(cameras) == 707
    selected = {}
    for camera in cameras:
        frame, cam = map(int, camera.image_name.split('_'))
        if cam in {0, 10, 1}:
            assert (frame, cam) not in selected
            selected[frame, cam] = camera
    assert set(selected) == {(f, c) for f in range(101) for c in [0, 10, 1]}
    for (frame, cam), camera in tqdm(sorted(selected.items()), desc='Render original views'):
        # No camera transform or Gaussian visibility filtering is applied.
        rgb = renderer.render(camera, model)['rgb'].clamp(0, 1)
        rgb = (rgb.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        rgb = cv2.resize(rgb, (960, 540), interpolation=cv2.INTER_AREA)
        assert cv2.imwrite(str(images / f'{frame:06d}_{cam:02d}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

timestamps = json.loads((Path(cfg.source_path) / 'timestamps_specific.json').read_text())
by_frame = {int(t['FRAME_IDX']): t for t in timestamps}
base = sense / '3dgs_format/v_lidar/street-gaussians-ns/2026-09-03_093737/renders/all/postprocess-rgb'
mapping = []
for frame in range(101):
    ts = str(by_frame[frame]['timestamp'])
    for cam, name, _ in views:
        assert (base / name / (ts + '.jpg')).is_file(), (frame, name, ts)
        mapping.append(dict(frame=frame, camera=cam, timestamp_ns=ts,
                            sensetime=str(base / name / (ts + '.jpg')),
                            streetgs=str(images / f'{frame:06d}_{cam:02d}.png')))

video = out / (sense.name[:10] + '_comparison.mp4')
command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', '2880x1160', '-r', '10', '-i', '-', '-an', '-c:v', 'libx264',
           '-threads', '4', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(video)]
encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
try:
    for frame in tqdm(range(101), desc='Compose comparison'):
        ts = str(by_frame[frame]['timestamp'])
        rows = []
        for method in ['SenseTime', 'StreetGS']:
            cells = []
            for cam, name, label in views:
                path = base / name / (ts + '.jpg') if method == 'SenseTime' else images / f'{frame:06d}_{cam:02d}.png'
                cell = cv2.imread(str(path))
                assert cell is not None, path
                cell = cv2.resize(cell, (960, 540), interpolation=cv2.INTER_AREA)
                if cam == 10:
                    cell[375:, :, :] = 0  # Same lower 30.56% hood mask for both methods.
                header = np.full((40, 960, 3), 24, dtype=np.uint8)
                tag = '70k' if method == 'SenseTime' else variant
                cv2.putText(header, f'{method} | {label} | {tag}', (16, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 1, cv2.LINE_AA)
                cells.append(np.vstack([header, cell]))
            rows.append(np.hstack(cells))
        canvas = np.vstack(rows)
        assert canvas.shape == (1160, 2880, 3)
        if frame in [0, 50, 100]:
            assert cv2.imwrite(str(out / f'preview_{frame:03d}.jpg'), canvas)
        encoder.stdin.write(canvas.tobytes())
finally:
    encoder.stdin.close()
assert encoder.wait() == 0
probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
    '-select_streams', 'v:0', '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate,duration',
    '-of', 'json', str(video)]))['streams'][0]
assert int(probe['nb_read_frames']) == 101 and probe['r_frame_rate'] == '10/1'
manifest = dict(streetgs_model=cfg.model_path, iteration=scene.loaded_iter,
                sensetime_sample=str(sense), variant=variant, shift_xyz=[0,0,0],
                hood_mask=dict(camera=10, first_masked_y=375, height=540),
                layout='SenseTime top; StreetGS bottom; left-front / front-FOV120 / right-front',
                probe=probe, frame_mapping=mapping)
(out / 'video_manifest.json').write_text(json.dumps(manifest, indent=2))
print('COMPLETE', video, probe, flush=True)
