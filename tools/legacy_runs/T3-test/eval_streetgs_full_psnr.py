#!/usr/bin/env python3
"""Evaluate every original training view directly from a final StreetGS checkpoint."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--sample', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--gpu', type=int, required=True)
args = ap.parse_args()
sample, output = args.sample.resolve(), args.output.resolve()
assert (sample/'train.complete').exists()
assert not output.exists(), output
repo = Path('/data/l3_data_test/street_gaussians-main-local-v2')
os.chdir(repo)
os.environ.update(PWD=str(repo), CUDA_VISIBLE_DEVICES=str(args.gpu),
                  OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
sys.path.insert(0, str(repo))
sys.argv = ['full_psnr', '--config', str(sample/'train.yaml'), 'mode', 'evaluate',
            'loaded_iter', '25000', 'gpus', str([args.gpu]), 'data_device', 'cpu']

import torch
from lib.config import cfg
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.utils.general_utils import safe_state
from lib.utils.loss_utils import psnr

started = time.time()
safe_state(cfg.eval.quiet)
views = []
with torch.no_grad():
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    assert scene.loaded_iter == 25000
    renderer = StreetGaussianRenderer()
    cameras = sorted(scene.getTrainCameras(), key=lambda camera: camera.image_name)
    expected = {f'{frame:06d}_{cam:02d}' for frame in range(101) for cam in [0,1,2,3,4,9,10]}
    assert len(cameras) == 707 and {c.image_name for c in cameras} == expected
    assert {p.stem for p in (sample/'converted/images').glob('*.jpg')} == expected
    for index, camera in enumerate(cameras):
        rgb = renderer.render(camera, gaussians)['rgb'].clamp(0, 1)
        gt = camera.original_image.to('cuda').clamp(0, 1)
        assert rgb.shape == gt.shape == (3, 900, 1600)
        value = float(psnr(rgb, gt).item())
        assert math.isfinite(value), camera.image_name
        if index == 0:
            reference = -10 * torch.log10(torch.mean((rgb-gt)**2))
            assert abs(value-float(reference)) < 1e-4, 'Unexpected PSNR definition'
        views.append(dict(image=camera.image_name, camera=int(camera.image_name.split('_')[-1]), psnr=value))
        if (index+1) % 50 == 0 or index+1 == len(cameras):
            print(f'FULL_PSNR_PROGRESS {index+1}/{len(cameras)}', flush=True)
per_camera = {}
for camera in [0,1,2,3,4,9,10]:
    values = [v['psnr'] for v in views if v['camera'] == camera]
    assert len(values) == 101
    per_camera[str(camera)] = dict(image_count=len(values), mean_psnr=statistics.mean(values))
mean = statistics.mean(v['psnr'] for v in views)
assert abs(mean-statistics.mean(v['mean_psnr'] for v in per_camera.values())) < 1e-10
result = dict(sample=str(sample), iteration=25000, image_count=707,
              split='all original training views', mask='full frame, unmasked',
              aggregation='arithmetic mean of per-image PSNR in dB',
              rendering='float RGB, clamped to [0,1], original camera poses, no image encoding',
              mean_psnr=mean, per_camera=per_camera, per_view=views,
              completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              elapsed_seconds=time.time()-started)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('x') as f:
    json.dump(result, f, indent=2)
print('FULL_PSNR_COMPLETE', sample.name, sample.parent.name, mean, flush=True)
