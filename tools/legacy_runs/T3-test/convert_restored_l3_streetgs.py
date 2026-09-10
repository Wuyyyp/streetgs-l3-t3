#!/usr/bin/env python3
"""Adapt restored L3 delivery data to StreetGS using the existing M18 converter.

Run with /usr/bin/python on wyp-l3-worker-0. Never overwrites an output folder.
All seven pinhole cameras are undistorted to the delivery's 1600x900 K_resize.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import sys
from multiprocessing import Pool

import cv2
import numpy as np

CONVERTER_ROOT = '/data/mnt/yswang-wan22/code/data_proc_local'
POSE_FIELD = 'ego2global_transformation_matrix_camera_reference_offset_optimized'
CAMERAS = {'left_front_camera': 0, 'right_front_camera': 1, 'rear_camera': 2,
           'left_rear_camera': 3, 'right_rear_camera': 4,
           'front_camera_fov30': 9, 'front_camera_fov120': 10}
ALIASES = {'front_camera_fov30': 'center_camera_fov30',
           'front_camera_fov120': 'center_camera_fov120'}
SIZE = (1600, 900)


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module.replace('numpy._core', 'numpy.core'), name)


def converter():
    sys.path.insert(0, CONVERTER_ROOT)
    from M18proc import M18_converter_xirang_pkl_parallel_v2 as legacy
    legacy.PROC_CAMERA_IDX = list(CAMERAS.values())
    return legacy


def resolve_image(path, prefix):
    candidates = [Path(prefix) / str(path).lstrip('/'), Path(path)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(str(candidates))


def convert_frame(task):
    index, original, restored, output, prefix = task
    cv2.setNumThreads(1)
    legacy = converter()
    info = copy.deepcopy(original)
    output = Path(output)
    ego = np.asarray(info[POSE_FIELD], dtype=np.float64)
    if ego.shape != (4, 4) or not np.isfinite(ego).all():
        raise ValueError(f'Invalid optimized ego pose: frame {index}')
    info['ego2global_transformation_matrix'] = ego
    # Tracks and compensated points refer to the camera reference time.
    reference_s = float(info['dynamic_compensation_reference_time_s'])
    info['timestamp'] = int(round(reference_s * 1e9))
    perception = info['sensors']['lidar']['perception']
    pcd = Path(restored) / perception['aws_path']
    if not pcd.is_file() or not perception.get('dynamic_compensated'):
        raise ValueError(f'Missing compensated perception: {pcd}')
    perception['aws_path'] = str(pcd)
    info['sensors']['lidar'] = {'perception': perception}

    masks, cameras = {}, {}
    for name, camera_id in CAMERAS.items():
        camera = info['sensors']['cams'][name]
        raw_path = resolve_image(camera['aws_path'], prefix)
        raw = cv2.imread(str(raw_path))
        if raw is None:
            raise ValueError(f'Cannot decode {raw_path}')
        K = np.asarray(camera['cam_intrinsic'], dtype=np.float64)
        new_K = np.asarray(camera['cam_intrinsic_resize'], dtype=np.float64)
        distortion = np.asarray(camera['d'], dtype=np.float64)
        maps = cv2.initUndistortRectifyMap(K, distortion, None, new_K, SIZE, cv2.CV_32FC1)
        image = cv2.remap(raw, *maps, interpolation=cv2.INTER_LINEAR)
        stem = f'{index:06d}_{camera_id:02d}'
        if not cv2.imwrite(str(output / 'images' / (stem + '.jpg')), image):
            raise IOError(stem)

        # Rasterize annotations on the original image, then apply the exact
        # same undistortion to the mask; resizing the original boxes is wrong.
        raw_mask = np.zeros(raw.shape[:2], dtype=np.uint8)
        h, w = raw_mask.shape
        for obj in info['objects']:
            if np.linalg.norm(obj.get('velocity', [1., 1., 1.])) < legacy.VELOCITY_THRESHOLD:
                continue
            for box in obj.get('info2d', []):
                if box.get('camera') not in (name, ALIASES.get(name, name)):
                    continue
                x, y, bw, bh = box['bbox']
                x0, y0 = max(0, int(x-bw/2)), max(0, int(y-bh/2))
                x1, y1 = min(w-1, int(x+bw/2)), min(h-1, int(y+bh/2))
                if x1 >= x0 and y1 >= y0:
                    cv2.rectangle(raw_mask, (x0, y0), (x1, y1), 255, -1)
        masks[stem] = cv2.remap(raw_mask, *maps, interpolation=cv2.INTER_NEAREST)
        camera['cam_intrinsic'] = new_K
        cameras[ALIASES.get(name, name)] = camera

    info['sensors']['cams'] = cameras
    for obj in info['objects']:
        for box in obj.get('info2d', []):
            box['camera'] = ALIASES.get(box.get('camera'), box.get('camera'))
    result = legacy.process_frame_chunk(
        [(index, info)], str(output), str(restored), ['timestamp', 'lidar', 'dynamic'],
        [SIZE[1]] * 11, [SIZE[0]] * 11, [1] * 11, False, index)
    if result['errors']:
        raise RuntimeError('\n'.join(result['errors']))
    for stem, mask in masks.items():
        if not cv2.imwrite(str(output / 'dynamic_mask' / (stem + '.png')), mask):
            raise IOError(stem)
        for folder in ('intrinsics', 'extrinsics', 'ego_pose'):
            matrix = np.loadtxt(output / folder / (stem + '.txt'))
            if not np.isfinite(matrix).all():
                raise ValueError(f'Invalid {folder}/{stem}')
        camera_id = int(stem[-2:])
        name = next(name for name, value in CAMERAS.items() if value == camera_id)
        original_camera = original['sensors']['cams'][name]
        np.testing.assert_allclose(np.loadtxt(output / 'ego_pose' / (stem + '.txt')),
                                   ego @ np.linalg.inv(original_camera['extrinsic']), atol=1e-8)
        np.testing.assert_allclose(np.loadtxt(output / 'intrinsics' / (stem + '.txt')),
                                   original_camera['cam_intrinsic_resize'], atol=1e-8)
        depth = np.load(output / 'lidar_depth' / (stem + '.npy'), allow_pickle=True).item()
        if (depth['mask'].shape != SIZE[::-1] or depth['mask'].sum() != len(depth['value'])
                or len(depth['value']) == 0 or not np.isfinite(depth['value']).all()
                or not (depth['value'] > 0).all()):
            raise ValueError(f'Invalid lidar depth: {stem}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restored-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-prefix', default='/data')
    parser.add_argument('--frame-start', type=int, default=0)
    parser.add_argument('--frame-count', type=int, default=101)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    source = args.restored_root / 'result_interpolation_optimized.pkl'
    with source.open('rb') as handle:
        infos = NumpyCompatUnpickler(handle).load()['infos']
    end = args.frame_start + args.frame_count
    if args.frame_start < 0 or args.frame_count < 2 or end > len(infos) or args.num_workers < 1:
        raise ValueError(f'Invalid frame range/workers: [{args.frame_start}, {end}), total={len(infos)}')
    selected = infos[args.frame_start:end]
    # Fail before creating output if required sources are missing.
    for info in selected:
        if POSE_FIELD not in info:
            raise KeyError(POSE_FIELD)
        for name in CAMERAS:
            resolve_image(info['sensors']['cams'][name]['aws_path'], args.source_prefix)
        pcd = args.restored_root / info['sensors']['lidar']['perception']['aws_path']
        if not pcd.is_file():
            raise FileNotFoundError(pcd)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('images', 'intrinsics', 'extrinsics', 'ego_pose', 'lidar_depth',
                 'intensity', 'dynamic_mask', 'track'):
        (args.output / name).mkdir()
    tasks = [(i, info, str(args.restored_root), str(args.output), args.source_prefix)
             for i, info in enumerate(selected)]
    results = []
    with Pool(min(args.num_workers, len(tasks))) as pool:
        for result in pool.imap_unordered(convert_frame, tasks):
            results.append(result)
            print(f'CONVERT_PROGRESS {len(results)}/{len(tasks)}', flush=True)
    results.sort(key=lambda result: result['worker_id'])
    legacy = converter()
    timestamps = legacy.merge_timestamps_streetgs(results)
    (args.output / 'timestamps.json').write_text(json.dumps(timestamps, indent=2))
    specific = [item for result in results for item in result['timestamps']]
    (args.output / 'timestamps_specific.json').write_text(json.dumps(specific, indent=2))
    legacy.merge_pointcloud_npz(str(args.output), len(tasks))
    tracks = [line for result in results for line in result['track_info_lines']]
    header = 'frame_id track_id class_name alpha height width length x y z heading speed\n'
    (args.output / 'track' / 'track_info.txt').write_text(
        header + ''.join(' '.join(line) + '\n' for line in tracks))
    visibility = {}
    for result in results:
        for track, frames in result['track_camera_vis'].items():
            visibility.setdefault(track, {}).update(frames)
    (args.output / 'track' / 'track_camera_vis.json').write_text(json.dumps(visibility))
    with np.load(args.output / 'pointcloud.npz', allow_pickle=True) as cloud:
        points, projections = cloud['pointcloud'].item(), cloud['camera_projection'].item()
        if set(points) != set(range(len(tasks))) or set(projections) != set(points):
            raise ValueError('Pointcloud frame IDs do not match converted frames')
        for frame in points:
            if not len(points[frame]) or len(points[frame]) != len(projections[frame]):
                raise ValueError(f'Invalid pointcloud frame {frame}')
    manifest = {'source_pkl': str(source), 'pose_field': POSE_FIELD,
                'source_frame_range_inclusive': [args.frame_start, end - 1],
                'converted_frame_range_inclusive': [0, len(tasks) - 1],
                'cameras': CAMERAS, 'image_size': SIZE, 'image_count': len(tasks) * 7,
                'undistorted': True, 'pointcloud': 'compensated_perception (ego coordinates)',
                'camera_pose': 'optimized_ego_to_world @ inverse(pkl_ego_to_camera)',
                'track_time': 'dynamic_compensation_reference_time_s',
                'frame_mapping': [{'output_frame': i, 'source_frame': args.frame_start+i,
                                   'source_timestamp': int(info['timestamp'])}
                                  for i, info in enumerate(selected)]}
    (args.output / 'conversion.complete.json').write_text(json.dumps(manifest, indent=2))
    print(f'CONVERT_COMPLETE {args.output} images={len(tasks)*7}', flush=True)


if __name__ == '__main__':
    main()
