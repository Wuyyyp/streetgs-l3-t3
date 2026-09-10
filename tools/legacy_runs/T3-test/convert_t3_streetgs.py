#!/usr/bin/env python3
"""Convert one restored T3 clip/pose variant with the existing M18 converter."""
import argparse
import copy
import csv
import itertools
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from convert_restored_l3_streetgs import NumpyCompatUnpickler, converter

CAMERAS = {'left_front_camera': 0, 'right_front_camera': 1, 'rear_camera': 2,
           'left_rear_camera': 3, 'right_rear_camera': 4,
           'center_camera_fov30': 9, 'center_camera_fov120': 10}
CLASSES = {'VEHICLE_CAR': 'Car', 'VEHICLE_SUV': 'Suv', 'VEHICLE_TRUCK': 'Truck',
           'VEHICLE_TRUCK_SMALL': 'Truck', 'VEHICLE_PICKUP': 'Truck',
           'VEHICLE_BUS': 'Bus', 'VEHICLE_TRIKE': 'Tricycle',
           'BIKE_BICYCLE': 'Bicycle', 'PEDESTRIAN': 'Pedestrian',
           'CONE': 'Cone', 'POLE': 'Bollards', 'STONE POLE': 'Bollards',
           'ISOLATION_BARRER': 'misc'}
SIZE = (1600, 900)
SIGNS = np.array(list(itertools.product([-1., 1.], repeat=3)))
EDGES = [(i, j) for i in range(8) for j in range(i+1, 8)
         if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1]


def projected_box(box, intrinsic, ego_to_camera):
    """Bounding rectangle of a 3D box clipped against the near plane/image."""
    x, y, z, length, width, height, yaw = box
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = (SIGNS * [length/2, width/2, height/2]) @ rotation.T + [x, y, z]
    corners = corners @ ego_to_camera[:3, :3].T + ego_to_camera[:3, 3]
    near = 0.1
    visible = list(corners[corners[:, 2] >= near])
    for i, j in EDGES:
        a, b = corners[i], corners[j]
        if (a[2] < near) != (b[2] < near):
            visible.append(a + (b-a) * ((near-a[2]) / (b[2]-a[2])))
    if not visible:
        return None
    pixels = np.asarray(visible) @ intrinsic.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    lo = np.maximum(pixels.min(axis=0), [0, 0])
    hi = np.minimum(pixels.max(axis=0), SIZE)
    if np.any(hi <= lo):
        return None
    return [*((lo+hi)/2).tolist(), *(hi-lo).tolist()]


def image_path(raw, name, camera):
    timestamp_us = int(Path(camera['data_path']).stem)
    assert timestamp_us == int(camera['timestamp'])
    return raw / 'camera' / name / (str(timestamp_us * 1000) + '.jpg')


def convert_frame(task):
    index, original, poses, restored, raw, output, shared_images, variant = task
    cv2.setNumThreads(1)
    legacy = converter()
    info = copy.deepcopy(original)
    field = 'ego2global_transformation_matrix_camera_reference_offset'
    if variant == 'after':
        field += '_optimized'
    ego = np.asarray(info[field], dtype=np.float64)
    info['ego2global_transformation_matrix'] = ego
    info['timestamp'] = int(round(float(info['dynamic_compensation_reference_time_s']) * 1e9))
    perception = info['sensors']['lidar']['perception']
    assert perception['point_coordinate_frame'] == 'ego' and perception['dynamic_compensated']
    perception['aws_path'] = str(restored / perception['aws_path'])
    info['sensors']['lidar'] = {'perception': perception}
    for name, cam_id in CAMERAS.items():
        cam = info['sensors']['cams'][name]
        assert cam.get('images_undistorted') is True, name
        stem = f'{index:06d}_{cam_id:02d}'
        if shared_images:
            (output / 'images' / (stem+'.jpg')).symlink_to(
                os.path.relpath(shared_images / (stem+'.jpg'), output / 'images'))
            # Source dimensions verified on the before conversion, same input calibration.
            source_size = (3840, 2160) if cam_id in (9, 10) else (1920, 1280)
        else:
            raw_image = cv2.imread(str(image_path(raw, name, cam)))
            if raw_image is None:
                raise ValueError(f'Cannot decode frame {index} camera {name}')
            source_size = raw_image.shape[1], raw_image.shape[0]
            expected = (3840, 2160) if cam_id in (9, 10) else (1920, 1280)
            assert source_size == expected, (name, source_size)
            image = cv2.resize(raw_image, SIZE, interpolation=cv2.INTER_AREA)
            assert cv2.imwrite(str(output / 'images' / (stem+'.jpg')), image)
        K = np.asarray(cam['cam_intrinsic'], dtype=np.float64).copy()
        K[0] *= SIZE[0] / source_size[0]
        K[1] *= SIZE[1] / source_size[1]
        cam['cam_intrinsic'] = K
        cam['extrinsic'] = np.linalg.inv(poses[name]) @ ego
        np.testing.assert_allclose(ego @ np.linalg.inv(cam['extrinsic']), poses[name], atol=1e-8)
    objects = []
    assert len(set(info['track_id'])) == len(info['track_id'])
    for j, box in enumerate(info['gt_boxes']):
        box = np.asarray(box, dtype=float)
        assert box.shape == (7,) and np.isfinite(box).all() and (box[3:6] > 0).all()
        obj = dict(id=int(info['track_id'][j]), type=CLASSES[info['gt_names'][j]],
                   size=box[3:6], location=box[:3], rotation=[0, 0, box[6]],
                   velocity=info['vels'][j], info2d=[])
        assert np.isfinite(obj['velocity']).all()
        for name, cam in info['sensors']['cams'].items():
            bbox = projected_box(box, cam['cam_intrinsic'], cam['extrinsic'])
            if bbox is not None:
                obj['info2d'].append(dict(camera=name, bbox=bbox))
        objects.append(obj)
    info['objects'] = objects
    result = legacy.process_frame_chunk([(index, info)], str(output), str(restored),
        ['timestamp', 'lidar', 'dynamic'], [SIZE[1]]*11, [SIZE[0]]*11, [1]*11, False, index)
    if result['errors']:
        raise RuntimeError('\n'.join(result['errors']))
    for name, cam_id in CAMERAS.items():
        stem = f'{index:06d}_{cam_id:02d}'
        # Legacy conversion omits camera poses when this camera has no LiDAR returns.
        np.savetxt(output/'ego_pose'/(stem+'.txt'), poses[name])
        np.testing.assert_allclose(np.loadtxt(output/'ego_pose'/(stem+'.txt')), poses[name], atol=1e-8)
        if not (output/'lidar_depth'/(stem+'.npy')).exists():
            np.save(output/'lidar_depth'/(stem+'.npy'),
                    dict(mask=np.zeros(SIZE[::-1], dtype=bool), value=np.array([], dtype=np.float32)))
            continue
        depth = np.load(output/'lidar_depth'/(stem+'.npy'), allow_pickle=True).item()
        assert depth['mask'].shape == SIZE[::-1]
        assert depth['mask'].sum() == len(depth['value'])
        assert np.isfinite(depth['value']).all()
        assert (depth['value'] > 0).all()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--restored-root', type=Path, required=True)
    ap.add_argument('--raw-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--variant', choices=['before', 'after'], required=True)
    ap.add_argument('--shared-images', type=Path)
    ap.add_argument('--frame-count', type=int, default=101)
    ap.add_argument('--frame-start', type=int, default=0)
    ap.add_argument('--num-workers', type=int, default=4)
    args = ap.parse_args()
    with (args.restored_root/'result_interpolation_optimized.pkl').open('rb') as f:
        infos = NumpyCompatUnpickler(f).load()['infos']
    assert args.frame_start >= 0 and 2 <= args.frame_count <= len(infos)-args.frame_start
    infos = infos[args.frame_start:args.frame_start+args.frame_count]
    csv_path = args.restored_root/f'image_extrinsics/image_camera_to_world_{args.variant}.csv'
    poses = [{} for _ in infos]
    for row in csv.DictReader(csv_path.open()):
        i, name = int(row['frame'])-args.frame_start, row['camera']
        if i < 0 or i >= len(infos) or name not in CAMERAS:
            continue
        assert name not in poses[i]
        cam = infos[i]['sensors']['cams'][name]
        assert row['image_filename'] == Path(cam['data_path']).name
        assert int(row['timestamp_ns']) == int(cam['timestamp'])*1000
        T = np.array([[float(row[f'T_cw_{r}{c}']) for c in range(4)] for r in range(4)])
        assert np.isfinite(T).all()
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-8)
        # Delivery CSV uses rounded calibration matrices; preserve their exact values.
        np.testing.assert_allclose(T[:3,:3].T@T[:3,:3], np.eye(3), atol=1e-4)
        # CSV poses are evaluated at each image timestamp, not the common frame time.
        poses[i][name] = T
    for i, info in enumerate(infos):
        assert set(poses[i]) == set(CAMERAS)
        for name, cam in info['sensors']['cams'].items():
            assert image_path(args.raw_root, name, cam).is_file()
        assert (args.restored_root/info['sensors']['lidar']['perception']['aws_path']).is_file()
        assert set(info['gt_names']) <= CLASSES.keys()
    if args.shared_images:
        assert (args.shared_images.parent/'conversion.complete.json').is_file()
    args.output.mkdir(parents=True, exist_ok=False)
    for folder in ['images', 'intrinsics', 'extrinsics', 'ego_pose', 'lidar_depth', 'intensity', 'dynamic_mask', 'track']:
        (args.output/folder).mkdir()
    tasks = [(i, info, poses[i], args.restored_root, args.raw_root, args.output,
              args.shared_images, args.variant) for i, info in enumerate(infos)]
    results = []
    with Pool(args.num_workers) as pool:
        for result in pool.imap_unordered(convert_frame, tasks):
            results.append(result)
            print(f'CONVERT_PROGRESS {len(results)}/{len(infos)}', flush=True)
    results.sort(key=lambda r: r['worker_id'])
    legacy = converter()
    (args.output/'timestamps.json').write_text(json.dumps(legacy.merge_timestamps_streetgs(results)))
    (args.output/'timestamps_specific.json').write_text(json.dumps([x for r in results for x in r['timestamps']]))
    legacy.merge_pointcloud_npz(str(args.output), len(infos))
    (args.output/'track/track_info.txt').write_text('frame_id track_id class_name alpha height width length x y z heading speed\n' +
        ''.join(' '.join(line)+'\n' for r in results for line in r['track_info_lines']))
    visibility = {}
    for r in results:
        for track, frames in r['track_camera_vis'].items():
            visibility.setdefault(track, {}).update(frames)
    (args.output/'track/track_camera_vis.json').write_text(json.dumps(visibility))
    with np.load(args.output/'pointcloud.npz', allow_pickle=True) as f:
        points, projections = f['pointcloud'].item(), f['camera_projection'].item()
        assert set(points) == set(projections) == set(range(len(infos)))
        assert all(len(points[i]) == len(projections[i]) > 0 for i in points)
    manifest = dict(source=str(args.restored_root), raw_source=str(args.raw_root), variant=args.variant,
        camera_pose_csv=str(csv_path), cameras=CAMERAS, image_size=SIZE,
        source_frame_range=[args.frame_start, args.frame_start+len(infos)-1],
        converted_frame_range=[0, len(infos)-1], image_count=len(infos)*7,
        images='Source images_undistorted=True; resize only, scale K on both axes',
        pointcloud='Shared compensated_perception in ego coordinates; variant-specific projection',
        tracks='gt_boxes/track_id/vels; near-plane-clipped 3D boxes projected for camera visibility/masks',
        class_mapping=CLASSES)
    (args.output/'conversion.complete.json').write_text(json.dumps(manifest, indent=2))
    print('CONVERT_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
