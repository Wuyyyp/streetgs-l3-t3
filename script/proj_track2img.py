#!/usr/bin/env python3
"""
Project 3D track bounding boxes onto camera images.

Reads track_info.txt, projects each object's 3D box to 2D on all camera images,
and saves the annotated images.

Usage:
    python script/proj_track2img.py \
        --data_dir /home/yusen/Downloads/yswang_data3/clip_M18-2_07_20251227122737_DF \
        --output_dir /home/yusen/Downloads/yswang_data3/clip_M18-2_07_20251227122737_DF/track_render
"""

import os
import sys
import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from collections import defaultdict


# Per-category colors (BGR)
CATEGORY_COLORS = {
    'Car': (0, 0, 255),         # red
    'Pedestrian': (0, 255, 0),  # green
    'Cyclist': (255, 0, 0),     # blue
    'Motorcyclist': (255, 255, 0),
    'Bus': (0, 255, 255),
    'Truck': (255, 0, 255),
}


def get_box_corners(h, w, l, heading):
    """Get 8 corners of a 3D bounding box in local frame.

    Box dimensions: height (z), width (y), length (x).
    Heading: rotation around z (yaw).
    Returns (8, 3) corners.
    """
    dx = l / 2.0
    dy = w / 2.0
    dz = h / 2.0

    corners = np.array([
        [-dx, -dy, -dz],     # bottom
        [ dx, -dy, -dz],
        [ dx,  dy, -dz],
        [-dx,  dy, -dz],
        [-dx, -dy,  dz],      # top
        [ dx, -dy,  dz],
        [ dx,  dy,  dz],
        [-dx,  dy,  dz],
    ], dtype=np.float32)

    # Rotate around z (heading)
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    corners = corners @ R.T

    return corners


def project_to_image(corners_ego, ego2cam, K):
    """Project 3D corners from ego frame to image plane.

    Returns (8, 2) pixel coordinates or None if any point is behind camera.
    """
    # ego → camera
    pts_h = np.concatenate([corners_ego, np.ones((len(corners_ego), 1))], axis=1)
    pts_cam = (ego2cam @ pts_h.T)[:3, :].T  # (8, 3)

    if (pts_cam[:, 2] <= 0.01).any():
        return None

    # Camera → image
    pts_img = (K @ pts_cam.T).T  # (8, 3)
    uv = pts_img[:, :2] / pts_img[:, 2:3]
    return uv


def draw_box_2d(img, corners_2d, color, thickness=2):
    """Draw 3D bounding box edges on image."""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),   # top
        (0, 4), (1, 5), (2, 6), (3, 7),   # vertical
    ]
    pts = corners_2d.astype(np.int32)
    for i, j in edges:
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, thickness)
    return img


def draw_box_filled(img, corners_2d, color, alpha=0.3):
    """Draw semi-transparent filled 3D box faces."""
    overlay = img.copy()
    faces = [
        [0, 1, 2, 3],   # bottom
        [4, 5, 6, 7],   # top
        [0, 1, 5, 4],   # front
        [2, 3, 7, 6],   # back
        [1, 2, 6, 5],   # right
        [0, 3, 7, 4],   # left
    ]
    pts = corners_2d.astype(np.int32)
    for face in faces:
        cv2.fillPoly(overlay, [pts[face]], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def main():
    parser = argparse.ArgumentParser(
        description='Project 3D track boxes onto camera images.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Dataset root (contains track/, images/, ego_pose/, intrinsics/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for rendered images')
    parser.add_argument('--cameras', nargs='+', default=['00', '01', '02', '03', '04', '09', '10'],
                        help='Camera suffixes to render')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Max frames to render (0 = all)')
    parser.add_argument('--fill', action='store_true',
                        help='Draw filled semi-transparent faces')
    parser.add_argument('--filter_dynamic', action='store_true',
                        help='Only draw objects with speed > threshold')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load track data: frame_id → list of objects
    track_path = os.path.join(args.data_dir, 'track', 'track_info.txt')
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"track_info.txt not found in {args.data_dir}")

    frame_objects = defaultdict(list)
    with open(track_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            obj = {
                'frame_id': int(parts[0]),
                'track_id': int(parts[1]),
                'class': parts[2],
                'alpha': float(parts[3]),
                'h': float(parts[4]),
                'w': float(parts[5]),
                'l': float(parts[6]),
                'cx': float(parts[7]),
                'cy': float(parts[8]),
                'cz': float(parts[9]),
                'heading': float(parts[10]),
                'speed': float(parts[11]),
            }
            frame_objects[obj['frame_id']].append(obj)

    frames = sorted(frame_objects.keys())
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    print(f"Loaded {sum(len(v) for v in frame_objects.values())} objects across {len(frames)} frames")

    # Load camera intrinsics (static per camera)
    K_dict = {}
    for cam in args.cameras:
        intrin_files = sorted(glob(os.path.join(args.data_dir, 'intrinsics', f'*_{cam}.txt')))
        if intrin_files:
            K_dict[cam] = np.loadtxt(intrin_files[0])

    # Process each frame
    for frame_id in tqdm(frames, desc="Rendering frames"):
        frame_str = f'{frame_id:06d}'

        # Load ego pose (ego→world)
        ego_path = os.path.join(args.data_dir, 'ego_pose', f'{frame_str}.txt')
        if not os.path.exists(ego_path):
            continue
        ego2world = np.loadtxt(ego_path)

        objs = frame_objects[frame_id]
        if args.filter_dynamic:
            objs = [o for o in objs if o['speed'] > 0.5]

        if not objs:
            continue

        for cam in args.cameras:
            cam_pose_path = os.path.join(args.data_dir, 'ego_pose', f'{frame_str}_{cam}.txt')
            if not os.path.exists(cam_pose_path):
                continue
            cam2world = np.loadtxt(cam_pose_path)

            img_path = os.path.join(args.data_dir, 'images', f'{frame_str}_{cam}.jpg')
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue

            # ego → camera
            world2cam = np.linalg.inv(cam2world)
            ego2cam = world2cam @ ego2world
            K = K_dict.get(cam)
            if K is None:
                continue

            for obj in objs:
                # Build 3D box corners in ego frame
                corners_local = get_box_corners(obj['h'], obj['w'], obj['l'], obj['heading'])
                corners_ego = corners_local + np.array([obj['cx'], obj['cy'], obj['cz']])

                uv = project_to_image(corners_ego, ego2cam, K)
                if uv is None:
                    continue

                color = CATEGORY_COLORS.get(obj['class'], (255, 255, 255))

                if args.fill:
                    img = draw_box_filled(img, uv, color)
                img = draw_box_2d(img, uv, color)

            out_path = os.path.join(args.output_dir, f'{frame_str}_{cam}.jpg')
            cv2.imwrite(out_path, img)

    print(f"Done → {args.output_dir}")


if __name__ == '__main__':
    main()
