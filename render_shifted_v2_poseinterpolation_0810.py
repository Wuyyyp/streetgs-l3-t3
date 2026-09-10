#!/usr/bin/env python3
"""
Shift ego poses and render novel trajectory images with pose interpolation.

Extends render_shifted_v2_0728.py with:
  --interp_ratio R : smooth shift ramping (default 0.0 = disabled).
                      e.g. 0.2 = first 10% frames ramp up (0→full shift),
                      last 10% ramp down (full→0), middle 80% at full shift.
  --frame_range S E : only save output for frames in [S, E] (inclusive).
                      Interpolation curve is computed within this range.
                      When omitted, uses all dataset frames.
# /mnt/wyp/l3_data_test/L3-reconstruction/reconstruction_300_segments/sample_002_clip_M18-2_07_20251202093910_DF_seg000_f0_100_left/streetgs_train_L3_front/configs/config_000000.yaml

Example:
    python render_shifted_v2_poseinterpolation_0728.py \
        --source_path /path/to/dataset \
        --model_path /path/to/model \
        --output_dir /path/to/output \
        --shift shift_right \
        --interp_ratio 0.2 \
        --frame_range 0 99 \
        --max_pixels 100
"""

import os
import sys
import argparse
import gc
import re
import cv2
import numpy as np
import torch
from collections import OrderedDict
from glob import glob
from tqdm import tqdm


# ==============================================================================
# Shift definitions
# Coordinate convention (derived from actual vehicle motion in this dataset):
#   x — right   (lateral)
#   y — forward (longitudinal)
#   z — up      (vertical)
# ==============================================================================
SHIFT_TYPES = {
    'shift_right':   [ 3.0, 0.0, 0.0],
    'shift_left':    [-3.0, 0.0, 0.0],
    'shift_up':      [ 0.0, 0.0, 2.0],
    'shift_forward': [ 0.0, 3.0, 0.0],
    'shift_right_01':   [ 1.0, 0.0, 0.0],
    'shift_right_02':   [ 2.0, 0.0, 0.0],
    'shift_left_01':    [-1.0, 0.0, 0.0],
    'shift_left_02':    [-2.0, 0.0, 0.0],

}


# ==============================================================================
# Interpolation weight computation
# ==============================================================================
def compute_interp_weights(min_fid, max_fid, ratio):
    """Compute per-frame interpolation weights for smooth shift ramping.

    Args:
        min_fid: minimum frame ID (inclusive)
        max_fid: maximum frame ID (inclusive)
        ratio: float in [0.0, 1.0]; 0.0 = constant weight=1.0 for all

    Returns:
        dict mapping frame_id -> float weight in [0.0, 1.0]
    """
    N = max_fid - min_fid + 1
    if ratio <= 0.0 or N <= 1:
        return {fid: 1.0 for fid in range(min_fid, max_fid + 1)}

    ramp = max(int(N * ratio / 2), 1)

    weights = {}
    for fid in range(min_fid, max_fid + 1):
        pos = fid - min_fid
        if pos < ramp:
            w = pos / ramp
        elif pos >= N - ramp:
            w = (N - 1 - pos) / ramp
        else:
            w = 1.0
        weights[fid] = w

    return weights


# ==============================================================================
# LiDAR topology surface visibility
# ==============================================================================

_TOPOLOGY_HORIZONTAL_ELEVATION_GATE_DEG = 0.03
_TOPOLOGY_RELATIVE_RANGE_JUMP = 0.02
_TOPOLOGY_SPACING_RATIO = 6.0


def _nearest_sorted_indices(source, target):
    positions = np.searchsorted(target, source)
    right = np.clip(positions, 0, len(target) - 1)
    left = np.clip(positions - 1, 0, len(target) - 1)
    return np.where(np.abs(source - target[left]) <=
                    np.abs(source - target[right]), left, right)


def _topology_edge_valid(a, b, points, ranges, unit):
    min_range = np.minimum(ranges[a], ranges[b])
    relative_jump = np.abs(ranges[a] - ranges[b]) / np.maximum(min_range, 1e-8)
    angle = np.arccos(np.clip(np.sum(unit[a] * unit[b], axis=1), -1, 1))
    distance = np.linalg.norm(points[a] - points[b], axis=1)
    spacing_ratio = distance / np.maximum(min_range * angle, 1e-8)
    return ((relative_jump <= _TOPOLOGY_RELATIVE_RANGE_JUMP) &
            (spacing_ratio <= _TOPOLOGY_SPACING_RATIO))


def build_front_lidar_topology_triangles(raw_points, timestamps):
    """Build conservative triangles from adjacent front-LiDAR scan samples."""
    ranges = np.linalg.norm(raw_points, axis=1)
    unit = raw_points / np.maximum(ranges[:, None], 1e-8)
    elevation = np.degrees(np.arctan2(
        raw_points[:, 2], np.hypot(raw_points[:, 0], raw_points[:, 1])))
    _, starts, counts = np.unique(timestamps, return_index=True, return_counts=True)
    triangles = []
    for column in range(len(starts) - 1):
        ia = np.arange(starts[column], starts[column] + counts[column])
        ib = np.arange(starts[column + 1], starts[column + 1] + counts[column + 1])
        ea, eb = elevation[ia], elevation[ib]
        ab = _nearest_sorted_indices(ea, eb)
        ba = _nearest_sorted_indices(eb, ea)
        mutual = ba[ab] == np.arange(len(ia))
        horizontal = mutual & (
            np.abs(ea - eb[ab]) <= _TOPOLOGY_HORIZONTAL_ELEVATION_GATE_DEG)
        rows = np.where(horizontal)[0]
        ha, hb = ia[rows], ib[ab[rows]]
        valid_h = _topology_edge_valid(ha, hb, raw_points, ranges, unit)
        ha, hb = ha[valid_h], hb[valid_h]
        adjacent = (np.diff(ha) == 1) & (np.diff(hb) == 1)
        q = np.where(adjacent)[0]
        if not len(q):
            continue
        a0, a1 = ha[q], ha[q + 1]
        b0, b1 = hb[q], hb[q + 1]
        valid_a = _topology_edge_valid(a0, a1, raw_points, ranges, unit)
        valid_b = _topology_edge_valid(b0, b1, raw_points, ranges, unit)
        valid_diagonal = _topology_edge_valid(b0, a1, raw_points, ranges, unit)
        first = valid_a & valid_diagonal
        second = valid_b & valid_diagonal
        if first.any():
            triangles.append(np.stack([a0[first], b0[first], a1[first]], axis=1))
        if second.any():
            triangles.append(np.stack([b0[second], b1[second], a1[second]], axis=1))
    if not triangles:
        return np.empty((0, 3), dtype=np.int32)
    return np.concatenate(triangles, axis=0).astype(np.int32)


def _unified_target_pixel_zbuffer(points_ego, indices, xs, ys,
                                  ego_pose, camera_pose, width):
    """Keep one nearest candidate per target pixel after all candidate unions."""
    if indices.numel() == 0:
        return indices, xs, ys
    points = points_ego[indices]
    world = points @ ego_pose[:3, :3].T + ego_pose[:3, 3]
    camera = (world - camera_pose[:3, 3]) @ camera_pose[:3, :3]
    depth = camera[:, 2].detach().cpu().numpy()
    flat = (ys * width + xs).detach().cpu().numpy()
    order = np.lexsort((depth, flat))
    flat_sorted = flat[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = flat_sorted[1:] != flat_sorted[:-1]
    keep = torch.from_numpy(order[first]).long().to(indices.device)
    return indices[keep], xs[keep], ys[keep]


def _apply_topology_surface_visibility(points_ego, indices, xs, ys,
                                       ego_pose, camera_pose, K,
                                       topology_triangles, front_point_count,
                                       tolerance):
    """Filter real LiDAR candidates against a front-LiDAR topology mesh."""
    if (indices.numel() == 0 or topology_triangles is None or
            len(topology_triangles) == 0 or front_point_count <= 0):
        return indices, xs, ys
    import open3d as o3d

    front = points_ego[:front_point_count]
    front_world = front @ ego_pose[:3, :3].T + ego_pose[:3, 3]
    front_camera = ((front_world - camera_pose[:3, 3]) @
                    camera_pose[:3, :3]).detach().cpu().numpy().astype(np.float32)
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(front_camera, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(topology_triangles, dtype=o3d.core.Dtype.Int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    x = xs.detach().cpu().numpy().astype(np.float32)
    y = ys.detach().cpu().numpy().astype(np.float32)
    K_np = K.detach().cpu().numpy()
    rays = np.zeros((len(x), 6), dtype=np.float32)
    rays[:, 3] = (x - K_np[0, 2]) / K_np[0, 0]
    rays[:, 4] = (y - K_np[1, 2]) / K_np[1, 1]
    rays[:, 5] = 1.0
    surface_depth = scene.cast_rays(
        o3d.core.Tensor(rays))["t_hit"].numpy()

    points = points_ego[indices]
    world = points @ ego_pose[:3, :3].T + ego_pose[:3, 3]
    camera = (world - camera_pose[:3, 3]) @ camera_pose[:3, :3]
    point_depth = camera[:, 2].detach().cpu().numpy()
    keep_np = (~np.isfinite(surface_depth) |
               (point_depth <= surface_depth * (1.0 + tolerance)))
    keep = torch.from_numpy(keep_np).to(indices.device)
    return indices[keep], xs[keep], ys[keep]


# ==============================================================================
# Outlier filter functions for lidar depth
# ==============================================================================
def _filter_minpool(xs_u, ys_u, ds_u, H, W, device, filter_win=5,
                    max_ratio=0.02, near_splat_depth=0.0,
                    near_splat_win=7):
    """Original L29: 5x5 min-pooling outlier filter.

    Rejects a point if its depth > min_neighbor_depth * (1 + max_ratio),
    where min_neighbor_depth is the minimum depth in the 5x5 spatial window.

    Args:
        xs_u, ys_u, ds_u: tensors of pixel coords and depths (after unique).
        H, W: image dimensions.
        device: torch device.
        filter_win: spatial window size (must be odd).
        max_ratio: relative depth tolerance.

    Returns:
        (xs_u, ys_u, ds_u, keep_mask) — filtered tensors and boolean mask.
    """
    if len(xs_u) == 0:
        return xs_u, ys_u, ds_u, torch.ones(0, dtype=torch.bool, device=device)

    import torch.nn.functional as F
    depth_map = torch.full((H, W), float('inf'), device=device, dtype=torch.float32)
    depth_map[ys_u, xs_u] = ds_u
    dmap = depth_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    dmap[dmap == float('inf')] = 1e10
    min_pooled = -F.max_pool2d(-dmap, kernel_size=filter_win, stride=1,
                               padding=filter_win // 2)
    min_neighbor = min_pooled[0, 0, ys_u, xs_u]

    # Optional conservative extension: only nearby points receive a larger
    # footprint. The splatted pixels are used solely as an occlusion buffer;
    # they are never emitted as new LiDAR samples.
    if near_splat_depth > 0 and near_splat_win > filter_win:
        near = ds_u < near_splat_depth
        if near.any():
            near_map = torch.full((H, W), 1e10, device=device,
                                  dtype=torch.float32)
            near_map[ys_u[near], xs_u[near]] = ds_u[near]
            near_pool = -F.max_pool2d(
                -near_map.unsqueeze(0).unsqueeze(0),
                kernel_size=near_splat_win, stride=1,
                padding=near_splat_win // 2)
            min_neighbor = torch.minimum(
                min_neighbor, near_pool[0, 0, ys_u, xs_u])

    keep = ds_u <= min_neighbor * (1 + max_ratio)
    return xs_u[keep], ys_u[keep], ds_u[keep], keep


def _filter_bilateral(xs_u, ys_u, ds_u, H, W, device, filter_win=5,
                       max_ratio=0.02, surface_ratio=0.05):
    """Bilateral outlier filter: only compare against same-surface neighbors.

    For each point, looks at the 5x5 spatial window and finds neighbors whose
    depth is within `surface_ratio` relative difference (i.e. likely on the same
    surface).  The outlier check is then applied ONLY against those same-surface
    neighbors.  Isolated points (no same-surface neighbors) are kept.

    Args:
        xs_u, ys_u, ds_u: tensors of pixel coords and depths (after unique).
        H, W: image dimensions.
        device: torch device.
        filter_win: spatial window size (must be odd).
        max_ratio: relative depth tolerance for outlier rejection.
        surface_ratio: max relative depth difference for same-surface grouping.

    Returns:
        (xs_u, ys_u, ds_u, keep_mask) — filtered tensors and boolean mask.
    """
    if len(xs_u) == 0:
        return xs_u, ys_u, ds_u, torch.ones(0, dtype=torch.bool, device=device)

    # Build sparse depth map
    depth_map = torch.full((H, W), float('inf'), device=device, dtype=torch.float32)
    depth_map[ys_u, xs_u] = ds_u

    half = filter_win // 2
    keep_mask = torch.ones(len(xs_u), dtype=torch.bool, device=device)

    for i in range(len(xs_u)):
        x, y, d = xs_u[i].item(), ys_u[i].item(), ds_u[i].item()

        # Extract 5x5 spatial neighborhood
        x0, x1 = max(0, x - half), min(W, x + half + 1)
        y0, y1 = max(0, y - half), min(H, y + half + 1)
        patch = depth_map[y0:y1, x0:x1]

        # Valid neighbors (has a projected point)
        valid = torch.isfinite(patch)
        if not valid.any():
            continue  # isolated point, no neighbors → keep

        neighbor_depths = patch[valid]

        # Same-surface neighbors: relative depth difference < surface_ratio
        relative_diff = torch.abs(neighbor_depths - d) / d
        same_surface = neighbor_depths[relative_diff < surface_ratio]

        if len(same_surface) == 0:
            continue  # no same-surface neighbors → keep (don't guess)

        min_same_surface = same_surface.min()
        if d > min_same_surface * (1 + max_ratio):
            keep_mask[i] = False  # outlier among same-surface points

    return xs_u[keep_mask], ys_u[keep_mask], ds_u[keep_mask], keep_mask


def shift_pose_4x4(pose_4x4, shift_xyz, R=None):
    """Apply shift_xyz in the local coordinate frame to a 4x4 pose matrix.

    If R is None, the pose's own rotation is used (suitable for frame/ego poses).
    If R is provided, it is used to rotate the shift vector (suitable for camera
    poses that should be shifted in the ego vehicle's local frame).
    """
    out = pose_4x4.copy()
    if R is None:
        R = out[:3, :3]
    out[:3, 3] += R @ np.asarray(shift_xyz, dtype=out.dtype)
    return out


# ---- EIG computation functions ----
def _zero_gaussian_grads(pc):
    for model_name in pc.model_name_id.keys():
        model = getattr(pc, model_name)
        for attr in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']:
            if hasattr(model, attr):
                param = getattr(model, attr)
                if isinstance(param, torch.Tensor):
                    if param.grad is not None:
                        param.grad = None
                    param.requires_grad_(False)

def _render_with_grad(viewpoint_camera, pc, renderer):
    torch.set_grad_enabled(True)
    for model_name in pc.model_name_id.keys():
        model = getattr(pc, model_name)
        for attr in ['_xyz', '_features_dc', '_scaling', '_rotation', '_opacity']:
            if hasattr(model, attr):
                p = getattr(model, attr)
                if isinstance(p, torch.Tensor):
                    p.requires_grad_(True)
    include_list = list(set(pc.model_name_id.keys()))
    pc.set_visibility(include_list); pc.parse_camera(viewpoint_camera)
    num_gaussians = pc.num_gaussians
    if num_gaussians == 0: return None
    means3D = pc.get_xyz; opacity = pc.get_opacity
    scales = pc.get_scaling; rotations = pc.get_rotation; shs = pc.get_features
    means3D.retain_grad(); opacity.retain_grad(); scales.retain_grad()
    rotations.retain_grad(); shs.retain_grad()
    from lib.utils.camera_utils import make_rasterizer
    bg_color = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
    rasterizer = make_rasterizer(viewpoint_camera, pc.max_sh_degree, bg_color, renderer.cfg.scaling_modifier)
    screenspace_points = torch.zeros((num_gaussians, 3), requires_grad=True, device="cuda")
    screenspace_points.retain_grad()
    rendered_color, _, _, rendered_acc, _ = rasterizer(
        means3D=means3D, means2D=screenspace_points, opacities=opacity,
        shs=shs, colors_precomp=None, scales=scales, rotations=rotations,
        cov3D_precomp=None, semantics=None)
    return {"rgb": rendered_color, "acc": rendered_acc,
            "params": {'means': means3D, 'rotations': rotations, 'scales': scales,
                       'opacities': opacity, 'shs': shs},
            "ranges": dict(pc.graph_gaussian_range)}

def _extract_per_gaussian_grad(params, ranges):
    H_view = {}
    for model_name, (start, end) in ranges.items():
        n_model = end - start; h = torch.zeros(n_model, device="cuda")
        for param_tensor in params.values():
            if param_tensor.grad is not None:
                g = param_tensor.grad.detach().abs()
                if g.dim() >= 2: g = g.reshape(g.shape[0], -1).sum(dim=1)
                h += g[start:end]
        H_view[model_name] = h
    return H_view

def _render_and_backward(viewpoint_camera, pc, renderer):
    result = _render_with_grad(viewpoint_camera, pc, renderer)
    if result is None: _zero_gaussian_grads(pc); return None
    rgb = result["rgb"]; params = result["params"]; ranges = result["ranges"]
    rgb.backward(gradient=torch.ones_like(rgb), retain_graph=False)
    H = _extract_per_gaussian_grad(params, ranges)
    del result; _zero_gaussian_grads(pc); torch.cuda.empty_cache()
    return H

def _compute_I_train(H_accum, reg_lambda=1e-6):
    model_ranges = {}; offset = 0; h_parts = []
    for model_name, h in H_accum.items():
        n = h.shape[0]; model_ranges[model_name] = (offset, offset + n)
        h_parts.append(h); offset += n
    if len(h_parts) == 0: return None, None, None, {}
    H_full = torch.cat(h_parts, dim=0)
    I_train = torch.reciprocal(H_full + reg_lambda)
    return H_full, I_train, torch.sqrt(I_train), model_ranges

def _render_eig_map(viewpoint_camera, pc, renderer, I_train, model_ranges, sky_eig_mode="zero"):
    result = _render_with_grad(viewpoint_camera, pc, renderer)
    if result is None: return None
    rgb = result["rgb"].detach().clone(); acc = result["acc"].detach().clone()
    params = result["params"]; ranges = result["ranges"]
    result["rgb"].backward(gradient=torch.ones_like(result["rgb"]), retain_graph=False)
    H_view_per_model = _extract_per_gaussian_grad(params, ranges)
    N_total = params['means'].shape[0]
    H_view_cpu = {k: v.cpu() for k, v in H_view_per_model.items()}
    del result, params, H_view_per_model
    _zero_gaussian_grads(pc); torch.cuda.empty_cache()
    H_view_full = torch.zeros(N_total, device="cuda")
    for model_name, (vs, ve) in dict(ranges).items():
        if model_name in H_view_cpu: H_view_full[vs:ve] = H_view_cpu[model_name].cuda()
    del H_view_cpu
    I_train_view = torch.zeros(N_total, device="cuda")
    for model_name, (gs, ge) in model_ranges.items():
        if model_name in ranges:
            vs, ve = ranges[model_name]
            I_train_view[vs:ve] = I_train[gs:ge]
    hessian_color = torch.stack([torch.zeros(N_total, device="cuda"),
                                  torch.sqrt(I_train_view), H_view_full * I_train_view], dim=1)
    del H_view_full, I_train_view
    with torch.no_grad():
        include_list = list(set(pc.model_name_id.keys()))
        pc.set_visibility(include_list); pc.parse_camera(viewpoint_camera)
        rendered_hessian = renderer.render_kernel(viewpoint_camera, pc, override_color=hessian_color)
        rgb_hessian = rendered_hessian["rgb"]
    gain_map_raw = rgb_hessian[2:3, :, :].clone()
    uncertainty_map = rgb_hessian[0:1, :, :].clone()
    del rendered_hessian, rgb_hessian, hessian_color; torch.cuda.empty_cache()
    opacity_mask = (acc > 0.1).float(); sky_mask = (acc <= 0.1).squeeze(0)
    gain_map_raw_2d = gain_map_raw.squeeze(0) * opacity_mask.squeeze(0)
    gain_map = gain_map_raw_2d / 1000.0; gain_map_vis = gain_map_raw_2d
    uncertainty_map = uncertainty_map * opacity_mask
    del gain_map_raw, gain_map_raw_2d
    if sky_eig_mode == "max" and sky_mask.any() and (gain_map > 0).any():
        gain_map[sky_mask] = gain_map[~sky_mask].max()
        gain_map_vis[sky_mask] = gain_map_vis[~sky_mask].max()
        uncertainty_map[:, sky_mask] = uncertainty_map[:, ~sky_mask].max()
    torch.cuda.empty_cache()
    return {"rgb": rgb, "gain_map": gain_map, "gain_map_vis": gain_map_vis,
            "uncertainty_map": uncertainty_map.squeeze(0), "acc": acc}

# ---- 2D-projected-size gaussian filter (v2 feature) ----
def filter_large_gaussians(gaussians, camera, max_pixels=100):
    """Temporarily hide gaussians whose estimated 2D projected radius exceeds max_pixels.

    Computes the camera-space 3D covariance, extracts the 2x2 image-plane block,
    then derives the max eigenvalue. This correctly handles elongated gaussians:
    if the long axis points toward the camera, the projected size stays small.

    Formula:  radius_2d = 3sigma * sqrt(lambda_max) * fx / z
    where lambda_max is the larger eigenvalue of the projected 2D covariance.
    """
    fx = camera.K[0, 0].item()
    cam_center = camera.camera_center                              # [3] world (torch)
    R_w2c = torch.from_numpy(camera.R.T).float().to(cam_center.device)  # [3, 3] world->camera

    from lib.utils.general_utils import quaternion_to_matrix

    saved = {}
    for model_name in gaussians.model_name_id.keys():
        model = getattr(gaussians, model_name)
        xyz = model.get_xyz              # [N, 3] world
        if len(xyz) == 0:
            continue
        scaling = model.get_scaling      # [N, 3] after exp -> world units
        rots = model.get_rotation        # [N, 4] quaternion (normalized)

        # Euclidean depth (approx; difference to true z-depth is <= cos(45deg) ~ 0.7)
        vec_w = xyz - cam_center.unsqueeze(0)          # [N, 3]
        z_cam = vec_w.norm(dim=1).clamp(min=0.1)       # [N]

        # Camera-space rotation of gaussian local axes
        # R_local:  local->world    R_w2c @ R_local:  local->camera
        R_local = quaternion_to_matrix(rots)                     # [N, 3, 3]
        R_cam = R_w2c.unsqueeze(0) @ R_local                     # [N, 3, 3]
        # R_cam[n, :, k] = camera-frame basis vector for local axis k

        s2 = scaling * scaling  # [N, 3]

        # 2x2 image-plane block of camera-space 3D covariance:
        #   Sigma_cam[i,j] = Sum_k  R_cam[i,k] * s^2_k * R_cam[j,k]   for i,j in {0,1,2}
        # We only need the {0,1}x{0,1} sub-block (x, y in image plane).
        cov_xx = (R_cam[:, 0, :] * s2 * R_cam[:, 0, :]).sum(dim=1)  # [N]
        cov_yy = (R_cam[:, 1, :] * s2 * R_cam[:, 1, :]).sum(dim=1)  # [N]
        cov_xy = (R_cam[:, 0, :] * s2 * R_cam[:, 1, :]).sum(dim=1)  # [N]

        # Larger eigenvalue of the 2x2 symmetric matrix [[cov_xx, cov_xy], [cov_xy, cov_yy]]
        trace = cov_xx + cov_yy
        disc = torch.sqrt((cov_xx - cov_yy).square() + 4.0 * cov_xy.square() + 1e-12)
        lambda_max = 0.5 * (trace + disc)  # [N]

        # 3-sigma 2D radius in pixels
        radius_2d = 3.0 * torch.sqrt(lambda_max) * fx / z_cam

        large_mask = radius_2d > max_pixels
        if large_mask.any():
            saved[model_name] = {
                'opacity': model._opacity.data[large_mask].clone(),
                'mask': large_mask,
            }
            model._opacity.data[large_mask] = -100.0
    return saved


def restore_gaussians(gaussians, saved):
    for model_name, data in saved.items():
        model = getattr(gaussians, model_name)
        model._opacity.data[data['mask']] = data['opacity']


def load_ego_poses(source_dir):
    ego_dir = os.path.join(source_dir, 'ego_pose')
    all_files = sorted(glob(os.path.join(ego_dir, '*.txt')))
    frame_poses = {}
    cam_poses = {}
    for fp in all_files:
        stem = os.path.basename(fp).replace('.txt', '')
        pose = np.loadtxt(fp)
        if '_' in stem:
            cam_poses[stem] = pose
        else:
            frame_poses[stem] = pose
    return frame_poses, cam_poses


def render_depth_map(points_ego, ego_pose, cam_pose, K, H, W, device,
                     use_filter=True, max_ratio=0.02, filter_win=5,
                     filter_method='minpool', surface_ratio=0.05):
    """Project LiDAR points and render depth map with outlier filtering.

    filter_method: 'minpool' (original L29) or 'bilateral' (same-surface aware).
    surface_ratio: only used when filter_method='bilateral'.
    """
    N = points_ego.shape[0]
    if N == 0:
        return np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=np.float32)

    R_ego, t_ego = ego_pose[:3, :3], ego_pose[:3, 3]
    points_world = points_ego @ R_ego.T + t_ego
    R_cam, t_cam = cam_pose[:3, :3], cam_pose[:3, 3]
    points_cam = (points_world - t_cam) @ R_cam

    depth = points_cam[:, 2]
    uv_h = points_cam @ K.T
    uv = uv_h[:, :2] / torch.clamp(uv_h[:, 2:3], 1e-8)

    valid = (depth > 0.01) & torch.isfinite(depth) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not valid.any():
        return np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=np.float32)

    xs = torch.clamp(torch.round(uv[valid, 0]).long(), 0, W - 1)
    ys = torch.clamp(torch.round(uv[valid, 1]).long(), 0, H - 1)
    ds = depth[valid]

    # sort by depth descending (original)
    # order = torch.argsort(-ds)
    # 2026-07-28: changed to ascending so np.unique keeps nearest per pixel
    order = torch.argsort(ds)
    xs, ys, ds = xs[order], ys[order], ds[order]

    # keep nearest per pixel via unique (same as L29)
    stacked = torch.stack([xs, ys], dim=1)
    _, unique_idx = np.unique(stacked.cpu().numpy(), axis=0, return_index=True)
    xs_u, ys_u, ds_u = xs[unique_idx], ys[unique_idx], ds[unique_idx]

    # 5x5 neighborhood outlier filter
    if use_filter and len(xs_u) > 0:
        if filter_method == 'bilateral':
            xs_u, ys_u, ds_u, _keep = _filter_bilateral(
                xs_u, ys_u, ds_u, H, W, device,
                filter_win=filter_win, max_ratio=max_ratio,
                surface_ratio=surface_ratio)
        else:
            xs_u, ys_u, ds_u, _keep = _filter_minpool(
                xs_u, ys_u, ds_u, H, W, device,
                filter_win=filter_win, max_ratio=max_ratio)

    mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    depth_out = torch.zeros((H, W), dtype=torch.float32, device=device)
    if len(xs_u) > 0:
        mask[ys_u, xs_u] = True
        depth_out[ys_u, xs_u] = ds_u

    return mask.cpu().numpy(), depth_out.cpu().numpy()


def get_surviving_indices(points_ego, ego_pose, cam_pose, K, H, W, device,
                          use_filter=True, max_ratio=0.02, filter_win=5,
                          filter_method='minpool', surface_ratio=0.05):
    """Project points to a camera and return indices of points that survive the filter.

    Mirrors render_depth_map pipeline (project -> sort -> nearest-per-pixel -> filter)
    but returns surviving point indices instead of a depth map.

    filter_method: 'minpool' (original L29) or 'bilateral' (same-surface aware).
    """
    N = points_ego.shape[0]
    if N == 0:
        return set()

    R_ego, t_ego = ego_pose[:3, :3], ego_pose[:3, 3]
    points_world = points_ego @ R_ego.T + t_ego
    R_cam, t_cam = cam_pose[:3, :3], cam_pose[:3, 3]
    points_cam = (points_world - t_cam) @ R_cam

    depth = points_cam[:, 2]
    uv_h = points_cam @ K.T
    uv = uv_h[:, :2] / torch.clamp(uv_h[:, 2:3], 1e-8)

    valid = (depth > 0.01) & torch.isfinite(depth) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not valid.any():
        return set()

    # Track original indices through the pipeline
    all_indices = torch.arange(N, device=device)
    valid_indices = all_indices[valid]

    xs = torch.clamp(torch.round(uv[valid, 0]).long(), 0, W - 1)
    ys = torch.clamp(torch.round(uv[valid, 1]).long(), 0, H - 1)
    ds = depth[valid]

    order = torch.argsort(ds)
    xs, ys, ds = xs[order], ys[order], ds[order]
    idx_ordered = valid_indices[order]

    # keep nearest per pixel via unique
    stacked = torch.stack([xs, ys], dim=1)
    _, unique_idx = np.unique(stacked.cpu().numpy(), axis=0, return_index=True)
    xs_u, ys_u, ds_u = xs[unique_idx], ys[unique_idx], ds[unique_idx]
    idx_u = idx_ordered[unique_idx]

    # 5x5 neighborhood outlier filter
    if use_filter and len(xs_u) > 0:
        if filter_method == 'bilateral':
            xs_u, ys_u, ds_u, _keep = _filter_bilateral(
                xs_u, ys_u, ds_u, H, W, device,
                filter_win=filter_win, max_ratio=max_ratio,
                surface_ratio=surface_ratio)
        else:
            xs_u, ys_u, ds_u, _keep = _filter_minpool(
                xs_u, ys_u, ds_u, H, W, device,
                filter_win=filter_win, max_ratio=max_ratio)
        idx_u = idx_u[_keep]

    return set(idx_u.cpu().numpy().tolist())


def vis_depth_as_color(depth, mask):
    """L29 style with PLASMA colormap, 1-99% percentile, plus 3px dilation."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    ys, xs = np.where(mask_dilated)
    if len(ys) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    # depth values come from original (non-dilated) points
    ds = depth[mask]
    if len(ds) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    near = np.percentile(ds, 1)
    far = np.percentile(ds, 99)
    if far <= near:
        far = near + 1.0

    depth_vis = depth.copy()
    depth_vis[~mask] = 0
    norm = np.clip((depth_vis - near) / (far - near), 0.0, 1.0)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    color[~mask_dilated] = (0, 0, 0)
    return color


def get_surviving_pixels(points_ego, ego_pose, cam_pose, K, H, W, device,
                          use_filter=True, max_ratio=0.02, filter_win=5,
                          filter_method='minpool', surface_ratio=0.05):
    """Project points to a camera and return (indices, xs, ys) of surviving points.

    Same pipeline as get_surviving_indices but also returns pixel coordinates.
    Returns three numpy arrays: original_point_indices, pixel_xs, pixel_ys.

    filter_method: 'minpool' (original L29) or 'bilateral' (same-surface aware).
    """
    N = points_ego.shape[0]
    if N == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.array([], dtype=np.int64))

    R_ego, t_ego = ego_pose[:3, :3], ego_pose[:3, 3]
    points_world = points_ego @ R_ego.T + t_ego
    R_cam, t_cam = cam_pose[:3, :3], cam_pose[:3, 3]
    points_cam = (points_world - t_cam) @ R_cam

    depth = points_cam[:, 2]
    uv_h = points_cam @ K.T
    uv = uv_h[:, :2] / torch.clamp(uv_h[:, 2:3], 1e-8)

    valid = (depth > 0.01) & torch.isfinite(depth) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not valid.any():
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.array([], dtype=np.int64))

    all_indices = torch.arange(N, device=device)
    valid_indices = all_indices[valid]

    xs = torch.clamp(torch.round(uv[valid, 0]).long(), 0, W - 1)
    ys = torch.clamp(torch.round(uv[valid, 1]).long(), 0, H - 1)
    ds = depth[valid]

    order = torch.argsort(ds)
    xs, ys, ds = xs[order], ys[order], ds[order]
    idx_ordered = valid_indices[order]

    # keep nearest per pixel via unique
    stacked = torch.stack([xs, ys], dim=1)
    _, unique_idx = np.unique(stacked.cpu().numpy(), axis=0, return_index=True)
    xs_u, ys_u, ds_u = xs[unique_idx], ys[unique_idx], ds[unique_idx]
    idx_u = idx_ordered[unique_idx]

    # 5x5 neighborhood outlier filter
    if use_filter and len(xs_u) > 0:
        if filter_method == 'bilateral':
            xs_u, ys_u, ds_u, _keep = _filter_bilateral(
                xs_u, ys_u, ds_u, H, W, device,
                filter_win=filter_win, max_ratio=max_ratio,
                surface_ratio=surface_ratio)
        else:
            xs_u, ys_u, ds_u, _keep = _filter_minpool(
                xs_u, ys_u, ds_u, H, W, device,
                filter_win=filter_win, max_ratio=max_ratio)
        idx_u = idx_u[_keep]

    return idx_u.cpu().numpy(), xs_u.cpu().numpy(), ys_u.cpu().numpy()


def get_surviving_pixels_torch(points_ego, ego_pose, cam_pose, K, H, W, device,
                                use_filter=True, max_ratio=0.02, filter_win=5,
                                filter_method='minpool', surface_ratio=0.05,
                                near_splat_depth=0.0, near_splat_win=7):
    """Same pipeline as get_surviving_pixels but returns torch tensors on device.

    Returns (indices, xs, ys) as torch tensors, or empty tensors if none survive.
    """
    N = points_ego.shape[0]
    empty = (torch.tensor([], dtype=torch.int64, device=device),
             torch.tensor([], dtype=torch.int64, device=device),
             torch.tensor([], dtype=torch.int64, device=device))
    if N == 0:
        return empty

    R_ego, t_ego = ego_pose[:3, :3], ego_pose[:3, 3]
    points_world = points_ego @ R_ego.T + t_ego
    R_cam, t_cam = cam_pose[:3, :3], cam_pose[:3, 3]
    points_cam = (points_world - t_cam) @ R_cam

    depth = points_cam[:, 2]
    uv_h = points_cam @ K.T
    uv = uv_h[:, :2] / torch.clamp(uv_h[:, 2:3], 1e-8)

    valid = (depth > 0.01) & torch.isfinite(depth) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not valid.any():
        return empty

    all_indices = torch.arange(N, device=device)
    valid_indices = all_indices[valid]

    xs = torch.clamp(torch.round(uv[valid, 0]).long(), 0, W - 1)
    ys = torch.clamp(torch.round(uv[valid, 1]).long(), 0, H - 1)
    ds = depth[valid]

    order = torch.argsort(ds)
    xs, ys, ds = xs[order], ys[order], ds[order]
    idx_ordered = valid_indices[order]

    # keep nearest per pixel via unique
    stacked = torch.stack([xs, ys], dim=1)
    _, unique_idx = np.unique(stacked.cpu().numpy(), axis=0, return_index=True)
    xs_u, ys_u, ds_u = xs[unique_idx], ys[unique_idx], ds[unique_idx]
    idx_u = idx_ordered[unique_idx]

    # 5x5 neighborhood outlier filter
    if use_filter and len(xs_u) > 0:
        if filter_method == 'bilateral':
            xs_u, ys_u, ds_u, _keep = _filter_bilateral(
                xs_u, ys_u, ds_u, H, W, device,
                filter_win=filter_win, max_ratio=max_ratio,
                surface_ratio=surface_ratio)
        else:
            xs_u, ys_u, ds_u, _keep = _filter_minpool(
                xs_u, ys_u, ds_u, H, W, device,
                filter_win=filter_win, max_ratio=max_ratio,
                near_splat_depth=near_splat_depth,
                near_splat_win=near_splat_win)
        idx_u = idx_u[_keep]

    return idx_u, xs_u, ys_u


def _apply_color_xform_torch(colors, method, params):
    """Apply color balance transform to a torch tensor of colors.

    Args:
        colors: [N, 3] float tensor (BGR)
        method: 'linear', 'matrix', or 'gamma'
        params: method-specific parameters

    Returns:
        [N, 3] float tensor (BGR), clamped to [0, 255]
    """
    if method == 'gamma':
        g, gm, o = params
        g_t = torch.tensor(g, dtype=torch.float32, device=colors.device)
        gm_t = torch.tensor(gm, dtype=torch.float32, device=colors.device)
        o_t = torch.tensor(o, dtype=torch.float32, device=colors.device)
        colors = g_t * torch.pow(colors.clamp(1, 255) / 255.0, gm_t) * 255.0 + o_t
    elif method == 'matrix':
        M = torch.tensor(params[0], dtype=torch.float32, device=colors.device)  # [3, 3]
        b = torch.tensor(params[1], dtype=torch.float32, device=colors.device)  # [3]
        colors = colors @ M.T + b
    else:  # linear
        gain = torch.tensor(params[0], dtype=torch.float32, device=colors.device)
        offset = torch.tensor(params[1], dtype=torch.float32, device=colors.device)
        colors = colors * gain + offset
    return torch.clamp(colors, 0, 255)


def _build_per_channel_lut(src_t, tgt_t, n_bins=256):
    """Build per-channel LUT via CDF matching on GPU. Returns 3×256 array."""
    device = src_t.device
    lut = torch.zeros(3, n_bins, device=device)
    for ch in range(3):
        s = src_t[:, ch]; t = tgt_t[:, ch]
        s_hist = torch.histc(s, bins=n_bins, min=0, max=255)
        t_hist = torch.histc(t, bins=n_bins, min=0, max=255)
        s_cdf = torch.cumsum(s_hist, 0) / (s_hist.sum() + 1e-8)
        t_cdf = torch.cumsum(t_hist, 0) / (t_hist.sum() + 1e-8)
        ti = 0
        for i in range(n_bins):
            while ti < n_bins - 1 and t_cdf[ti] < s_cdf[i]:
                ti += 1
            lut[ch, i] = ti * 255.0 / (n_bins - 1)
    return lut


def _apply_lut_np(colors, lut):
    """Apply per-channel LUT to numpy array (N,3) uint8 → (N,3) uint8."""
    colors_t = torch.from_numpy(colors).long().to(lut.device)
    out_t = torch.zeros_like(colors_t)
    for ch in range(3):
        out_t[:, ch] = lut[ch, colors_t[:, ch]]
    return out_t.cpu().numpy()


def _torch_intersect1d(a, b):
    """GPU version of np.intersect1d. Returns (common, idx_a, idx_b)."""
    combined = torch.cat([a, b])
    uniq, inv = torch.unique(combined, return_inverse=True)
    counts = torch.bincount(inv)
    dup = uniq[counts > 1]
    # Find positions in a and b
    a_mask = torch.isin(a, dup)
    b_mask = torch.isin(b, dup)
    a_idx = torch.where(a_mask)[0]
    b_idx = torch.where(b_mask)[0]
    # Align by value
    a_vals, a_sort = torch.sort(a[a_idx])
    b_vals, b_sort = torch.sort(b[b_idx])
    return a_vals, a_idx[a_sort], b_idx[b_sort]


def render_warped_color_image_v2(points_ego, ego_pose, cam_pose_shifted,
                                  K_render, H_render, W_render, cam_id, device,
                                  frame_cam_data,
                                  filter_method='minpool', surface_ratio=0.05,
                                  target_filter_win=5,
                                  near_splat_depth=0.0, near_splat_win=7,
                                  fallback_cam=None, partner_cam=None,
                                  partner_lut=None,
                                  topology_triangles=None,
                                  topology_front_count=0,
                                  topology_tolerance=0.02):
    """Render warped LiDAR point colors into a novel camera view using torch.

    Strategy:
    1. Project shifted points to the shifted camera → surviving pixels.
    2. If partner_cam set, also project to partner's shifted camera, union indices.
       Partner-only points get colored from partner's GT + LUT.
    3. Re-project primary points through primary camera's original pose
       to sample GT color.
    4. If fallback_cam set, remaining points colored from fallback camera + LUT.
    5. Points not visible from any camera stay black.

    Args:
        points_ego: [N, 3] tensor, shifted LiDAR points in ego frame
        ego_pose: [4, 4] tensor, ego vehicle pose
        cam_pose_shifted: [4, 4] tensor, shifted camera extrinsic (W2C)
        K_render: [3, 3] tensor, camera intrinsics (render resolution)
        H_render, W_render: int, render image dimensions
        cam_id: int, current camera ID
        device: torch device
        frame_cam_data: dict cam_id -> (c2w_original_4x4, K_3x3, H, W, gt_img_np)
        filter_method: 'minpool' or 'bilateral'
        surface_ratio: float

    Returns:
        warped_img: np.ndarray [H_render, W_render, 3] uint8 BGR
    """
    # Step 1: Project all points to the processing (shifted) camera
    idx_survive, xs_survive, ys_survive = get_surviving_pixels_torch(
        points_ego, ego_pose, cam_pose_shifted, K_render,
        H_render, W_render, device,
        filter_method=filter_method, surface_ratio=surface_ratio,
        filter_win=target_filter_win,
        near_splat_depth=near_splat_depth,
        near_splat_win=near_splat_win)

    primary_count = int(idx_survive.shape[0])

    # Step 1b: Optionally add partner-visible points as candidates. The target
    # camera z-buffer and topology surface below make the final visibility
    # decision for the combined set.
    if partner_cam is not None and isinstance(partner_cam, tuple):
        c2w_p, K_p, Hp, Wp = partner_cam
        idx_p, xs_p, ys_p = get_surviving_pixels_torch(
            points_ego, ego_pose, c2w_p, K_p, Hp, Wp, device,
            use_filter=False)
        if idx_p.shape[0] > 0:
            pts_p = points_ego[idx_p]
            pts_w = pts_p @ ego_pose[:3,:3].T + ego_pose[:3,3]
            pts_c = (pts_w - cam_pose_shifted[:3,3]) @ cam_pose_shifted[:3,:3]
            depth_p = pts_c[:,2]
            valid_p = (depth_p > 0.01) & torch.isfinite(depth_p)
            if valid_p.any():
                uv_h = pts_c[valid_p] @ K_render.T
                uv = uv_h[:,:2] / torch.clamp(uv_h[:,2:3], 1e-8)
                in_b = (uv[:,0] >= 0) & (uv[:,0] < W_render) & \
                       (uv[:,1] >= 0) & (uv[:,1] < H_render)
                if in_b.any():
                    idx_pv = idx_p[valid_p][in_b]
                    xs_pv = torch.clamp(torch.round(uv[in_b,0]).long(), 0, W_render-1)
                    ys_pv = torch.clamp(torch.round(uv[in_b,1]).long(), 0, H_render-1)
                    idx_survive = torch.cat([idx_survive, idx_pv])
                    xs_survive = torch.cat([xs_survive, xs_pv])
                    ys_survive = torch.cat([ys_survive, ys_pv])

    idx_survive, xs_survive, ys_survive = _unified_target_pixel_zbuffer(
        points_ego, idx_survive, xs_survive, ys_survive,
        ego_pose, cam_pose_shifted, W_render)
    before_topology = int(idx_survive.shape[0])
    idx_survive, xs_survive, ys_survive = _apply_topology_surface_visibility(
        points_ego, idx_survive, xs_survive, ys_survive,
        ego_pose, cam_pose_shifted, K_render,
        topology_triangles, topology_front_count, topology_tolerance)

    M = idx_survive.shape[0]
    visibility_stats = {
        'primary_count': primary_count,
        'final_count': int(M),
        'partner_added': max(0, before_topology - primary_count),
        'topology_removed': before_topology - int(M),
    }
    warped_img = np.zeros((H_render, W_render, 3), dtype=np.uint8)
    fallback_masks = {}  # {cam_id: bool mask}
    if M == 0:
        return warped_img, fallback_masks, None, None, None, visibility_stats

    pts_survive = points_ego[idx_survive]  # [M, 3]
    remaining = torch.ones(M, dtype=torch.bool, device=device)

    R_ego = ego_pose[:3, :3]
    t_ego = ego_pose[:3, 3]

    # Step 2: Only use this camera's own GT colors.
    # Points not visible from this camera are left black.
    cid = cam_id
    if cid in frame_cam_data:
        c2w_orig_np, K_np, H_cam, W_cam, gt_img_np = frame_cam_data[cid]
        rem_idx = torch.where(remaining)[0]
        pts_rem = pts_survive[rem_idx]

        c2w_orig = torch.from_numpy(c2w_orig_np).float().to(device)
        K_cam = torch.from_numpy(K_np).float().to(device)
        gt_img_t = torch.from_numpy(gt_img_np.astype(np.float32)).to(device)

        pts_world = pts_rem @ R_ego.T + t_ego
        R_cam = c2w_orig[:3, :3]
        t_cam = c2w_orig[:3, 3]
        pts_cam = (pts_world - t_cam) @ R_cam

        depth_cam = pts_cam[:, 2]
        uv_h = pts_cam @ K_cam.T
        uv = uv_h[:, :2] / torch.clamp(uv_h[:, 2:3], 1e-8)

        valid_proj = (depth_cam > 0.01) & torch.isfinite(depth_cam) & \
                     (uv[:, 0] >= 0) & (uv[:, 0] < W_cam - 1) & \
                     (uv[:, 1] >= 0) & (uv[:, 1] < H_cam - 1)

        if valid_proj.any():
            px = torch.clamp(torch.round(uv[valid_proj, 0]).long(), 0, W_cam - 1)
            py = torch.clamp(torch.round(uv[valid_proj, 1]).long(), 0, H_cam - 1)
            colors = gt_img_t[py, px]

            global_valid = rem_idx[valid_proj]
            out_x = xs_survive[global_valid].cpu().numpy().astype(int)
            out_y = ys_survive[global_valid].cpu().numpy().astype(int)

            colors_np = torch.clamp(colors, 0, 255).cpu().numpy().astype(np.uint8)
            warped_img[out_y, out_x] = colors_np
            remaining[global_valid] = False

    # Step 3: Fallback cameras in priority order (partner first, then side)
    fb_list = []
    if isinstance(fallback_cam, (list, tuple)):
        fb_list = list(fallback_cam)
    elif fallback_cam is not None:
        fb_list = [fallback_cam]
    for fc in fb_list:
        if fc is None or fc not in frame_cam_data or not remaining.any():
            continue
        c2w_orig_np, K_np, H_cam, W_cam, gt_img_np = frame_cam_data[fc]
        rem_idx = torch.where(remaining)[0]
        pts_rem = pts_survive[rem_idx]

        c2w_orig = torch.from_numpy(c2w_orig_np).float().to(device)
        K_cam = torch.from_numpy(K_np).float().to(device)
        gt_img_t = torch.from_numpy(gt_img_np.astype(np.float32)).to(device)

        pts_world = pts_rem @ R_ego.T + t_ego
        R_cam = c2w_orig[:3, :3]
        t_cam = c2w_orig[:3, 3]
        pts_cam = (pts_world - t_cam) @ R_cam

        depth_cam = pts_cam[:, 2]
        uv_h = pts_cam @ K_cam.T
        uv = uv_h[:, :2] / torch.clamp(uv_h[:, 2:3], 1e-8)

        valid_proj = (depth_cam > 0.01) & torch.isfinite(depth_cam) & \
                     (uv[:, 0] >= 0) & (uv[:, 0] < W_cam - 1) & \
                     (uv[:, 1] >= 0) & (uv[:, 1] < H_cam - 1)

        if valid_proj.any():
            px = torch.clamp(torch.round(uv[valid_proj, 0]).long(), 0, W_cam - 1)
            py = torch.clamp(torch.round(uv[valid_proj, 1]).long(), 0, H_cam - 1)
            colors = gt_img_t[py, px]

            global_valid = rem_idx[valid_proj]
            out_x = xs_survive[global_valid].cpu().numpy().astype(int)
            out_y = ys_survive[global_valid].cpu().numpy().astype(int)
            colors_np = torch.clamp(colors, 0, 255).cpu().numpy().astype(np.uint8)
            warped_img[out_y, out_x] = colors_np
            if fc not in fallback_masks:
                fallback_masks[fc] = np.zeros((H_render, W_render), dtype=bool)
            fallback_masks[fc][out_y, out_x] = True
            remaining[global_valid] = False

    return (warped_img, fallback_masks, idx_survive, xs_survive, ys_survive,
            visibility_stats)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Shift ego poses and render novel trajectory images with pose interpolation.',
    )
    parser.add_argument('--source_path', type=str, required=True,
                        help='Source dataset root (must contain ego_pose/, images/, etc.)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Trained model directory (contains configs/, trained_model/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for rendered results')
    parser.add_argument('--shift', type=str, default='shift_right',
                        choices=list(SHIFT_TYPES.keys()),
                        help='Predefined shift type (default: shift_right)')
    parser.add_argument('--shift_vector', type=float, nargs=3, default=None,
                        metavar=('DX', 'DY', 'DZ'),
                        help='Custom shift vector in meters (overrides --shift)')
    parser.add_argument('--loaded_iter', type=int, default=-1,
                        help='Checkpoint iteration to load (-1 = latest)')
    parser.add_argument('--vis_lidar_depth', action='store_true',
                        help='Render and save lidar depth visualization images')
    parser.add_argument('--max_pixels', type=float, default=0.0,
                        help='Hide gaussians whose estimated 2D radius exceeds this (pixels). '
                             '0 = disabled. Recommended: 50-150 for novel views.')
    parser.add_argument('--filter_camera_rejected', action='store_true',
                        help='Exclude points with camera_projection=[-1,-1,-1] from lidar depth '
                             'rendering. When set, pointcloud.npz is NOT saved.')
    parser.add_argument('--interp_ratio', type=float, default=0.0,
                        help='Interpolation ratio for smooth shift ramping (default: 0.0). '
                             '0.2 = first 10%% frames ramp up, last 10%% ramp down, '
                             'middle 80%% at full shift. 0.0 = no interpolation.')
    parser.add_argument('--frame_range', type=int, nargs=2, default=None,
                        metavar=('START', 'END'),
                        help='Only save output images for frames in [START, END] (inclusive). '
                             'Interpolation curve is computed within this range. '
                             'When omitted, all frames are rendered and saved.')
    parser.add_argument('--fps', type=int, default=10,
                        help='Video FPS for output MP4 files (default: 10)')
    parser.add_argument('--filter_method', type=str, default='minpool',
                        choices=['minpool', 'bilateral'],
                        help='Outlier filter method for lidar depth (default: minpool). '
                             'minpool: 5x5 min-pooling (original L29). '
                             'bilateral: same-surface depth comparison.')
    parser.add_argument('--bilateral_surface_ratio', type=float, default=0.05,
                        help='Max relative depth difference for same-surface neighbors '
                             'in bilateral filter (default: 0.05 = 5%%).')
    parser.add_argument('--target_filter_win', type=int, default=5,
                        help='Odd target-camera depth neighborhood size. For minpool, '
                             'this is mathematically equivalent to the same-sized '
                             'depth splat visibility test (default: 5).')
    parser.add_argument('--near_splat_depth', type=float, default=0.0,
                        help='Apply a larger occlusion footprint only to points '
                             'nearer than this depth in meters. 0 disables it.')
    parser.add_argument('--near_splat_win', type=int, default=7,
                        help='Odd footprint size for near points (default: 7).')
    parser.add_argument('--enable_partner_geometry', action='store_true',
                        help='Allow configured shifted partner cameras to add candidate '
                             'points. The target camera still decides final visibility.')
    parser.add_argument('--partner_geometry_camera_ids', type=int, nargs='+',
                        default=[9, 10],
                        help='Target camera IDs that may receive partner geometry '
                             '(default: 9 10).')
    parser.add_argument('--disable_topology_visibility',
                        dest='topology_visibility', action='store_false',
                        help='Disable front-LiDAR topology surface visibility. '
                             'It is enabled by default.')
    parser.set_defaults(topology_visibility=True)
    parser.add_argument('--front_lidar_dir', type=str, default=None,
                        help='Directory containing raw front_lidar PCD files. Required '
                             'unless topology visibility is disabled.')
    parser.add_argument('--front_lidar_frame_offset', type=int, default=None,
                        help='Raw PCD index offset. By default it is inferred from the '
                             'source directory suffix _fSTART_END.')
    parser.add_argument('--topology_visibility_tolerance', type=float, default=0.02,
                        help='Relative target depth tolerance for topology visibility '
                             '(default: 0.02).')
    parser.add_argument('--camera_ids', type=int, nargs='+', default=[9, 10],
                        help='Only render and save images for these camera IDs '
                             '(default: 9 10). Example: --camera_ids 9 10')
    parser.add_argument('--render_scale', type=float, default=5.0/6.0,
                        help='Render resolution scale (default: 5/6). '
                             '1.0 = full res, 0.5 = half res. '
                             'Original paper uses 1600px max width; 5/6 '
                             'compensates for loadCam cap.')
    parser.add_argument('--color_balance', type=str, default='linear',
                        choices=['linear', 'matrix', 'gamma'],
                        help='Color balance method for cross-camera warped color. '
                             'linear: per-channel gain+offset (default). '
                             'matrix: 3x3 color matrix + offset (12 params). '
                             'gamma: gain * src^gamma + offset (nonlinear).')
    parser.add_argument('--downscale', type=int, default=None,
                        help='Also output downscaled videos (e.g. 2 = half res). '
                             'RGB/GT are resized; warped color is re-projected '
                             'at lower res to keep LiDAR point density.')
    parser.add_argument('--output_eig', action='store_true',
                        help='Compute and save EIG maps.')
    parser.add_argument('--num_observed', type=int, default=-1,
                        help='Training views for EIG H-accum (-1=all).')
    parser.add_argument('--sky_eig_mode', type=str, default='zero', choices=['zero','max'],
                        help='Sky EIG handling.')
    parser.add_argument('--ori_csv', type=str,
                        default='/mnt/yswang-wan22/dataset/L3_restore/all_jsons.csv',
                        help='CSV mapping clip_name+camera_idx to original json. '
                             'Used to look up caption and ori_json path.')
    args = parser.parse_args()

    if args.interp_ratio < 0.0:
        parser.error('--interp_ratio must be >= 0.0')
    if args.target_filter_win < 1 or args.target_filter_win % 2 == 0:
        parser.error('--target_filter_win must be a positive odd integer')
    if args.near_splat_depth < 0:
        parser.error('--near_splat_depth must be >= 0')
    if args.near_splat_win < 1 or args.near_splat_win % 2 == 0:
        parser.error('--near_splat_win must be a positive odd integer')
    if args.topology_visibility and not args.front_lidar_dir:
        parser.error('--front_lidar_dir is required because topology visibility '
                     'is enabled by default')
    if args.topology_visibility_tolerance < 0:
        parser.error('--topology_visibility_tolerance must be >= 0')
    if args.frame_range is not None:
        if args.frame_range[0] > args.frame_range[1]:
            parser.error('--frame_range START must be <= END')

    shift_vec = args.shift_vector if args.shift_vector is not None else SHIFT_TYPES[args.shift]
    shift_name = args.shift if args.shift_vector is None else 'custom_shift'
    shift_vec_np = np.array(shift_vec, dtype=np.float64)

    is_interp = args.interp_ratio > 0.0
    has_frame_range = args.frame_range is not None

    print("=" * 64)
    print(f"Source path    : {args.source_path}")
    print(f"Model path     : {args.model_path}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Shift          : {shift_name} {shift_vec}")
    print(f"Interp ratio   : {args.interp_ratio}" + (" (interpolation mode)" if is_interp else " (constant)"))
    if has_frame_range:
        print(f"Frame range    : [{args.frame_range[0]}, {args.frame_range[1]}]")
    filter_desc = f"{args.filter_method}"
    if args.filter_method == 'bilateral':
        filter_desc += f" (surface_ratio={args.bilateral_surface_ratio})"
    print(f"Filter method  : {filter_desc}")
    print(f"Target filt win: {args.target_filter_win}x{args.target_filter_win}")
    if args.near_splat_depth > 0:
        print(f"Near splat     : depth < {args.near_splat_depth}m, "
              f"{args.near_splat_win}x{args.near_splat_win}")
    print(f"Camera IDs     : {args.camera_ids}")
    if args.topology_visibility:
        print(f"Topology       : front LiDAR ({args.front_lidar_dir})")
    print("=" * 64)

    front_lidar_files = []
    topology_cache = OrderedDict()
    topology_frame_offset = 0
    if args.topology_visibility:
        if not os.path.isdir(args.front_lidar_dir):
            raise FileNotFoundError(
                f"front_lidar directory not found: {args.front_lidar_dir}")
        front_lidar_files = sorted(glob(os.path.join(args.front_lidar_dir, '*.pcd')))
        if not front_lidar_files:
            raise FileNotFoundError(
                f"no PCD files found in: {args.front_lidar_dir}")
        if args.front_lidar_frame_offset is not None:
            topology_frame_offset = args.front_lidar_frame_offset
        else:
            scene_name = os.path.basename(os.path.dirname(
                args.source_path.rstrip('/')))
            match = re.search(r'_f(\d+)_\d+', scene_name)
            if match:
                topology_frame_offset = int(match.group(1))

    def load_frame_topology(frame_id):
        if frame_id in topology_cache:
            triangles, count = topology_cache.pop(frame_id)
            topology_cache[frame_id] = (triangles, count)
            return triangles, count
        raw_index = frame_id + topology_frame_offset
        if raw_index < 0 or raw_index >= len(front_lidar_files):
            raise IndexError(
                f"front_lidar PCD index {raw_index} is unavailable for frame {frame_id}")
        import open3d as o3d
        raw = o3d.t.io.read_point_cloud(front_lidar_files[raw_index])
        raw_points = raw.point.positions.numpy().astype(np.float64)
        timestamps = raw.point.timestamp.numpy().reshape(-1)
        triangles = build_front_lidar_topology_triangles(raw_points, timestamps)
        topology_cache[frame_id] = (triangles, len(raw_points))
        while len(topology_cache) > 2:
            topology_cache.popitem(last=False)
        return triangles, len(raw_points)

    # ---- Step 1: Configure rendering with ORIGINAL source path ----
    # sys.argv must be set BEFORE importing lib.config (it parses args at import)
    config_yaml = os.path.join(args.model_path, 'configs', 'config_000000.yaml')
    if not os.path.exists(config_yaml):
        raise FileNotFoundError(f"Config not found: {config_yaml}")

    sys.argv = [
        'render_shifted',
        '--config', config_yaml,
        '--mode', 'trajectory',
        'source_path', args.source_path,
        'model_path', args.model_path,
    ]

    from lib.config import cfg
    cfg.mode = 'trajectory'  # override YAML's mode: train
    if args.loaded_iter > 0:
        cfg.loaded_iter = args.loaded_iter

    # loadCam caps resolution at 1600px width. Override resolution_scales so
    # rendered output matches the source data's original resolution.
    cfg.resolution_scales = [args.render_scale]
    render_scale = args.render_scale

    import torch
    from tqdm import tqdm
    from lib.models.street_gaussian_model import StreetGaussianModel
    from lib.models.street_gaussian_renderer import StreetGaussianRenderer
    from lib.datasets.dataset import Dataset
    from lib.models.scene import Scene
    from lib.utils.general_utils import safe_state
    import torchvision

    cfg.render.save_image = True
    safe_state(cfg.eval.quiet)

    print(f"Config source_path: {cfg.source_path}")
    print(f"Config model_path: {cfg.model_path}")

    # ---- Step 2: Load dataset and model ----
    print(f"\n[Step 2] Loading dataset and model...")

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)

        images_dir = os.path.join(save_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        train_cameras = scene.getTrainCameras(scale=render_scale)
        test_cameras = scene.getTestCameras(scale=render_scale)
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        print(f"Loaded {len(cameras)} cameras "
              f"({len(train_cameras)} train + {len(test_cameras)} test)")

        # ---- EIG: Accumulate H and compute I_train ----
        I_train = model_ranges = None
        _eig_dir = os.path.join(save_dir, 'eig')
        if args.output_eig:
            num_obs = args.num_observed
            if num_obs == -1: num_obs = len(train_cameras)
            elif num_obs <= 0: num_obs = max(len(train_cameras) // 2, 1)
            observed = train_cameras[:num_obs]
            print(f"\n[EIG Phase 1] Accumulating H on {len(observed)} training views...")
            H_accum = {}
            for idx, cam in enumerate(tqdm(observed, desc="H accum", leave=False)):
                Hv = _render_and_backward(cam, gaussians, renderer)
                if Hv is not None:
                    for name, h in Hv.items():
                        H_accum[name] = H_accum.get(name, 0) + h
                if (idx + 1) % 10 == 0: torch.cuda.empty_cache(); gc.collect()
            torch.set_grad_enabled(False); torch.cuda.empty_cache()
            print(f"  Accumulated H for {len(H_accum)} sub-models")
            print(f"[EIG Phase 2] Computing I_train...")
            _, I_train, _, model_ranges = _compute_I_train(H_accum)
            del H_accum; gc.collect(); torch.cuda.empty_cache()
            if I_train is not None:
                print(f"  I_train ready: {I_train.shape[0]:,} Gaussians")
            else:
                print(f"  ERROR: No Gaussians, disabling EIG")
                args.output_eig = False

        # ---- Extract unique frame IDs and compute interpolation weights ----
        frame_ids_set = set()
        for cam in cameras:
            fid_str = cam.image_name.split('_')[0]
            frame_ids_set.add(int(fid_str))
        sorted_frame_ids = sorted(frame_ids_set)
        N_frames = len(sorted_frame_ids)
        print(f"Unique frames : {N_frames}")

        # Determine the interpolation range
        if has_frame_range:
            interp_min = args.frame_range[0]
            interp_max = args.frame_range[1]
        else:
            interp_min = sorted_frame_ids[0]
            interp_max = sorted_frame_ids[-1]

        interp_weights = compute_interp_weights(interp_min, interp_max, args.interp_ratio)

        if is_interp:
            ramp = max(int((interp_max - interp_min + 1) * args.interp_ratio / 2), 1)
            w_min = min(interp_weights.values())
            w_max = max(interp_weights.values())
            print(f"  Ramp frames  : {ramp} up + {ramp} down (range [{interp_min}, {interp_max}])")
            print(f"  Weight range : [{w_min:.4f}, {w_max:.4f}]")

        if has_frame_range:
            START, END = args.frame_range
            n_range_frames = sum(1 for fid in sorted_frame_ids if START <= fid <= END)
            if n_range_frames == 0:
                print(f"  WARNING: frame_range [{START}, {END}] does not overlap "
                      f"with dataset frame range [{sorted_frame_ids[0]}, {sorted_frame_ids[-1]}]. "
                      f"No output will be saved.")
            else:
                print(f"  Saving frames: [{START}, {END}] ({n_range_frames}/{N_frames} frames)")

        # ---- Step 3: Write shifted ego_pose files ----
        print(f"\n[Step 3] Writing shifted ego_pose files...")
        frame_poses, cam_poses = load_ego_poses(args.source_path)
        ego_out = os.path.join(save_dir, 'ego_pose')
        os.makedirs(ego_out, exist_ok=True)

        for stem, pose in frame_poses.items():
            fid = int(stem)
            w = interp_weights.get(fid, 1.0)
            frame_shift = (w * shift_vec_np).astype(np.float64) if is_interp else shift_vec_np
            shifted = shift_pose_4x4(pose, frame_shift)
            np.savetxt(os.path.join(ego_out, f'{stem}.txt'), shifted, fmt='%.16e')

        for stem, pose in cam_poses.items():
            frame_id_str = stem.rsplit('_', 1)[0]
            fid = int(frame_id_str)
            w = interp_weights.get(fid, 1.0)
            frame_shift = (w * shift_vec_np).astype(np.float64) if is_interp else shift_vec_np
            R_ego = frame_poses[frame_id_str][:3, :3] if frame_id_str in frame_poses else None
            shifted = shift_pose_4x4(pose, frame_shift, R=R_ego)
            np.savetxt(os.path.join(ego_out, f'{stem}.txt'), shifted, fmt='%.16e')

        print(f"  wrote {len(frame_poses) + len(cam_poses)} shifted pose files to {ego_out}")

        # ---- Step 3.2: Copy timestamps ----
        import shutil
        for ts_name in ['timestamps.json', 'timestamps_specific.json']:
            ts_src = os.path.join(args.source_path, ts_name)
            if os.path.exists(ts_src):
                shutil.copy2(ts_src, os.path.join(save_dir, ts_name))
                print(f"  copied {ts_name} -> {save_dir}")

        # Copy intrinsics and extrinsics folders
        for folder in ['intrinsics', 'extrinsics']:
            src_dir = os.path.join(args.source_path, folder)
            dst_dir = os.path.join(save_dir, folder)
            if os.path.isdir(src_dir):
                if os.path.exists(dst_dir):
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                print(f"  copied {folder}/ -> {save_dir}")

        # ---- Step 3.3: Shift track objects ----
        track_src = os.path.join(args.source_path, 'track')
        track_out = os.path.join(save_dir, 'track')
        if os.path.isdir(track_src):
            os.makedirs(track_out, exist_ok=True)

            if is_interp:
                print(f"\n[Step 3.3] Skipping track_info.txt shift "
                      f"(interpolation mode: shift varies per frame)")
            else:
                print(f"\n[Step 3.3] Shifting track objects...")
                # Shift track_info.txt -- columns 7,8,9 are box_center_x,y,z in ego frame
                track_info_path = os.path.join(track_src, 'track_info.txt')
                if os.path.exists(track_info_path):
                    with open(track_info_path, 'r') as f:
                        lines = f.readlines()
                    header = lines[0]
                    shifted_lines = [header]
                    for line in lines[1:]:
                        parts = line.strip().split()
                        if len(parts) >= 10:
                            cx = float(parts[7]) - shift_vec_np[0]
                            cy = float(parts[8]) - shift_vec_np[1]
                            cz = float(parts[9]) - shift_vec_np[2]
                            parts[7] = f'{cx:.16e}'
                            parts[8] = f'{cy:.16e}'
                            parts[9] = f'{cz:.16e}'
                        shifted_lines.append(' '.join(parts) + '\n')
                    with open(os.path.join(track_out, 'track_info.txt'), 'w') as f:
                        f.writelines(shifted_lines)
                    print(f"  shifted track_info.txt -> {track_out}")

            # Copy track_camera_vis.json (visibility data, no spatial info to shift)
            vis_src = os.path.join(track_src, 'track_camera_vis.json')
            if os.path.exists(vis_src):
                shutil.copy2(vis_src, os.path.join(track_out, 'track_camera_vis.json'))
                print(f"  copied track_camera_vis.json -> {track_out}")
        else:
            print(f"  No track/ folder found, skipping")

        # ---- Step 3.5: Load pointcloud and build multi-view colors ----
        print(f"\n[Step 3.5] Loading pointcloud and building multi-view colors...")
        npz_path = os.path.join(args.source_path, 'pointcloud.npz')
        camera_projection = None
        pc_multiview = {}  # {frame_idx: {'xyz': ndarray(N,3), 'frame_cams': {cid: (c2w,K,H,W,gt)}, 'color_xform': ...}}
        global_fallback_lut = {}  # {(fb_cam, render_cam): lut}
        if os.path.exists(npz_path):
            pc_data = np.load(npz_path, allow_pickle=True)
            pc_orig = pc_data['pointcloud'].item()
            if 'camera_projection' in pc_data:
                camera_projection = pc_data['camera_projection'].item()

            # ---- Gather original camera data grouped by frame ----
            # For each camera, extract: cam_id, ego_pose, c2w, K, H, W, GT image
            frame_cam_data = {}  # {frame_idx: [(cam_id, ego_pose, c2w, K, H, W, gt_img), ...]}
            for cam in cameras:
                fid = int(cam.image_name.split('_')[0])
                if has_frame_range and not (START <= fid <= END):
                    continue  # only need cameras within frame_range for coloring
                cid = int(cam.image_name.split('_')[-1])
                # Load GT image
                gt_img = None
                for ext in ['.png', '.jpg']:
                    gt_path = os.path.join(args.source_path, 'images',
                                           f'{fid:06d}_{cid:02d}{ext}')
                    if os.path.exists(gt_path):
                        gt_img = cv2.imread(gt_path)
                        if gt_img is not None:
                            break
                if gt_img is None:
                    continue
                ego_pose = cam.ego_pose.cpu().numpy() if torch.is_tensor(cam.ego_pose) \
                    else cam.ego_pose
                c2w = cam.get_extrinsic()
                K_np = cam.get_intrinsic()
                H, W = cam.image_height, cam.image_width
                frame_cam_data.setdefault(fid, []).append(
                    (cid, ego_pose, c2w, K_np, H, W, gt_img))

            # ---- Build per-frame data for on-the-fly torch warped color rendering ----
            # Store raw camera data (c2w, K, GT image) per camera per frame.
            # Color extraction is done on-the-fly using torch projection during rendering.
            # Color balance transforms are still pre-computed here (need point correspondences).
            frames_to_color = sorted(set(pc_orig.keys()) & set(frame_cam_data.keys()))
            print(f"  Building per-frame data for {len(frames_to_color)} frames...")
            for frame_idx in tqdm(frames_to_color, desc="  Per-frame data",
                                  leave=False):
                pts_ego = pc_orig[frame_idx].astype(np.float32)
                pts_torch = torch.from_numpy(pts_ego).to('cuda')

                # Store raw per-camera data + temporary sampled colors for color_xform
                frame_cams = {}
                cam_views_for_xform = {}

                for (cid, ego_pose_np, c2w, K_np, H, W, gt_img) \
                        in frame_cam_data[frame_idx]:
                    # Store raw data for on-the-fly rendering
                    frame_cams[cid] = (c2w.copy(), K_np.copy(), H, W, gt_img.copy())

                    # Project points to this camera for color_xform computation
                    cam_pose = torch.from_numpy(c2w).float().to('cuda')
                    ego_pose_t = torch.from_numpy(ego_pose_np).float().to('cuda')
                    K_t = torch.from_numpy(K_np).float().to('cuda')

                    indices, xs, ys = get_surviving_pixels(
                        pts_torch, ego_pose_t, cam_pose, K_t, H, W, 'cuda',
                        filter_method=args.filter_method,
                        surface_ratio=args.bilateral_surface_ratio)

                    if len(indices) > 0:
                        gt_h, gt_w = gt_img.shape[:2]
                        if gt_w != W or gt_h != H:
                            scale_x = gt_w / W
                            scale_y = gt_h / H
                            xs_gt = np.clip((xs.astype(np.float32) * scale_x).astype(np.int32), 0, gt_w - 1)
                            ys_gt = np.clip((ys.astype(np.float32) * scale_y).astype(np.int32), 0, gt_h - 1)
                        else:
                            xs_gt, ys_gt = xs, ys
                        colors = gt_img[ys_gt, xs_gt]  # BGR uint8 (M, 3)
                        cam_data = np.zeros((len(indices), 4), dtype=np.float32)
                        cam_data[:, 0] = indices.astype(np.float32)
                        cam_data[:, 1:4] = colors.astype(np.float32)
                        cam_views_for_xform[cid] = cam_data

                # ---- Compute per-camera-pair color balance transforms ----
                color_xform = {}
                cam_ids_list = sorted(cam_views_for_xform.keys())
                cb_method = args.color_balance
                for src in cam_ids_list:
                    color_xform[src] = {}
                    sv = cam_views_for_xform[src]
                    for dst in cam_ids_list:
                        if src == dst:
                            color_xform[src][dst] = (cb_method, None)
                            continue
                        dv = cam_views_for_xform[dst]
                        common, si, di = np.intersect1d(
                            sv[:, 0].astype(np.int64), dv[:, 0].astype(np.int64),
                            return_indices=True)
                        if len(common) < 50:
                            color_xform[src][dst] = (cb_method, None)
                            continue
                        src_c = sv[si, 1:4].astype(np.float64)
                        dst_c = dv[di, 1:4].astype(np.float64)
                        N_common = src_c.shape[0]
                        if N_common > 10000:
                            idx_sub = np.random.RandomState(42).choice(N_common, 10000, replace=False)
                            src_c, dst_c = src_c[idx_sub], dst_c[idx_sub]

                        if cb_method == 'linear':
                            gains = np.zeros(3, dtype=np.float32)
                            offsets = np.zeros(3, dtype=np.float32)
                            for ch in range(3):
                                s = src_c[:, ch]
                                d = dst_c[:, ch]
                                A = np.stack([s, np.ones_like(s)], axis=1)
                                sol, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
                                gains[ch] = float(sol[0])
                                offsets[ch] = float(sol[1])
                            params = (gains, offsets)

                        elif cb_method == 'matrix':
                            gains = np.zeros((3, 3), dtype=np.float32)
                            offsets = np.zeros(3, dtype=np.float32)
                            for ch in range(3):
                                d = dst_c[:, ch]
                                A = np.concatenate([src_c, np.ones((src_c.shape[0], 1), dtype=np.float64)], axis=1)
                                sol, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
                                gains[ch, :] = sol[:3]
                                offsets[ch] = float(sol[3])
                            params = (gains, offsets)

                        elif cb_method == 'gamma':
                            from scipy.optimize import least_squares
                            gains = np.zeros(3, dtype=np.float32)
                            gammas = np.zeros(3, dtype=np.float32)
                            offsets = np.zeros(3, dtype=np.float32)
                            for ch in range(3):
                                s = src_c[:, ch].clip(1.0, 255.0)
                                d = dst_c[:, ch]
                                def cost(p):
                                    g, gm, o = p
                                    pred = g * np.power(s / 255.0, gm) * 255.0 + o
                                    return np.clip(pred, 0, 255) - d
                                res = least_squares(cost, [1.0, 1.0, 0.0],
                                                    bounds=([0.01, 0.2, -100], [5.0, 3.0, 100]),
                                                    max_nfev=200)
                                gains[ch] = float(res.x[0])
                                gammas[ch] = float(res.x[1])
                                offsets[ch] = float(res.x[2])
                            params = (gains, gammas, offsets)

                        color_xform[src][dst] = (cb_method, params)

                pc_multiview[frame_idx] = {
                    'xyz': pts_ego,
                    'frame_cams': frame_cams,
                    'color_xform': color_xform,
                }
                del pts_torch

            # ---- Build LUTs for fallback (side→forward) and partner (9↔10) ----
            print(f"  Building fallback color LUTs...")
            for src_c in [0, 1]:
                global_fallback_lut[src_c] = None
            # Build LUT for both fallback (0/1→9/10) and partner (9↔10)
            lut_pairs = []
            for r in [9, 10]:
                for f in [0, 1]:
                    lut_pairs.append((f, r, 'fallback'))
            lut_pairs.append((9, 10, 'partner'))
            lut_pairs.append((10, 9, 'partner'))
            for src_c, dst_c, _kind in lut_pairs:
                src_all = []
                tgt_all = []
                for fid in frames_to_color[:20]:  # sample frames for speed
                    fd = frame_cam_data[fid]
                    src_info = next((x for x in fd if x[0] == src_c), None)
                    tgt_info = next((x for x in fd if x[0] == dst_c), None)
                    if src_info is None or tgt_info is None:
                        continue
                    pts_t = torch.from_numpy(pc_orig[fid].astype(np.float32)).to('cuda')
                    data = {}
                    for ci, info in enumerate([src_info, tgt_info]):
                        cid, ep, c2w, K, H, W, gt = info
                        ep_t = torch.from_numpy(ep).float().to('cuda')
                        c2w_t = torch.from_numpy(c2w).float().to('cuda')
                        K_t = torch.from_numpy(K).float().to('cuda')
                        idx, xs, ys = get_surviving_pixels_torch(
                            pts_t, ep_t, c2w_t, K_t, H, W, 'cuda',
                            use_filter=True, filter_method=args.filter_method,
                            surface_ratio=args.bilateral_surface_ratio)
                        if len(idx) > 0:
                            gth, gtw = gt.shape[:2]
                            if gtw != W or gth != H:
                                xs_gt = (xs.float() * gtw / W).long().clamp(0, gtw-1)
                                ys_gt = (ys.float() * gth / H).long().clamp(0, gth-1)
                            else:
                                xs_gt, ys_gt = xs, ys
                            data[ci] = (idx, xs_gt, ys_gt, torch.from_numpy(gt.astype(np.float32)).to('cuda'))
                    del pts_t
                    if 0 not in data or 1 not in data:
                        continue
                    # Find co-visible points
                    common, si, ti = _torch_intersect1d(data[0][0], data[1][0])
                    if common.shape[0] < 20:
                        continue
                    si, ti = si.cpu().numpy(), ti.cpu().numpy()
                    # Extract 5×5 patches from GT images
                    half = 2
                    for i in range(min(len(si), 500)):
                        # Source (fallback) 5×5 patch
                        sx, sy = data[0][1][si[i]].item(), data[0][2][si[i]].item()
                        x0s, x1s = max(0, sx-half), min(data[0][3].shape[1], sx+half+1)
                        y0s, y1s = max(0, sy-half), min(data[0][3].shape[0], sy+half+1)
                        if x1s > x0s and y1s > y0s:
                            src_all.append(data[0][3][y0s:y1s, x0s:x1s].reshape(-1, 3))
                        # Target (render) 5×5 patch
                        tx, ty = data[1][1][ti[i]].item(), data[1][2][ti[i]].item()
                        x0t, x1t = max(0, tx-half), min(data[1][3].shape[1], tx+half+1)
                        y0t, y1t = max(0, ty-half), min(data[1][3].shape[0], ty+half+1)
                        if x1t > x0t and y1t > y0t:
                            tgt_all.append(data[1][3][y0t:y1t, x0t:x1t].reshape(-1, 3))
                if src_all and tgt_all:
                    src_t = torch.cat(src_all, dim=0)
                    tgt_t = torch.cat(tgt_all, dim=0)
                    if src_t.shape[0] >= 100:
                        lut = _build_per_channel_lut(src_t, tgt_t)
                        global_fallback_lut[(src_c, dst_c)] = lut
                        print(f"  LUT [{_kind}] cam{src_c}→cam{dst_c}: "
                              f"built ({src_t.shape[0]} samples)")
            torch.cuda.empty_cache()

            # Clean up (no longer needed)
            camera_projection = None
            pc_shifted = None
            print(f"  Built per-frame data for {len(pc_multiview)} frames")
        else:
            print(f"  WARNING: pointcloud.npz not found, skipping warped color")
            pc_multiview = {}
            pc_shifted = None

        lidar_depth_vis_dir = None  # disabled
        warped_color_dir = os.path.join(save_dir, 'warped_color')
        os.makedirs(warped_color_dir, exist_ok=True)
        gt_images_dir = os.path.join(save_dir, 'gt_images')
        os.makedirs(gt_images_dir, exist_ok=True)
        lidar_depth_dir = os.path.join(save_dir, 'lidar_depth')
        os.makedirs(lidar_depth_dir, exist_ok=True)

        # Downscale output dirs
        if args.downscale is not None:
            ds = args.downscale
            images_ds_dir = os.path.join(save_dir, f'images_ds{ds}')
            warped_ds_dir = os.path.join(save_dir, f'warped_color_ds{ds}')
            gt_ds_dir = os.path.join(save_dir, f'gt_images_ds{ds}')
            os.makedirs(images_ds_dir, exist_ok=True)
            os.makedirs(warped_ds_dir, exist_ok=True)
            os.makedirs(gt_ds_dir, exist_ok=True)
        else:
            images_ds_dir = warped_ds_dir = gt_ds_dir = None

        # ---- Step 4: Shift camera poses and render ----
        print(f"\n[Step 4] Rendering...")
        if is_interp:
            print(f"  Shift vector (ego frame): {shift_vec} x interp_weight")
        else:
            print(f"  Shift vector (ego frame): {shift_vec}")

        # camera_projection update is disabled (multi-view coloring supersedes it)
        do_projection = False

        # Reset camera_projection to invalid (-1) only if we'll update it
        if do_projection:
            for frame_idx in camera_projection:
                camera_projection[frame_idx][:] = -1

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            # ---- Determine per-frame shift ----
            frame_id_str = camera.image_name.split('_')[0]
            frame_idx = int(frame_id_str)
            w = interp_weights.get(frame_idx, 1.0)

            if is_interp:
                frame_shift = (w * shift_vec_np).astype(np.float64)
            else:
                frame_shift = shift_vec_np

            # ---- Check if this camera ID should be rendered ----
            cam_id_str = camera.image_name.split('_')[-1]
            cam_id = int(cam_id_str)
            if cam_id not in args.camera_ids:
                continue  # skip unwanted cameras entirely

            # ---- Check if this frame is within the output range ----
            if has_frame_range:
                START, END = args.frame_range
                if not (START <= frame_idx <= END):
                    continue  # skip frames outside range

            # ---- Shift camera pose and render ----
            c2w_orig = camera.get_extrinsic()
            R_ego = camera.ego_pose.cpu().numpy()[:3, :3]

            new_c2w = shift_pose_4x4(c2w_orig, frame_shift, R=R_ego)
            camera.set_extrinsic(new_c2w)

            if args.max_pixels > 0:
                saved_opacity = filter_large_gaussians(gaussians, camera, args.max_pixels)

            result = renderer.render_all(camera, gaussians)

            if args.max_pixels > 0:
                restore_gaussians(gaussians, saved_opacity)

            # ---- Save RGB image ----
            torchvision.utils.save_image(
                result['rgb'],
                os.path.join(images_dir, f'{camera.image_name}.jpg'))

            # ---- EIG maps ----
            if args.output_eig and I_train is not None:
                os.makedirs(_eig_dir, exist_ok=True)
                try:
                    eig_res = _render_eig_map(camera, gaussians, renderer,
                                              I_train, model_ranges,
                                              sky_eig_mode=args.sky_eig_mode)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); gc.collect(); eig_res = None
                if eig_res is not None:
                    torch.save(eig_res['gain_map'].cpu(),
                               os.path.join(_eig_dir, f'{camera.image_name}_gain.pt'))
                    _gm = eig_res['gain_map_vis'].cpu().numpy()
                    _gm_norm = (_gm - _gm.min()) / (_gm.max() - _gm.min() + 1e-8)
                    _gm_color = cv2.applyColorMap((_gm_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    cv2.imwrite(os.path.join(_eig_dir, f'{camera.image_name}_gain.png'), _gm_color)
                    _um = eig_res['uncertainty_map'].cpu().numpy()
                    _um_norm = (_um - _um.min()) / (_um.max() - _um.min() + 1e-8)
                    _um_color = cv2.applyColorMap((_um_norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
                    cv2.imwrite(os.path.join(_eig_dir, f'{camera.image_name}_uncertainty.png'), _um_color)
                    del eig_res
                torch.set_grad_enabled(False); torch.cuda.empty_cache(); gc.collect()

            # ---- Warped color: on-the-fly torch projection ----
            # 1) project all points to the processing (shifted) camera,
            # 2) extract color from this cam's original GT image first
            #    (re-project through unshifted camera pose),
            # 3) fall back to other cameras for remaining points.
            pts = None; ego_pose_out = None; cam_shifted_out = None
            K_out = None; H_out = None; W_out = None; mv_out = None
            warped_img = None
            if pc_multiview is not None and frame_idx in pc_multiview:
                mv = pc_multiview[frame_idx]
                pts = torch.from_numpy(mv['xyz'].astype(np.float32)).to('cuda')

                ego_pose = camera.ego_pose
                cam_shifted = torch.from_numpy(new_c2w).float().to('cuda')
                K = camera.K
                H, W = camera.image_height, camera.image_width

                # Partner camera: cam9↔cam10 share surviving points
                _partner = None
                _partner_cid = 9 if cam_id == 10 else (10 if cam_id == 9 else None)
                if _partner_cid is not None and _partner_cid in mv['frame_cams']:
                    pc2w, pK, pH, pW, _ = mv['frame_cams'][_partner_cid]
                    pnew_c2w = shift_pose_4x4(pc2w, frame_shift, R_ego)
                    _partner = (torch.from_numpy(pnew_c2w).float().to('cuda'),
                                torch.from_numpy(pK).float().to('cuda'), pH, pW)

                # Fallback priority: partner first (9↔10), then side camera
                _side = None
                if 'shift_left' in args.shift: _side = 0
                elif 'shift_right' in args.shift: _side = 1
                _fb_list = [c for c in [_partner_cid, _side] if c is not None]

                _topology_triangles = None
                _topology_front_count = 0
                if args.topology_visibility:
                    _topology_triangles, _topology_front_count = load_frame_topology(
                        frame_idx)
                    if _topology_front_count > pts.shape[0]:
                        raise RuntimeError(
                            f"frame {frame_idx}: front LiDAR has "
                            f"{_topology_front_count} points, combined cloud has "
                            f"{pts.shape[0]}")

                warped_img, fallback_masks, warp_idx, warp_xs, warp_ys, warp_stats = render_warped_color_image_v2(
                    pts, ego_pose, cam_shifted, K, H, W, cam_id, 'cuda',
                    mv['frame_cams'],
                    filter_method=args.filter_method,
                    surface_ratio=args.bilateral_surface_ratio,
                    target_filter_win=args.target_filter_win,
                    near_splat_depth=args.near_splat_depth,
                    near_splat_win=args.near_splat_win,
                    fallback_cam=_fb_list,
                    partner_cam=_partner if (
                        args.enable_partner_geometry and
                        cam_id in args.partner_geometry_camera_ids) else None,
                    topology_triangles=_topology_triangles,
                    topology_front_count=_topology_front_count,
                    topology_tolerance=args.topology_visibility_tolerance)

                print(f"[WarpStats] frame={frame_idx:06d} cam={cam_id:02d} "
                      f"primary={warp_stats['primary_count']} "
                      f"partner_added={warp_stats['partner_added']} "
                      f"topology_removed={warp_stats['topology_removed']} "
                      f"final={warp_stats['final_count']}")

                pass  # partner LUT removed — apply only via fallback LUT

                # Save for downscale section
                ego_pose_out = ego_pose; cam_shifted_out = cam_shifted
                K_out = K; H_out = H; W_out = W; mv_out = mv

                # Apply per-source LUT to fallback pixels
                for _fc, _fbm in fallback_masks.items():
                    _lut_key = (_fc, cam_id)
                    if _fbm.any() and global_fallback_lut.get(_lut_key) is not None:
                        lut = global_fallback_lut[_lut_key]
                        warped_img[_fbm] = _apply_lut_np(warped_img[_fbm], lut)

                cv2.imwrite(os.path.join(warped_color_dir, f'{camera.image_name}.png'), warped_img)

            # ---- Lidar depth: reuse the target-view surviving points ----
            if pts is not None and warp_idx is not None:
                mask = np.zeros((H, W), dtype=bool)
                depth = np.zeros((H, W), dtype=np.float32)
                indices_d = warp_idx.cpu().numpy()
                xs_d = warp_xs.cpu().numpy()
                ys_d = warp_ys.cpu().numpy()
                if len(indices_d) > 0:
                    pts_np = pts.cpu().numpy()
                    R_ego_np = ego_pose[:3, :3].cpu().numpy()
                    t_ego_np = ego_pose[:3, 3].cpu().numpy()
                    R_cam_np = cam_shifted[:3, :3].cpu().numpy()
                    t_cam_np = cam_shifted[:3, 3].cpu().numpy()
                    pts_world = pts_np @ R_ego_np.T + t_ego_np
                    pts_cam = (pts_world - t_cam_np) @ R_cam_np
                    point_depths = pts_cam[indices_d, 2]
                    mask[ys_d, xs_d] = True
                    depth[ys_d, xs_d] = point_depths
                value_sparse = depth[mask].astype(np.float32)
                np.save(os.path.join(lidar_depth_dir, f'{camera.image_name}.npy'),
                        {'mask': mask, 'value': value_sparse})

            # ---- Copy GT image for original_video ----
            cam_id_str = camera.image_name.split('_')[-1]
            gt_src = os.path.join(
                args.source_path, 'images', f'{camera.image_name}.png')
            if not os.path.exists(gt_src):
                gt_src = os.path.join(
                    args.source_path, 'images', f'{camera.image_name}.jpg')
            if os.path.exists(gt_src):
                shutil.copy2(gt_src,
                             os.path.join(gt_images_dir,
                                          f'{camera.image_name}.jpg'))

            # ---- Downscale output ----
            if args.downscale is not None:
                ds = args.downscale
                H_ds, W_ds = H // ds, W // ds

                # RGB: resize from rendered result
                rgb_np = (result['rgb'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rgb_ds = cv2.resize(rgb_np, (W_ds, H_ds))
                cv2.imwrite(os.path.join(images_ds_dir, f'{camera.image_name}.jpg'),
                            rgb_ds[:, :, ::-1])  # RGB -> BGR

                # Warped color: downscale preferring colored pixels over black
                if warped_img is not None:
                    warped_ds = np.zeros((H_ds, W_ds, 3), dtype=np.uint8)
                    has_color = (warped_img.sum(axis=2) > 0)  # (H, W)
                    for dy in range(ds):
                        for dx in range(ds):
                            # For each offset, pick first non-black source pixel
                            src = warped_img[dy::ds, dx::ds, :]  # (H_ds, W_ds, 3)
                            mask = has_color[dy::ds, dx::ds]      # (H_ds, W_ds)
                            empty = ~(warped_ds.sum(axis=2) > 0)
                            fill = mask & empty
                            warped_ds[fill] = src[fill]
                    cv2.imwrite(os.path.join(warped_ds_dir, f'{camera.image_name}.png'), warped_ds)

                # GT: resize
                if os.path.exists(gt_src):
                    gt_img = cv2.imread(gt_src)
                    gt_ds = cv2.resize(gt_img, (W_ds, H_ds))
                    cv2.imwrite(os.path.join(gt_ds_dir, f'{camera.image_name}.jpg'), gt_ds)

            # ---- Camera projection update (only when saving pointcloud.npz) ----
            if do_projection and frame_idx in camera_projection:
                cam_suffix = camera.image_name.split('_')[-1]
                cam_idx = int(cam_suffix)
                K_np = camera.get_intrinsic()
                ego_pose_np = camera.ego_pose.cpu().numpy()
                ego2cam = np.linalg.inv(c2w_orig) @ ego_pose_np

                pts_h = np.concatenate(
                    [pc_shifted[frame_idx][:, :3],
                     np.ones((pc_shifted[frame_idx].shape[0], 1))],
                    axis=-1)
                pts_cam = ego2cam @ pts_h.T

                valid_z = pts_cam[2, :] > 0
                pts_img = K_np @ pts_cam[:3, :]
                pts_img[:2, :] /= pts_img[2, :]

                in_bounds = (
                    valid_z &
                    (pts_img[0, :] >= 0) & (pts_img[0, :] < W) &
                    (pts_img[1, :] >= 0) & (pts_img[1, :] < H)
                )
                update_idx = np.where(in_bounds)[0]
                if len(update_idx) > 0:
                    x_coords = np.clip(
                        np.round(pts_img[0, update_idx]).astype(np.int32), 0, W - 1)
                    y_coords = np.clip(
                        np.round(pts_img[1, update_idx]).astype(np.int32), 0, H - 1)
                    cam_proj = camera_projection[frame_idx]
                    cam_proj[update_idx, 0] = cam_idx
                    cam_proj[update_idx, 1] = x_coords
                    cam_proj[update_idx, 2] = y_coords

        # ---- Generate JSON index per camera ----
        import json as _json
        img_begin = args.frame_range[0] if has_frame_range else sorted_frame_ids[0]
        img_end = args.frame_range[1] if has_frame_range else sorted_frame_ids[-1]

        # Extract clip name from source_path
        _src_parts = os.path.normpath(args.source_path).split('/')
        _clip_name = next((p for p in _src_parts if p.startswith('sample_')), '')

        # Build lookup from ori_csv: (clip_name, camera_idx) -> json_path
        _ori_lookup = {}
        if os.path.exists(args.ori_csv):
            with open(args.ori_csv, 'r') as _f:
                _header = _f.readline()
                for _line in _f:
                    _parts = _line.strip().split(',')
                    if len(_parts) >= 4:
                        _ori_lookup[(_parts[1], int(_parts[2]))] = _parts[3]

        def _write_cam_json(cid, images_d, gt_d, warp_d, ds_suffix):
            render_list = []; gt_list = []; warp_list = []; eig_list = []; depth_list = []
            for fid in range(img_begin, img_end + 1):
                name = f'{fid:06d}_{cid:02d}'
                render_list.append(os.path.join(images_d, f'{name}.jpg'))
                gt_list.append(os.path.join(gt_d, f'{name}.jpg'))
                warp_list.append(os.path.join(warp_d, f'{name}.png'))
                if not ds_suffix:  # lidar depth only for full res
                    depth_list.append(os.path.join(lidar_depth_dir, f'{name}.npy'))
                if args.output_eig:
                    eig_list.append(os.path.join(_eig_dir, f'{name}_gain.pt'))
            # Look up original json
            ori_json = ''
            caption = ''
            ori_key = (_clip_name, cid)
            if ori_key in _ori_lookup:
                ori_json = _ori_lookup[ori_key]
                if os.path.exists(ori_json):
                    try:
                        with open(ori_json, 'r') as _of:
                            _oj = _json.load(_of)
                            caption = _oj.get('caption', '')
                    except Exception:
                        pass
            j = {
                'label': 'infer',
                'clip_name': _clip_name,
                'camera_idx': cid,
                'image_idx_begin': img_begin,
                'image_idx_end': img_end,
                'shift': shift_vec,
                'shift_type': shift_name,
                'interpolation_ratio': args.interp_ratio,
                'downscale': args.downscale if ds_suffix else None,
                'ori_json': ori_json,
                'caption': caption,
                'render_image': render_list,
                'gt_image': gt_list,
                'warp_color_image': warp_list,
                'lidar_depth': depth_list,
                'eig_gain_pt': eig_list,
            }
            json_path = os.path.join(save_dir, f'cam{cid:02d}{ds_suffix}.json')
            with open(json_path, 'w') as fp:
                _json.dump(j, fp, indent=2)
            print(f"  Wrote {json_path}")
            return json_path

        for cid in args.camera_ids:
            _write_cam_json(cid, images_dir, gt_images_dir, warped_color_dir, '')
        if args.downscale is not None:
            ds = args.downscale
            for cid in args.camera_ids:
                _write_cam_json(cid, images_ds_dir, gt_ds_dir, warped_ds_dir, f'_ds{ds}')

        # ---- Step 6: Save shifted pointcloud ----
        if is_interp:
            print(f"\n[Step 6] Skipping pointcloud.npz "
                  f"(interpolation mode: shift varies per frame)")
        elif has_frame_range:
            print(f"\n[Step 6] Skipping pointcloud.npz "
                  f"(frame_range: camera_projection is incomplete)")
        elif pc_shifted is not None and not args.filter_camera_rejected:
            print(f"\n[Step 6] Saving shifted pointcloud...")
            camera_projection_int = {
                k: v.astype(np.int16) for k, v in camera_projection.items()
            } if camera_projection is not None else None
            npz_out = os.path.join(save_dir, 'pointcloud.npz')
            np.savez_compressed(npz_out, pointcloud=pc_shifted,
                                camera_projection=camera_projection_int)
            print(f"  Saved -> {npz_out}")

        print(f"\nDone! Output saved to {save_dir}")


if __name__ == '__main__':
    main()
