#!/usr/bin/env python3
"""
render_eig_shift.py - Render novel trajectory views (shifted camera poses) with EIG maps.

Based on:
  - render_eig.py (FisherRF EIG computation)
  - render_shifted_v2.py (novel view rendering via pose shifting)

Workflow:
  Phase 1: Accumulate Fisher information H_total from original training views
  Phase 2: Compute I_train = 1 / (H_total + lambda)
  Phase 3: Shift camera poses, render novel views + EIG gain maps

Usage:
  python render_eig_shift.py \
      --source_path /path/to/dataset \
      --model_path /path/to/model \
      --output_dir /path/to/output \
      --shift shift_right \
      --sky_eig_mode zero \
      --max_pixels 100
"""

import os
import sys
import argparse
import numpy as np
import torch
import gc
from tqdm import tqdm
from glob import glob
from typing import Dict, Tuple, Optional

# ==============================================================================
# Shift definitions
# Coordinate convention: x=right, y=forward, z=up
# ==============================================================================
SHIFT_TYPES = {
    'shift_right':   [ 3.0, 0.0, 0.0],
    'shift_left':    [-3.0, 0.0, 0.0],
    'shift_up':      [ 0.0, 0.0, 2.0],
    'shift_forward': [ 0.0, 3.0, 0.0],
    'shift_none': [ 0.0, 0.0, 0.0],
}


def shift_pose_4x4(pose_4x4, shift_xyz, R=None):
    """Apply shift_xyz in the local coordinate frame to a 4x4 pose matrix."""
    out = pose_4x4.copy()
    if R is None:
        R = out[:3, :3]
    out[:3, 3] += R @ np.asarray(shift_xyz, dtype=out.dtype)
    return out


def filter_large_gaussians(gaussians, camera, max_pixels=100):
    """Hide gaussians whose estimated 2D projected radius exceeds max_pixels."""
    from lib.utils.general_utils import quaternion_to_matrix

    fx = camera.K[0, 0].item()
    cam_center = camera.camera_center
    R_w2c = torch.from_numpy(camera.R.T).float().to(cam_center.device)

    saved = {}
    for model_name in gaussians.model_name_id.keys():
        model = getattr(gaussians, model_name)
        xyz = model.get_xyz
        if len(xyz) == 0:
            continue
        scaling = model.get_scaling
        rots = model.get_rotation

        vec_w = xyz - cam_center.unsqueeze(0)
        z_cam = vec_w.norm(dim=1).clamp(min=0.1)

        R_local = quaternion_to_matrix(rots)
        R_cam = R_w2c.unsqueeze(0) @ R_local

        s2 = scaling * scaling
        cov_xx = (R_cam[:, 0, :] * s2 * R_cam[:, 0, :]).sum(dim=1)
        cov_yy = (R_cam[:, 1, :] * s2 * R_cam[:, 1, :]).sum(dim=1)
        cov_xy = (R_cam[:, 0, :] * s2 * R_cam[:, 1, :]).sum(dim=1)

        trace = cov_xx + cov_yy
        disc = torch.sqrt((cov_xx - cov_yy).square() + 4.0 * cov_xy.square() + 1e-12)
        lambda_max = 0.5 * (trace + disc)
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
    """Restore gaussian opacities after filtering."""
    for model_name, data in saved.items():
        model = getattr(gaussians, model_name)
        model._opacity.data[data['mask']] = data['opacity']


def load_ego_poses(source_dir):
    """Load frame and camera ego poses from source directory."""
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


# ==============================================================================
# EIG computation functions (adapted from render_eig.py)
# ==============================================================================

def zero_gaussian_grads(pc):
    """Zero gradients on all leaf parameters in the StreetGaussianModel."""
    for model_name in pc.model_name_id.keys():
        model = getattr(pc, model_name)
        for attr in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity', '_semantic']:
            if hasattr(model, attr):
                param = getattr(model, attr)
                if isinstance(param, torch.Tensor) and param.grad is not None:
                    param.grad = None


def render_with_grad(viewpoint_camera, pc, renderer) -> Optional[Dict]:
    """Render a view with gradient tracking on all Gaussian parameters."""
    include_list = list(set(pc.model_name_id.keys()))
    pc.set_visibility(include_list)
    pc.parse_camera(viewpoint_camera)

    num_gaussians = pc.num_gaussians
    if num_gaussians == 0:
        return None

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    means3D.retain_grad()
    opacity.retain_grad()
    scales.retain_grad()
    rotations.retain_grad()
    shs.retain_grad()

    from lib.utils.camera_utils import make_rasterizer
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    scaling_modifier = renderer.cfg.scaling_modifier
    rasterizer = make_rasterizer(viewpoint_camera, pc.max_sh_degree, bg_color, scaling_modifier)

    screenspace_points = torch.zeros((num_gaussians, 3), requires_grad=True, device="cuda")
    screenspace_points.retain_grad()

    rendered_color, _, rendered_depth, rendered_acc, _ = rasterizer(
        means3D=means3D, means2D=screenspace_points, opacities=opacity,
        shs=shs, colors_precomp=None, scales=scales, rotations=rotations,
        cov3D_precomp=None, semantics=None,
    )

    return {
        "rgb": rendered_color,
        "acc": rendered_acc,
        "params": {
            'means': means3D, 'rotations': rotations, 'scales': scales,
            'opacities': opacity, 'shs': shs,
        },
        "ranges": dict(pc.graph_gaussian_range),
    }


def extract_per_gaussian_grad(params, ranges) -> Dict[str, torch.Tensor]:
    """After backward(), extract per-Gaussian gradient magnitude."""
    H_view = {}
    for model_name, (start, end) in ranges.items():
        n_model = end - start
        h = torch.zeros(n_model, device="cuda")
        for param_tensor in params.values():
            if param_tensor.grad is not None:
                g = param_tensor.grad.detach().abs()
                if g.dim() >= 2:
                    g = g.reshape(g.shape[0], -1).sum(dim=1)
                h += g[start:end]
        H_view[model_name] = h
    return H_view


def render_and_backward(viewpoint_camera, pc, renderer) -> Optional[Dict[str, torch.Tensor]]:
    """Render + backprop unit gradient, return per-model H contributions."""
    result = render_with_grad(viewpoint_camera, pc, renderer)
    if result is None:
        zero_gaussian_grads(pc)
        return None

    rgb = result["rgb"]
    params = result["params"]
    ranges = result["ranges"]

    grad_tensor = torch.ones_like(rgb)
    rgb.backward(gradient=grad_tensor, retain_graph=False)

    H = extract_per_gaussian_grad(params, ranges)

    del result
    zero_gaussian_grads(pc)
    torch.cuda.empty_cache()

    return H


def compute_I_train(H_accum: Dict[str, torch.Tensor], reg_lambda: float = 1e-6):
    """Concatenate per-model H and compute I_train = 1 / (H_full + lambda)."""
    model_ranges = {}
    offset = 0
    h_parts = []
    for model_name, h in H_accum.items():
        n = h.shape[0]
        model_ranges[model_name] = (offset, offset + n)
        h_parts.append(h)
        offset += n

    if len(h_parts) == 0:
        return None, None, None, {}

    H_full = torch.cat(h_parts, dim=0)
    I_train = torch.reciprocal(H_full + reg_lambda)
    I_train_sqrt = torch.sqrt(I_train)
    return H_full, I_train, I_train_sqrt, model_ranges


def render_eig_map(viewpoint_camera, pc, renderer, I_train, model_ranges,
                   sky_eig_mode="zero") -> Optional[Dict[str, torch.Tensor]]:
    """
    For a single view: render + backprop to get H_view, compute
    I_acq = H_view * I_train, splat to pixel space.
    """
    # Step 1: Render + backprop to get H_view
    result = render_with_grad(viewpoint_camera, pc, renderer)
    if result is None:
        return None

    rgb = result["rgb"].detach().clone()
    acc = result["acc"].detach().clone()
    params = result["params"]
    ranges = result["ranges"]

    grad_tensor = torch.ones_like(result["rgb"])
    result["rgb"].backward(gradient=grad_tensor, retain_graph=False)

    H_view_per_model = extract_per_gaussian_grad(params, ranges)
    N_total = params['means'].shape[0]

    # Move H_view to CPU to free GPU memory before second render
    H_view_per_model_cpu = {k: v.cpu() for k, v in H_view_per_model.items()}
    ranges_copy = dict(ranges)

    # Aggressively free ALL GPU memory from first render
    del result, params, H_view_per_model
    zero_gaussian_grads(pc)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # Step 2: Compute per-Gaussian EIG values
    H_view_full = torch.zeros(N_total, device="cuda")
    for model_name, (view_start, view_end) in ranges_copy.items():
        if model_name in H_view_per_model_cpu:
            H_view_full[view_start:view_end] = H_view_per_model_cpu[model_name].cuda()
    del H_view_per_model_cpu

    I_train_view = torch.zeros(N_total, device="cuda")
    I_train_sqrt_view = torch.zeros(N_total, device="cuda")
    H_total_view = torch.zeros(N_total, device="cuda")
    for model_name, (global_start, global_end) in model_ranges.items():
        if model_name in ranges_copy:
            view_start, view_end = ranges_copy[model_name]
            I_train_view[view_start:view_end] = I_train[global_start:global_end]
            I_train_sqrt_view[view_start:view_end] = torch.sqrt(I_train[global_start:global_end])
            H_total_view[view_start:view_end] = (1.0 / I_train[global_start:global_end].clamp(min=1e-12)) - 1e-6

    I_acq = H_view_full * I_train_view
    hessian_color = torch.stack([H_total_view, I_train_sqrt_view, I_acq], dim=1)

    del H_view_full, I_train_view, I_train_sqrt_view, H_total_view, I_acq, ranges_copy

    # Step 3: Splat hessian_color to pixel space
    with torch.no_grad():
        include_list = list(set(pc.model_name_id.keys()))
        pc.set_visibility(include_list)
        pc.parse_camera(viewpoint_camera)
        rendered_hessian = renderer.render_kernel(viewpoint_camera, pc, override_color=hessian_color)
        rgb_hessian = rendered_hessian["rgb"]

    gain_map_raw = rgb_hessian[2:3, :, :].clone()
    uncertainty_map_full = rgb_hessian[0:1, :, :].clone()
    cov_map_full = rgb_hessian[1:2, :, :].clone()

    # Explicitly free render_kernel result to release CUDA memory
    del rendered_hessian, rgb_hessian
    del hessian_color
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    opacity_mask = (acc > 0.1).float()
    sky_mask = (acc <= 0.1).squeeze(0)

    gain_map_raw_2d = gain_map_raw.squeeze(0) * opacity_mask.squeeze(0)
    gain_map = gain_map_raw_2d / 1000.0          # for .pt save (FaithFusion scale)
    gain_map_vis = gain_map_raw_2d               # for visualization (wider range)
    uncertainty_map = uncertainty_map_full * opacity_mask
    cov_map = cov_map_full * opacity_mask

    del gain_map_raw, gain_map_raw_2d, uncertainty_map_full, cov_map_full, opacity_mask, acc

    if sky_eig_mode == "max" and sky_mask.any() and (gain_map > 0).any():
        max_gain = gain_map[~sky_mask].max()
        max_gain_vis = gain_map_vis[~sky_mask].max()
        gain_map[sky_mask] = max_gain
        gain_map_vis[sky_mask] = max_gain_vis
        uncertainty_map[:, sky_mask] = uncertainty_map[:, ~sky_mask].max()
        cov_map[:, sky_mask] = cov_map[:, ~sky_mask].max()

    torch.cuda.empty_cache()
    return {
        "rgb": rgb,
        "gain_map": gain_map,            # [H, W] for .pt save (/1000 scale)
        "gain_map_vis": gain_map_vis,    # [H, W] for visualization (raw scale)
        "uncertainty_map": uncertainty_map.squeeze(0),
        "cov_map": cov_map.squeeze(0),
    }


def save_tensor_as_colormap(tensor_2d, path, cmap='turbo', label='', vmin=None, vmax=None,
                           pct_low=2, pct_high=98):
    """Save a [H, W] float tensor as a colormapped PNG with adaptive-percentile colorbar."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    arr = tensor_2d.detach().cpu().numpy()

    pct_lo = max(0.0, min(100.0, pct_low))
    pct_hi = max(0.0, min(100.0, pct_high))

    if vmin is None:
        vmin = float(np.percentile(arr, pct_lo))
    if vmax is None:
        vmax = float(np.percentile(arr, pct_hi))
    if vmax - vmin < 1e-8:
        vmin, vmax = float(arr.min()), float(arr.max())
    if vmax - vmin < 1e-8:
        vmin, vmax = 0.0, 1.0

    normed = np.clip((arr - vmin) / (vmax - vmin), 0, 1)

    cmap_obj = plt.get_cmap(cmap)
    fig, (ax_img, ax_cbar) = plt.subplots(1, 2, figsize=(12, 6),
                                           gridspec_kw={'width_ratios': [20, 1]})
    ax_img.imshow(normed, cmap=cmap_obj, vmin=0, vmax=1)
    ax_img.axis('off')

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(ax_cbar, cmap=cmap_obj, norm=norm, orientation='vertical')
    if label:
        cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=8)

    plt.tight_layout(pad=0.5)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Render shifted novel views with EIG maps.')
    parser.add_argument('--source_path', type=str, required=True,
                        help='Source dataset root')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Trained model directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--shift', type=str, default='shift_right',
                        choices=list(SHIFT_TYPES.keys()),
                        help='Predefined shift type')
    parser.add_argument('--shift_vector', type=float, nargs=3, default=None,
                        metavar=('DX', 'DY', 'DZ'),
                        help='Custom shift vector in meters')
    parser.add_argument('--sky_eig_mode', type=str, default='zero',
                        choices=['zero', 'max'],
                        help='Sky region EIG handling')
    parser.add_argument('--max_pixels', type=float, default=0.0,
                        help='Filter large 2D gaussians. 0=disabled, recommended 50-150')
    parser.add_argument('--loaded_iter', type=int, default=-1,
                        help='Checkpoint iteration (-1 = latest)')
    parser.add_argument('--num_observed', type=int, default=0,
                        help='Number of training views for H accumulation '
                             '(0=auto: first half, -1=all views)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID (overrides config gpus setting)')
    args = parser.parse_args()

    shift_vec = args.shift_vector if args.shift_vector is not None else SHIFT_TYPES[args.shift]
    shift_name = args.shift if args.shift_vector is None else 'custom_shift'
    shift_vec_np = np.array(shift_vec, dtype=np.float64)

    print("=" * 64)
    print(f"Source   : {args.source_path}")
    print(f"Model    : {args.model_path}")
    print(f"Output   : {args.output_dir}")
    print(f"Shift    : {shift_name} {list(shift_vec)}")
    print(f"Sky EIG  : {args.sky_eig_mode}")
    print(f"MaxPixels: {args.max_pixels}")
    print("=" * 64)

    # ---- Setup config ----
    config_yaml = os.path.join(args.model_path, 'configs', 'config_000000.yaml')
    if not os.path.exists(config_yaml):
        raise FileNotFoundError(f"Config not found: {config_yaml}")

    sys.argv = [
        'render_eig_shift',
        '--config', config_yaml,
        '--mode', 'trajectory',
        'source_path', args.source_path,
        'model_path', args.model_path,
    ]

    from lib.config import cfg
    cfg.mode = 'evaluate'
    if args.loaded_iter > 0:
        cfg.loaded_iter = args.loaded_iter

    from lib.utils.general_utils import safe_state
    if args.gpu is not None:
        import os as _os
        _os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        cfg.gpus = [-1]  # prevent parse_cfg from overriding our setting
    safe_state(cfg.eval.quiet)

    import torchvision
    from lib.models.street_gaussian_model import StreetGaussianModel
    from lib.models.street_gaussian_renderer import StreetGaussianRenderer
    from lib.datasets.dataset import Dataset
    from lib.models.scene import Scene

    # ---- Load dataset and model ----
    print("\n[1/4] Loading dataset and model...")
    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

    os.makedirs(args.output_dir, exist_ok=True)

    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    print(f"Loaded: {len(train_cameras)} train + {len(test_cameras)} test cameras")

    # ================================================================
    # Phase 1: Accumulate Fisher information on training views
    # ================================================================
    if args.num_observed == -1:
        num_observed = len(train_cameras)
    elif args.num_observed > 0:
        num_observed = args.num_observed
    else:
        num_observed = max(len(train_cameras) // 2, 1)
    observed_cams = train_cameras[:num_observed]
    print(f"\n[2/4] Phase 1: Accumulating H on {len(observed_cams)} training views")

    H_accum = {}
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    t_start.record()

    for idx, camera in enumerate(tqdm(observed_cams, desc="H accum")):
        H_view = render_and_backward(camera, gaussians, renderer)
        if H_view is not None:
            for name, h in H_view.items():
                if name not in H_accum:
                    H_accum[name] = h
                elif h.shape[0] == H_accum[name].shape[0]:
                    H_accum[name] += h
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache(); gc.collect()

    t_end.record()
    torch.cuda.synchronize()
    print(f"Phase 1 done in {t_start.elapsed_time(t_end)/1000:.1f}s, {len(H_accum)} sub-models accumulated")

    # ================================================================
    # Phase 2: Compute I_train
    # ================================================================
    print(f"\n[3/4] Phase 2: Computing I_train")
    H_full, I_train, I_train_sqrt, model_ranges = compute_I_train(H_accum)
    if H_full is None:
        print("ERROR: No Gaussians to compute EIG!")
        return
    print(f"Total: {H_full.shape[0]:,} Gaussians")

    # Free H_accum (no longer needed after I_train is computed)
    del H_accum
    gc.collect()
    torch.cuda.empty_cache()

    # ================================================================
    # Phase 3: Shift cameras and render EIG
    # ================================================================
    print(f"\n[4/4] Phase 3: Rendering shifted views + EIG")
    print(f"  Shift: {shift_vec}")

    cameras = train_cameras + test_cameras
    cameras = list(sorted(cameras, key=lambda x: x.id))

    images_dir = os.path.join(args.output_dir, 'images')
    eig_dir = os.path.join(args.output_dir, 'eig')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(eig_dir, exist_ok=True)

    t_start.record()

    batch_size = 8  # Process this many cameras before deep cleanup
    for idx, camera in enumerate(tqdm(cameras, desc="Shifted views")):
        # Shift camera pose
        c2w_orig = camera.get_extrinsic()
        R_ego = camera.ego_pose.cpu().numpy()[:3, :3]
        new_c2w = shift_pose_4x4(c2w_orig, shift_vec_np, R=R_ego)
        camera.set_extrinsic(new_c2w)

        # Filter large gaussians BEFORE EIG rendering too
        saved_opacity = None
        if args.max_pixels > 0:
            saved_opacity = filter_large_gaussians(gaussians, camera, args.max_pixels)

        # Render EIG map (this also gives us a raw RGB render)
        try:
            eig_result = render_eig_map(camera, gaussians, renderer, I_train, model_ranges,
                                        sky_eig_mode=args.sky_eig_mode)
        except torch.cuda.OutOfMemoryError:
            print(f"\n  WARNING: OOM on camera {idx} (id={camera.id}). "
                  f"Freeing cache and retrying...")
            # Aggressive cleanup and retry once
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
            try:
                eig_result = render_eig_map(camera, gaussians, renderer, I_train, model_ranges,
                                            sky_eig_mode=args.sky_eig_mode)
            except torch.cuda.OutOfMemoryError:
                print(f"  SKIPPED camera {idx}: OOM persists after retry")
                eig_result = None

        # Restore gaussians AFTER EIG rendering
        if saved_opacity is not None:
            restore_gaussians(gaussians, saved_opacity)

        # Save results
        if eig_result is not None:
            save_tensor_as_colormap(eig_result['gain_map_vis'],
                                    os.path.join(eig_dir, f'{camera.image_name}_gain.png'),
                                    cmap='turbo', label=f'EIG Gain ({shift_name})')
            save_tensor_as_colormap(eig_result['uncertainty_map'],
                                    os.path.join(eig_dir, f'{camera.image_name}_uncertainty.png'),
                                    cmap='plasma', label='Uncertainty')
            save_tensor_as_colormap(eig_result['cov_map'],
                                    os.path.join(eig_dir, f'{camera.image_name}_cov.png'),
                                    cmap='viridis', label='Covariance')
            torch.save(eig_result['gain_map'].cpu(),
                       os.path.join(eig_dir, f'{camera.image_name}_gain.pt'))

            # Save raw RGB from EIG rendering
            torchvision.utils.save_image(eig_result['rgb'].clamp(0, 1),
                                         os.path.join(images_dir, f'{camera.image_name}.jpg'))
            del eig_result

        # Aggressive cleanup between iterations
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        # Deep cleanup every batch: sync + empty cache + collect + reset pool
        if (idx + 1) % batch_size == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            # Periodically run ipc_collect to release inter-process memory
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()

    t_end.record()
    torch.cuda.synchronize()
    elapsed = t_start.elapsed_time(t_end) / 1000
    print(f"Phase 3 done in {elapsed:.1f}s ({len(cameras)/elapsed:.1f} it/s)")

    # Memory summary
    peak = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"\nDone! GPU peak: {peak:.2f} GB")
    print(f"RGB images: {images_dir}")
    print(f"EIG maps  : {eig_dir}")


if __name__ == '__main__':
    main()
