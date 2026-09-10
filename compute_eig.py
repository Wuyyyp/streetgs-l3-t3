#!/usr/bin/env python3
"""
Standalone script to compute EIG (Expected Information Gain) scores
from a trained street_gaussians model checkpoint.

This script loads a trained model checkpoint, uses the FaithFusion modified
CUDA rasterizer (with atomicAddSquare) to compute per-pixel uncertainty maps
following the Fisher Information / Expected Information Gain approach.

Usage:
    python compute_eig.py \
        --ckpt_path /path/to/iteration_10000.pth \
        --dataset_path /path/to/dataset \
        --cameras_json /path/to/cameras.json \
        --output_dir /path/to/output \
        --max_train_views 50 \
        --render_scale 0.5

Output:
    gain_{idx}.pt  - Per-pixel EIG gain maps (torch tensors)
    rgb_{idx}.png  - Rendered RGB images for reference
"""

import argparse
import json
import math
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Union

# Ensure the FaithFusion/diff directory is in the Python path so the modified
# CUDA rasterizer can be imported.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FAITHFUSION_DIFF_DIR = os.path.join(_SCRIPT_DIR, "FaithFusion-main", "diff")
if _FAITHFUSION_DIFF_DIR not in sys.path:
    sys.path.insert(0, _FAITHFUSION_DIFF_DIR)

# ---------------------------------------------------------------------------
# Direct import of the modified CUDA rasterizer (compiled from FaithFusion/diff)
# We avoid importing FaithFusion's Python modules because they pull in heavy
# dependencies (pytorch3d, gsplat, sklearn, etc.) that are not needed for EIG.
# Instead, we define the required dataclasses and render function inline.
# ---------------------------------------------------------------------------
from modified_diff_gaussian_rasterization import (
    GaussianRasterizer as ModifiedGaussianRasterizer,
    GaussianRasterizationSettings,
)
from diff_gaussian_rasterization import (
    GaussianRasterizer as StandardGaussianRasterizer,
    GaussianRasterizationSettings as StandardGaussianRasterizationSettings,
)
import nvdiffrast.torch as dr


# FaithFusion-compatible dataclass definitions (from models/gaussians/basics.py)
@dataclass
class dataclass_camera:
    camtoworlds: torch.Tensor
    camtoworlds_gt: torch.Tensor
    Ks: torch.Tensor
    H: int
    W: int


@dataclass
class dataclass_gs:
    _opacities: torch.Tensor
    _means: torch.Tensor
    _rgbs: Optional[torch.Tensor]
    _scales: torch.Tensor
    _quats: torch.Tensor
    _shs: torch.Tensor
    detach_keys: List[str]
    extras: Optional[Dict[str, torch.Tensor]] = None

    def set_grad_controller(self, detach_keys):
        self.detach_keys = detach_keys

    @property
    def opacities(self):
        if "activated_opacities" in self.detach_keys:
            return self._opacities.detach()
        return self._opacities

    @property
    def means(self):
        if "means" in self.detach_keys:
            return self._means.detach()
        return self._means

    @property
    def rgbs(self):
        if "colors" in self.detach_keys:
            return self._rgbs.detach()
        return self._rgbs

    @property
    def scales(self):
        if "scales" in self.detach_keys:
            return self._scales.detach()
        return self._scales

    @property
    def quats(self):
        if "quats" in self.detach_keys:
            return self._quats.detach()
        return self._quats

    @property
    def shs(self):
        if "shs" in self.detach_keys:
            return self._shs.detach()
        return self._shs


# FaithFusion modified_render (from models/trainers/origin_gs_renderer/__init__.py)
def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def _getProjectionMatrix(znear, zfar, K, H, W):
    """
    Build OpenGL-style perspective projection matrix matching
    street_gaussians' getProjectionMatrixK exactly.

    Args:
        znear: near plane distance
        zfar:  far plane distance
        K:     3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        H:     image height in pixels
        W:     image width in pixels
    """
    fx = K[0, 0].item() if isinstance(K, torch.Tensor) else K[0, 0]
    fy = K[1, 1].item() if isinstance(K, torch.Tensor) else K[1, 1]
    cx = K[0, 2].item() if isinstance(K, torch.Tensor) else K[0, 2]
    cy = K[1, 2].item() if isinstance(K, torch.Tensor) else K[1, 2]

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2 * fx / W
    P[1, 1] = 2 * fy / H
    P[0, 2] = -1 + 2 * (cx / W)
    P[1, 2] = -1 + 2 * (cy / H)
    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign
    return P


def modified_render(
    gs: dataclass_gs,
    cam: dataclass_camera,
    opaticy_mask: Optional[torch.Tensor],
    is_train_set: bool,
    override_color=None,
    zfar=0.01,     # NOTE: naming is swapped — caller should pass far plane here
    znear=100.0,   # NOTE: naming is swapped — caller should pass near plane here
):
    """
    Render using the modified CUDA rasterizer with atomicAddSquare for
    Hessian diagonal computation.

    Args:
        gs: Gaussian parameters (dataclass_gs)
        cam: Camera parameters (dataclass_camera)
        opaticy_mask: Optional per-Gaussian opacity mask
        is_train_set: If True, use SH rendering (for gradient computation)
        override_color: If provided, use as colors_precomp instead of SH
        zfar: Far plane distance (default 0.01, but naming is swapped)
        znear: Near plane distance (default 100.0, but naming is swapped)

    Returns:
        dict with keys: render, viewspace_points, visibility_filter, radii,
                        depth, pixel_gaussian_counter, opacity, params_output
    """
    # Screenspace points for densification gradient
    screenspace_points = torch.zeros_like(
        gs.means, dtype=gs.means.dtype, requires_grad=True, device="cuda"
    ) + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(focal2fov(cam.Ks[0, 0], int(cam.W)) * 0.5)
    tanfovy = math.tan(focal2fov(cam.Ks[1, 1], int(cam.H)) * 0.5)

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    world_view_transform = cam.camtoworlds.inverse().transpose(0, 1)
    projection_matrix = _getProjectionMatrix(
        znear=znear, zfar=zfar, K=cam.Ks, H=cam.H, W=cam.W
    ).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam.H),
        image_width=int(cam.W),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=1,
        campos=camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = ModifiedGaussianRasterizer(raster_settings=raster_settings)

    means3D = gs.means
    means2D = screenspace_points
    opacity = gs.opacities.squeeze()
    if opaticy_mask is not None:
        opacity = opacity * opaticy_mask
    opacity = opacity.unsqueeze(-1)

    cov3D_precomp = None
    scales = gs.scales
    rotations = gs.quats

    # SH or precomputed colors
    shs = None
    colors_precomp = None
    if is_train_set or override_color is None:
        # Use SH rendering — retain grad for Hessian computation
        shs = gs.shs
        shs.retain_grad()
    else:
        if override_color is not None:
            colors_precomp = override_color
        else:
            colors_precomp = gs.rgbs

    means3D.retain_grad()
    opacity.retain_grad()
    scales.retain_grad()
    rotations.retain_grad()

    # Rasterize
    rendered_image, depth, radii, pixel_gaussian_counter = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    params_output = {
        "means": means3D,
        "rotations": rotations,
        "scales": scales,
        "opacities": opacity,
        "shs": shs,
    }

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "pixel_gaussian_counter": pixel_gaussian_counter,
        "opacity": depth,  # named "opacity" but actually depth (for API compat)
        "params_output": params_output,
    }


def load_gaussian_params(ckpt_path: str, device: str = "cuda",
                         include_objects: bool = False, gaussian_subsample: int = 1):
    """
    Load Gaussian parameters from a street_gaussians checkpoint and merge them
    into a single dataclass_gs object.

    The checkpoint contains:
      - 'background': static background Gaussians (standard SH format, sh_degree=1)
      - 'obj_XXX': per-object tracked Gaussians (Fourier-encoded features, NOT
        standard SH — these require per-frame decoding via get_features_fourier())

    By default, only background is loaded because objects use a different feature
    representation (8-channel Fourier-encoded features vs 4-channel SH). Set
    include_objects=True to also load objects (padded/truncated to match).

    Each Gaussian group has: xyz, feature_dc, feature_rest, scaling, rotation,
    opacity, semantic, spatial_lr_scale, denom, max_radii2D.
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    all_xyz = []
    all_scales = []
    all_rotations = []
    all_opacities = []
    all_shs = []

    gaussian_keys = []
    for key, value in ckpt.items():
        if isinstance(value, dict) and "xyz" in value:
            if include_objects or key == "background":
                gaussian_keys.append(key)

    print(f"Found {len(gaussian_keys)} Gaussian groups to load "
          f"(include_objects={include_objects})")

    # First pass: determine max SH channels across all groups
    max_sh_channels = 0
    for key in sorted(gaussian_keys):
        value = ckpt[key]
        dc_dim = value["feature_dc"].shape[1]
        rest_dim = value.get("feature_rest", torch.zeros(1, 0, 3)).shape[1]
        total_dim = dc_dim + rest_dim
        max_sh_channels = max(max_sh_channels, total_dim)

    # For standard sh_degree=1 rendering, we only need 4 SH coefficients.
    # The CUDA rasterizer is configured with sh_degree=1, so extra channels
    # are ignored during forward pass and get zero gradient during backward.
    # We pad all groups to the same channel count so torch.cat works.
    sh_channels = max_sh_channels
    print(f"Max SH channels across groups: {sh_channels} "
          f"(sh_degree=1 uses first 4 only)")

    for key in sorted(gaussian_keys):
        value = ckpt[key]
        n_gauss = value["xyz"].shape[0]

        all_xyz.append(value["xyz"])
        all_scales.append(value["scaling"])
        all_rotations.append(value["rotation"])
        all_opacities.append(value["opacity"])

        # Combine feature_dc + feature_rest and pad to uniform channel count
        feature_dc = value["feature_dc"]    # [N, dc_dim, 3]
        feature_rest = value.get("feature_rest")
        if feature_rest is not None and feature_rest.shape[1] > 0:
            shs = torch.cat([feature_dc, feature_rest], dim=1)  # [N, dc+rest, 3]
        else:
            shs = feature_dc

        # Pad to max_sh_channels if needed
        curr_channels = shs.shape[1]
        if curr_channels < sh_channels:
            padding = torch.zeros(n_gauss, sh_channels - curr_channels, 3)
            shs = torch.cat([shs, padding], dim=1)

        all_shs.append(shs)

        if key == "background":
            print(f"  {key}: {n_gauss:,} gaussians, SH={curr_channels} coeffs "
                  f"(standard sh_degree=1)")
        else:
            print(f"  {key}: {n_gauss:,} gaussians, features={curr_channels} "
                  f"channels (Fourier-encoded, padded to {sh_channels})")

    # Concatenate all Gaussians
    xyz = torch.cat(all_xyz, dim=0)
    scales = torch.cat(all_scales, dim=0)
    rotations = torch.cat(all_rotations, dim=0)
    opacities = torch.cat(all_opacities, dim=0)
    shs = torch.cat(all_shs, dim=0)

    total_n = xyz.shape[0]
    print(f"Total: {total_n:,} Gaussians, SH shape: {shs.shape}")

    # Optional: subsample Gaussians to reduce GPU memory
    if gaussian_subsample > 1:
        print(f"Subsampling: keeping 1/{gaussian_subsample} of Gaussians")
        keep_indices = torch.randperm(total_n, device="cpu")[:total_n // gaussian_subsample]
        keep_indices = keep_indices.sort()[0]
        xyz = xyz[keep_indices]
        scales = scales[keep_indices]
        rotations = rotations[keep_indices]
        opacities = opacities[keep_indices]
        shs = shs[keep_indices]
        total_n = xyz.shape[0]
        print(f"After subsampling: {total_n:,} Gaussians")

    # Move to GPU and enable gradients
    xyz = xyz.to(device).requires_grad_(True)
    scales = scales.to(device).requires_grad_(True)
    rotations = rotations.to(device).requires_grad_(True)
    opacities = opacities.to(device).requires_grad_(True)
    shs = shs.to(device).requires_grad_(True)

    # Create dataclass_gs (FaithFusion format)
    gs = dataclass_gs(
        _opacities=opacities,
        _means=xyz,
        _rgbs=None,       # Not used; we always render via SH
        _scales=scales,
        _quats=rotations,  # street_gaussians stores quaternions as 'rotation'
        _shs=shs,
        detach_keys=[],
    )
    return gs


def load_cameras(cameras_json_path: str, device: str = "cuda"):
    """
    Load camera parameters from cameras.json.

    Each entry: {id, img_name, width, height, position, rotation, fx, fy}

    Returns a list of dicts with camtoworlds (4x4), Ks (3x3), H, W, img_name, id.
    """
    with open(cameras_json_path, "r") as f:
        cam_data = json.load(f)

    cameras = []
    for cam in cam_data:
        # cameras.json stores:
        #   "rotation": 3x3 camera-to-world rotation matrix (R_c2w)
        #   "position": 3-vector w2c translation (T_w2c, NOT camera center!)
        #
        # The street_gaussians code reconstructs the world-to-camera matrix as:
        #   w2c = getWorld2View2(R=rotation, T=position)
        # where:
        #   w2c[:3,:3] = rotation.T
        #   w2c[:3,3] = position
        # The camera center in world coords is: cam_center = -rotation @ position
        rotation = np.array(cam["rotation"])   # 3x3 R_c2w
        t_w2c = np.array(cam["position"])      # w2c translation vector

        # Build correct camera-to-world matrix
        # c2w = inverse of [[R.T, t], [0,0,0,1]]
        #     = [[R, -R@t], [0,0,0,1]]
        cam_center = -rotation @ t_w2c  # = -R_c2w @ T_w2c

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = cam_center

        # Build intrinsic matrix K (3x3)
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = cam["fx"]
        K[1, 1] = cam["fy"]
        K[0, 2] = cam["width"] / 2.0
        K[1, 2] = cam["height"] / 2.0

        cameras.append({
            "id": cam["id"],
            "img_name": cam["img_name"],
            "camtoworlds": torch.from_numpy(c2w).float().to(device),
            "Ks": torch.from_numpy(K).float().to(device),
            "H": cam["height"],
            "W": cam["width"],
        })

    print(f"Loaded {len(cameras)} cameras")
    unique_cams = sorted(set(c["img_name"].split("_")[1] for c in cameras))
    frames = sorted(set(int(c["img_name"].split("_")[0]) for c in cameras))
    print(f"  Camera IDs: {unique_cams}")
    print(f"  Frame range: {min(frames)}-{max(frames)} ({len(frames)} frames)")
    return cameras


def zero_grads(gs: dataclass_gs):
    """Manually zero out gradients of all Gaussian parameters."""
    for param in [gs._means, gs._scales, gs._quats, gs._opacities, gs._shs]:
        if param is not None:
            try:
                param.grad = None
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Sky cubemap and color correction (from street_gaussians rendering pipeline)
# ---------------------------------------------------------------------------

def load_postprocessing_params(ckpt_path: str, device: str = "cuda"):
    """Load sky cubemap and color correction params from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    sky_cube_map = ckpt["sky_cubemap"]["params"]["sky_cube_map"]  # [6, 1024, 1024, 3]
    affine_trans = ckpt["color_correction"]["params"]["affine_trans"]  # [N, 3, 4]
    print(f"Sky cubemap: {list(sky_cube_map.shape)}")
    print(f"Color correction: {list(affine_trans.shape)}")
    return sky_cube_map, affine_trans


def compute_ray_directions(c2w: torch.Tensor, K: torch.Tensor, H: int, W: int):
    """
    Compute normalized world-space ray directions for each pixel.

    This follows street_gaussians' get_rays_torch() using the camera-to-world
    matrix directly.

    Args:
        c2w: [4, 4] camera-to-world matrix
        K:   [3, 3] intrinsic matrix
        H, W: image dimensions

    Returns:
        rays_d: [H, W, 3] normalized ray directions in world space
    """
    # Camera center in world space = c2w[:3, 3]
    rays_o = c2w[:3, 3]  # [3]

    # Pixel coordinates in camera space
    i, j = torch.meshgrid(
        torch.arange(W, device=c2w.device, dtype=torch.float32),
        torch.arange(H, device=c2w.device, dtype=torch.float32),
        indexing="xy",
    )
    xy1 = torch.stack([i + 0.5, j + 0.5, torch.ones_like(i)], dim=2)  # [H, W, 3]

    # Back-project to camera space: pixel_cam = K^{-1} @ pixel_homo
    K_inv = torch.inverse(K)
    pixel_camera = torch.matmul(xy1, K_inv.T)  # [H, W, 3]

    # Rotate from camera to world: pixel_world = R_c2w @ pixel_cam
    R_c2w = c2w[:3, :3]  # [3, 3]
    pixel_world = torch.matmul(pixel_camera, R_c2w.T)  # [H, W, 3]

    # Direction from camera center to pixel in world space
    rays_d = pixel_world  # rays originate from camera center at origin in cam space
    rays_d = rays_d / torch.norm(rays_d, dim=2, keepdim=True).clamp(min=1e-8)

    return rays_d


def render_beauty(
    gs: dataclass_gs,
    c2w: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    sky_cube_map: torch.Tensor,
    affine_matrix: torch.Tensor,
):
    """
    Render a photorealistic image with the full street_gaussians pipeline:
    Gaussian rasterization → sky blending → color correction.

    Uses the STANDARD rasterizer (which outputs accumulated alpha for blending).

    Args:
        gs:            Gaussian parameters
        c2w:           [4, 4] camera-to-world matrix
        K:             [3, 3] intrinsic matrix (scaled to render resolution)
        H, W:          render resolution
        sky_cube_map:  [6, 1024, 1024, 3] sky cubemap
        affine_matrix: [3, 4] color correction affine transform

    Returns:
        rgb_final: [H, W, 3] final rendered RGB
    """
    device = c2w.device

    # Build transforms
    wvt = c2w.inverse().T  # w2c.T = world_view_transform
    tfx = math.tan(focal2fov(K[0, 0], W) * 0.5)
    tfy = math.tan(focal2fov(K[1, 1], H) * 0.5)

    proj = _getProjectionMatrix(znear=0.01, zfar=1000.0, K=K, H=H, W=W)
    fpt = (wvt.unsqueeze(0) @ proj.T.cuda().unsqueeze(0)).squeeze(0)
    cc = c2w[:3, 3]

    # Standard rasterizer settings
    rset = StandardGaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=tfx, tanfovy=tfy,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=wvt, projmatrix=fpt,
        sh_degree=1, campos=cc,
        prefiltered=False, debug=False,
    )
    rasterizer = StandardGaussianRasterizer(rset)

    # Screen-space points
    ssp = torch.zeros_like(gs._means, requires_grad=True, device=device) + 0

    # Rasterize with standard rasterizer (outputs alpha)
    color, radii, depth, alpha, semantic = rasterizer(
        means3D=gs._means,
        means2D=ssp,
        shs=gs._shs,
        colors_precomp=None,
        opacities=gs._opacities,
        scales=gs._scales,
        rotations=gs._quats,
        cov3D_precomp=None,
    )
    # color: [3, H, W], alpha: [1, H, W] (accumulated opacity)

    # ---- Sky blending ----
    rays_d = compute_ray_directions(c2w, K, H, W)  # [H, W, 3]

    # nvdiffrast cubemap sampling
    sky_color = dr.texture(
        sky_cube_map[None, ...],          # [1, 6, 1024, 1024, 3]
        rays_d[None, ...],                # [1, H, W, 3]
        filter_mode="linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]
    sky_color = sky_color[0].permute(2, 0, 1).clamp(0.0, 1.0)  # [3, H, W]

    # Blend: rgb = foreground + sky * (1 - alpha)
    rgb_blended = color + sky_color * (1.0 - alpha)

    # ---- Color correction ----
    A = affine_matrix[:3, :3]  # [3, 3]
    b = affine_matrix[:3, 3]   # [3]
    rgb_corrected = torch.einsum("ij,jhw->ihw", A, rgb_blended) + b.unsqueeze(-1).unsqueeze(-1)
    rgb_corrected = rgb_corrected.clamp(0.0, 1.0)

    # [3, H, W] → [H, W, 3]
    return rgb_corrected.permute(1, 2, 0).detach()


def compute_eig(
    gs: dataclass_gs,
    cameras: list,
    output_dir: str,
    max_train_views: int = -1,
    max_novel_views: int = -1,
    render_scale: float = 0.5,
    device: str = "cuda",
    sky_cube_map: torch.Tensor = None,
    affine_trans: torch.Tensor = None,
):
    """
    Compute EIG scores following the FaithFusion approach.

    Phase 1 (Training views):
      - Render each training view with is_train_set=True
      - Accumulate per-Gaussian Hessian diagonals (via atomicAddSquare in CUDA)
      - Compute I_train = 1 / (H_full + reg_lambda)

    Phase 2 (Novel views):
      - For each view, compute per-view Hessian: H_view
      - Compute acquisition: I_acq = H_view * I_train
      - Render gain_map via override_color
      - Apply log transform and save
    """
    os.makedirs(output_dir, exist_ok=True)

    # Limit number of views for Hessian accumulation
    train_cameras = cameras
    if max_train_views > 0:
        train_cameras = cameras[:max_train_views]

    novel_cameras = cameras
    if max_novel_views > 0:
        novel_cameras = cameras[:max_novel_views]

    H_per_gaussian = {}

    # =========================================================================
    # Phase 1: Accumulate Hessian over training views
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Phase 1: Accumulating Hessian over {len(train_cameras)} training views")
    print(f"{'='*60}")

    for i, cam_data in enumerate(tqdm(train_cameras, desc="Hessian accumulation")):
        H, W = cam_data["H"], cam_data["W"]
        render_H = int(H * render_scale)
        render_W = int(W * render_scale)

        # Build dataclass_camera at the target rendering resolution
        K_scaled = cam_data["Ks"].clone()
        K_scaled[0, 0] *= render_scale  # fx
        K_scaled[1, 1] *= render_scale  # fy
        K_scaled[0, 2] *= render_scale  # cx
        K_scaled[1, 2] *= render_scale  # cy

        cam = dataclass_camera(
            camtoworlds=cam_data["camtoworlds"],
            camtoworlds_gt=cam_data["camtoworlds"],
            Ks=K_scaled,
            H=render_H,
            W=render_W,
        )

        # Render with gradient tracking (is_train_set=True)
        # NOTE: modified_render has swapped parameter names:
        #   zfar (param) <- near plane value, znear (param) <- far plane value
        # We pass near=0.01 as znear, far=1000.0 as zfar
        render_pkg = modified_render(
            gs, cam,
            opaticy_mask=None,
            is_train_set=True,
            override_color=None,
            zfar=1000.0,   # actually far plane (parameter name is swapped)
            znear=0.01,    # actually near plane (parameter name is swapped)
        )

        rendered_rgb = render_pkg["render"].permute(1, 2, 0)  # [H, W, 3]
        params_output = render_pkg["params_output"]
        hit_cnt = render_pkg["pixel_gaussian_counter"]

        # Compute gradient of rendered RGB w.r.t Gaussian parameters.
        # We use torch.autograd.grad() (instead of .backward()) to avoid
        # graph retention issues when the same leaf tensors are reused
        # across multiple forward/backward passes.
        used_params_list = ["means", "rotations", "scales", "opacities", "shs"]
        grad_inputs = []
        for params_name in used_params_list:
            if params_output[params_name] is not None:
                grad_inputs.append(params_output[params_name])

        # grad_tensor = ones (uniform gradient), hit-count-normalized
        grad_tensor = torch.ones_like(rendered_rgb)
        min_hit = 1e-6
        hit_cnt_safe = torch.max(
            hit_cnt, torch.tensor(min_hit, device=device)
        ).unsqueeze(-1).expand(grad_tensor.shape)
        grad_tensor = grad_tensor / hit_cnt_safe

        grads = torch.autograd.grad(
            outputs=rendered_rgb,
            inputs=grad_inputs,
            grad_outputs=grad_tensor,
            retain_graph=False,
            allow_unused=True,
        )

        # Initialize Hessian storage on first iteration
        if len(H_per_gaussian) == 0:
            for params_name in used_params_list:
                if params_output[params_name] is not None:
                    H_num = params_output[params_name].shape[0]
                    H_per_gaussian[params_name] = torch.zeros(
                        H_num, device=device
                    )

        # Accumulate per-Gaussian Hessian diagonals
        for idx, params_name in enumerate(used_params_list):
            if params_name in H_per_gaussian and grads[idx] is not None:
                grad = grads[idx].detach()
                # Sum over all dimensions except the first (Gaussian dimension)
                tmp_gs_H = grad.reshape(grad.shape[0], -1).sum(dim=1)
                H_per_gaussian[params_name] += tmp_gs_H

        # Free memory
        del grads, grad_tensor, rendered_rgb, render_pkg, params_output
        torch.cuda.empty_cache()

        # Log memory usage periodically
        if (i + 1) % 10 == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  [{i+1}/{len(train_cameras)}] GPU memory: {mem_mb:.0f} MB")

    # =========================================================================
    # Compute I_train (inverse Fisher information)
    # =========================================================================
    print("\nComputing I_train (inverse Fisher information)...")
    reg_lambda = 1e-6

    H_per_gaussian_full = next(iter(H_per_gaussian.values()))
    for key, tensor in H_per_gaussian.items():
        if tensor is not H_per_gaussian_full:
            H_per_gaussian_full = H_per_gaussian_full + tensor

    I_train = torch.reciprocal(H_per_gaussian_full + reg_lambda)
    I_train_sqrt = torch.sqrt(I_train)

    print(f"  H_full range: [{H_per_gaussian_full.min().item():.6f}, "
          f"{H_per_gaussian_full.max().item():.6f}]")
    print(f"  I_train range: [{I_train.min().item():.6f}, "
          f"{I_train.max().item():.6f}]")

    # =========================================================================
    # Phase 2: Compute gain maps for novel views
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: Computing gain maps for {len(novel_cameras)} views")
    print(f"{'='*60}")

    for i, cam_data in enumerate(tqdm(novel_cameras, desc="Gain map rendering")):
        H, W = cam_data["H"], cam_data["W"]
        render_H = int(H * render_scale)
        render_W = int(W * render_scale)

        K_scaled = cam_data["Ks"].clone()
        K_scaled[0, 0] *= render_scale
        K_scaled[1, 1] *= render_scale
        K_scaled[0, 2] *= render_scale
        K_scaled[1, 2] *= render_scale

        cam = dataclass_camera(
            camtoworlds=cam_data["camtoworlds"],
            camtoworlds_gt=cam_data["camtoworlds"],
            Ks=K_scaled,
            H=render_H,
            W=render_W,
        )

        # ---- Pass 1: Render with SH to get params_output and gradients ----
        render_pkg = modified_render(
            gs, cam,
            opaticy_mask=None,
            is_train_set=False,
            override_color=None,
            zfar=1000.0,
            znear=0.01,
        )

        rendered_rgb = render_pkg["render"].permute(1, 2, 0)
        params_output = render_pkg["params_output"]
        rendered_opacity = render_pkg["opacity"]  # [H, W] (named "opacity", actually depth)

        # Compute per-view Hessian using torch.autograd.grad()
        grad_inputs_p2 = []
        for params_name in H_per_gaussian.keys():
            if params_output[params_name] is not None:
                grad_inputs_p2.append(params_output[params_name])

        grad_tensor_p2 = torch.ones_like(rendered_rgb)
        grads_p2 = torch.autograd.grad(
            outputs=rendered_rgb,
            inputs=grad_inputs_p2,
            grad_outputs=grad_tensor_p2,
            retain_graph=False,
            allow_unused=True,
        )

        # Accumulate view-specific Hessian
        H_view_gaussian = None
        for idx, params_name in enumerate(H_per_gaussian.keys()):
            if grads_p2[idx] is not None:
                grad = grads_p2[idx].detach()
                tmp_gs_H = grad.reshape(grad.shape[0], -1).sum(dim=1)
                if H_view_gaussian is None:
                    H_view_gaussian = tmp_gs_H
                else:
                    H_view_gaussian = H_view_gaussian + tmp_gs_H

        # Compute acquisition function: I_acq = H_view * I_train
        I_acq = H_view_gaussian * I_train

        # Build override_color tensor for uncertainty rendering
        hessian_color = torch.cat([
            H_per_gaussian_full.unsqueeze(1),   # channel 0: total Hessian
            I_train_sqrt.unsqueeze(1),          # channel 1: sqrt covariance
            I_acq.unsqueeze(1),                 # channel 2: information gain
        ], dim=1)

        zero_grads(gs)

        # ---- Pass 2: Render uncertainty maps via override_color ----
        render_pkg2 = modified_render(
            gs, cam,
            opaticy_mask=None,
            is_train_set=False,
            override_color=hessian_color,
            zfar=1000.0,
            znear=0.01,
        )

        uncertainty_map_full = render_pkg2["render"].permute(1, 2, 0)  # [H, W, 3]
        hit_cnt2 = render_pkg2["pixel_gaussian_counter"]

        # Extract gain map (channel 2) and mask by opacity
        opacity_mask = rendered_opacity > 0.1
        gain_map = uncertainty_map_full[:, :, 2] * opacity_mask.float()

        min_hit = 1e-6
        hit_cnt_safe = torch.max(
            hit_cnt2, torch.tensor(min_hit, device=device)
        )
        gain_map = gain_map / hit_cnt_safe / 1000.0

        # Post-process: log transform and sky sentinel
        gain_map = torch.log(gain_map + 1.0) + 1e-9
        gain_map[~opacity_mask] = 100.0  # Sentinel for sky / no-Gaussian pixels

        # Save gain map
        torch.save(gain_map.cpu(), os.path.join(output_dir, f"gain_{i:04d}.pt"))

        # ---- Beauty render (sky + color correction) for visualization ----
        if sky_cube_map is not None and affine_trans is not None:
            # Get color correction matrix for this camera
            cam_id = cam_data["id"]
            aff = affine_trans[cam_id]  # [3, 4]

            with torch.no_grad():
                beauty_rgb = render_beauty(
                    gs, cam_data["camtoworlds"], K_scaled,
                    render_H, render_W, sky_cube_map, aff,
                )
        else:
            beauty_rgb = rendered_rgb.detach()

        # Save rendered RGB
        rgb_np = beauty_rgb.cpu().numpy().clip(0, 1)
        rgb_img = Image.fromarray((rgb_np * 255).astype(np.uint8))
        rgb_img.save(os.path.join(output_dir, f"rgb_{i:04d}.png"))

    # Save metadata
    metadata = {
        "num_gaussians": gs._means.shape[0],
        "num_train_views": len(train_cameras),
        "num_novel_views": len(novel_cameras),
        "render_scale": render_scale,
        "reg_lambda": reg_lambda,
        "H_full_min": H_per_gaussian_full.min().item(),
        "H_full_max": H_per_gaussian_full.max().item(),
        "I_train_min": I_train.min().item(),
        "I_train_max": I_train.max().item(),
        "camera_info": [
            {"id": c["id"], "img_name": c["img_name"]}
            for c in novel_cameras
        ],
    }
    with open(os.path.join(output_dir, "eig_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {output_dir}")
    print(f"  - gain_XXXX.pt : EIG gain maps")
    print(f"  - rgb_XXXX.png : Rendered RGB images")
    print(f"  - eig_metadata.json : Run metadata")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute EIG (Expected Information Gain) scores "
                    "from a trained street_gaussians model"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True,
        help="Path to the trained checkpoint (e.g., iteration_10000.pth)"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None,
        help="Path to the dataset directory (not required, but useful for logging)"
    )
    parser.add_argument(
        "--cameras_json", type=str, required=True,
        help="Path to cameras.json (in the experiment output directory)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save EIG results"
    )
    parser.add_argument(
        "--max_train_views", type=int, default=50,
        help="Max number of training views for Hessian accumulation (-1 = all)"
    )
    parser.add_argument(
        "--max_novel_views", type=int, default=20,
        help="Max number of views to compute gain maps for (-1 = all)"
    )
    parser.add_argument(
        "--render_scale", type=float, default=0.5,
        help="Render resolution scale factor (e.g., 0.5 = half resolution)"
    )
    parser.add_argument(
        "--include_objects", action="store_true",
        help="Also load object (tracked) Gaussians in addition to background. "
             "NOTE: objects use Fourier-encoded features, NOT standard SH. "
             "Their rendered colors may be incorrect without per-frame decoding."
    )
    parser.add_argument(
        "--gaussian_subsample", type=int, default=1,
        help="Subsample Gaussians by factor N to reduce GPU memory "
             "(1=all, 2=half, 4=quarter, etc.). WARNING: subsampling may "
             "cause all-black renders for narrow-FOV cameras."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)"
    )
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    if not os.path.exists(args.cameras_json):
        raise FileNotFoundError(f"cameras.json not found: {args.cameras_json}")

    # Load Gaussian parameters
    gs = load_gaussian_params(args.ckpt_path, args.device,
                              include_objects=args.include_objects,
                              gaussian_subsample=args.gaussian_subsample)

    # Load sky cubemap and color correction
    sky_cube_map, affine_trans = load_postprocessing_params(
        args.ckpt_path, args.device
    )

    # Load cameras
    cameras = load_cameras(args.cameras_json, args.device)

    # Compute EIG
    compute_eig(
        gs=gs,
        cameras=cameras,
        output_dir=args.output_dir,
        max_train_views=args.max_train_views,
        max_novel_views=args.max_novel_views,
        render_scale=args.render_scale,
        device=args.device,
        sky_cube_map=sky_cube_map,
        affine_trans=affine_trans,
    )


if __name__ == "__main__":
    main()
