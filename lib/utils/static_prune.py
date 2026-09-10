"""Optional GSAPro contribution pruning for StreetGS static background."""
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from lib.config import cfg


def scheduled_prune_iterations(optim, total_iterations):
    if not optim.get("static_prune_enabled", False):
        return set()
    interval = optim.get("static_prune_interval", 5000)
    threshold = optim.get("static_prune_cdf_threshold", 0.99)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("static_prune_interval must be a positive integer")
    if not 0 < threshold < 1:
        raise ValueError("static_prune_cdf_threshold must be between 0 and 1")
    end = optim.get("static_prune_until_iter", -1)
    if isinstance(end, bool) or not isinstance(end, int) or end < -1:
        raise ValueError("static_prune_until_iter must be -1 or a nonnegative integer")
    if end == -1:
        # Preserve the original schedule when no independent end is configured.
        return set(range(interval, min(optim.densify_until_iter, total_iterations) + 1, interval))
    if end > optim.densify_until_iter or end > total_iterations:
        raise ValueError("static_prune_until_iter must not exceed densification or training end")
    steps = set(range(interval, end + 1, interval))
    if end > 0:
        steps.add(end)
    return steps


def load_importance_backend():
    module_root = Path(__file__).resolve().parents[2] / "submodules/diff-gaussian-rasterization_ms"
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    return GaussianRasterizationSettings, GaussianRasterizer


def _static_background_importance(viewpoint_camera, background):
    """Return per-background-Gaussian quantities used by GSAPro miniprune."""
    ImportanceRasterizationSettings, ImportanceRasterizer = load_importance_backend()
    raster_settings = ImportanceRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=background.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = ImportanceRasterizer(raster_settings=raster_settings)
    means2d = torch.zeros_like(background.get_xyz)
    _, _, accum_weights, area_proj, area_max = rasterizer(
        means3D=background.get_xyz,
        means2D=means2d,
        shs=background.get_features,
        colors_precomp=None,
        opacities=background.get_opacity,
        scales=background.get_scaling,
        rotations=background.get_rotation,
        cov3D_precomp=None,
    )
    return accum_weights, area_proj, area_max


def _cdf_keep_mask(importance, threshold):
    values, _ = torch.sort(importance.flatten() + 1e-6)
    total = values.sum()
    if not torch.isfinite(total) or total <= 0:
        raise RuntimeError("static prune importance sum is invalid")
    cumulative = torch.cumsum(values, dim=0)
    split_candidates = ((cumulative / total) > (1.0 - threshold)).nonzero()
    if split_candidates.numel() == 0:
        return torch.ones_like(importance, dtype=torch.bool), values[-1]
    split_value = values[split_candidates.min()]
    return importance.flatten() > split_value, split_value


@torch.no_grad()
def prune_static_background(iteration, gaussians, cameras):
    """Run GSAPro importance pruning on the static background only."""
    threshold = float(cfg.optim.get("static_prune_cdf_threshold", 0.99))
    background = gaussians.background
    if background.background_mask is not None:
        raise RuntimeError("static background mask must be None during global pruning")

    before = int(background.get_xyz.shape[0])
    dynamic_before = {
        name: int(getattr(gaussians, name).get_xyz.shape[0])
        for name in gaussians.obj_list
    }
    importance = torch.zeros(before, dtype=torch.float32, device="cuda")
    observed = torch.zeros(before, dtype=torch.bool, device="cuda")
    started = time.time()

    for camera in tqdm(cameras, desc=f"Static prune importance @ {iteration}", leave=False):
        accum_weights, area_proj, area_max = _static_background_importance(camera, background)
        visible = area_max != 0
        observed |= visible
        importance[visible] += (accum_weights / area_proj)[visible]

    importance[~observed] = 0
    if not torch.isfinite(importance).all():
        raise RuntimeError("static prune importance contains NaN or Inf")
    keep_mask, split_value = _cdf_keep_mask(importance, threshold)
    prune_mask = ~keep_mask
    prune_count = int(prune_mask.sum().item())
    if prune_count <= 0 or prune_count >= before:
        raise RuntimeError(f"refusing invalid static prune count {prune_count}/{before}")

    background.prune_points(prune_mask)
    after = int(background.get_xyz.shape[0])
    dynamic_after = {
        name: int(getattr(gaussians, name).get_xyz.shape[0])
        for name in gaussians.obj_list
    }
    if dynamic_after != dynamic_before:
        raise RuntimeError("dynamic Gaussian counts changed during static-only pruning")
    if after != before - prune_count:
        raise RuntimeError("static Gaussian count mismatch after pruning")

    event = {
        "iteration": iteration,
        "threshold": threshold,
        "camera_count": len(cameras),
        "static_before": before,
        "static_after": after,
        "static_pruned": prune_count,
        "static_prune_fraction": prune_count / before,
        "dynamic_total_unchanged": sum(dynamic_after.values()),
        "cdf_split_value_with_epsilon": float(split_value.item()),
        "elapsed_seconds": time.time() - started,
    }
    log_path = os.path.join(cfg.model_path, "static_prune_events.jsonl")
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print("\n[STATIC BACKGROUND PRUNE] " + json.dumps(event, ensure_ascii=False))
    del importance, observed, keep_mask, prune_mask
    torch.cuda.empty_cache()
    return event

