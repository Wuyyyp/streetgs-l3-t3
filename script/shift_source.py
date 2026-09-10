#!/usr/bin/env python3
"""
Generate novel trajectory datasets for reconstruction by shifting the original
ego poses. Reads experiment config to locate source data, then produces shifted
variants that can be used directly as --source_path for training / rendering.

Example usage:
    # From experiment directory (auto-discovers source data via cfg_args)
    python script/shift_source.py \
        --exp_dir /mnt/yswang-wan22/exp/test_streetgs_clip_M18-2_07_20251227122737_DF_4v_crt \
        --output_dir /mnt/yswang-wan22/dataset/streetgs_l3/shifted_M18-2_07

    # From source data root directly
    python script/shift_source.py \
        --source_dir /mnt/yswang-wan22/dataset/streetgs_l3/clip_M18-2_07_20251227122737_DF \
        --output_dir /mnt/yswang-wan22/dataset/streetgs_l3/shifted_M18-2_07

    # Generate a single shift variant
    python script/shift_source.py \
        --exp_dir /mnt/yswang-wan22/exp/test_streetgs_clip_M18-2_07_20251227122737_DF_4v_crt \
        --output_dir /mnt/yswang-wan22/dataset/streetgs_l3/shifted_M18-2_07 \
        --shift shift_right

    # Custom shift vector
    python script/shift_source.py \
        --exp_dir /mnt/yswang-wan22/exp/test_streetgs_clip_M18-2_07_20251227122737_DF_4v_crt \
        --output_dir /mnt/yswang-wan22/dataset/streetgs_l3/shifted_M18-2_07 \
        --shift_vector 2.0 0.0 -1.0
"""

import os
import re
import json
import shutil
import argparse
import numpy as np
from glob import glob


# ==============================================================================
# Dictionary: relative shift type → 3-D translation applied to every ego pose.
# Unit: meters.
# Coordinate convention (right-hand, typical autonomous-driving frame):
#   x — forward (longitudinal)
#   y — left   (lateral)
#   z — up     (vertical)
# ==============================================================================
SHIFT_TYPES = {
    'shift_right': [0.0, -1.0, 0.0],   # lateral +1 m to the right (y-negative)
    'shift_left':  [0.0,  1.0, 0.0],   # lateral +1 m to the left  (y-positive)
    'shift_up':    [0.0,  0.0, 1.0],   # vertical +1 m upward       (z-positive)
    'shift_forward': [1.0, 0.0, 0.0],  # longitudinal +1 m forward  (x-positive)
}


# ---------------------------------------------------------------------------
# Helper: parse the cfg_args Namespace dump produced by the training harness
# ---------------------------------------------------------------------------
def parse_cfg_args(exp_dir: str):
    """
    Read  <exp_dir>/cfg_args  and extract  source_path / model_path.

    The file looks like:
        Namespace(model_path='...', sh_degree=1, source_path='...', ...)
    """
    cfg_path = os.path.join(exp_dir, 'cfg_args')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"cfg_args not found in {exp_dir}")

    with open(cfg_path, 'r') as f:
        content = f.read().strip()

    source_path = None
    model_path = None

    # Match quoted values (handles both single and double quotes)
    m = re.search(r"""source_path\s*=\s*['"]([^'"]*)['"]""", content)
    if m:
        source_path = m.group(1)
    m = re.search(r"""model_path\s*=\s*['"]([^'"]*)['"]""", content)
    if m:
        model_path = m.group(1)

    if source_path is None:
        raise ValueError(f"Could not parse source_path from {cfg_path}")

    return source_path, model_path


# ---------------------------------------------------------------------------
# Ego-pose I/O
# ---------------------------------------------------------------------------
def load_ego_poses(source_dir: str):
    """
    Return two dicts keyed by stem (no extension):
        frame_poses : {frame_id: 4×4 ndarray}   — files without underscore
        cam_poses   : {frame_cam: 4×4 ndarray}  — files with underscore
    """
    ego_dir = os.path.join(source_dir, 'ego_pose')
    if not os.path.isdir(ego_dir):
        raise FileNotFoundError(f"ego_pose directory not found: {ego_dir}")

    all_files = sorted(glob(os.path.join(ego_dir, '*.txt')))
    if not all_files:
        raise RuntimeError(f"No .txt files found in {ego_dir}")

    frame_poses = {}
    cam_poses = {}

    for fp in all_files:
        stem = os.path.basename(fp).replace('.txt', '')
        pose = np.loadtxt(fp)
        if '_' in stem:
            cam_poses[stem] = pose
        else:
            frame_poses[stem] = pose

    print(f"[load] {len(frame_poses)} frame poses + {len(cam_poses)} camera poses "
          f"from {ego_dir}")
    return frame_poses, cam_poses


def shift_pose(pose_4x4: np.ndarray, shift_xyz) -> np.ndarray:
    """Add *shift_xyz* to the translation column of a 4×4 pose matrix."""
    out = pose_4x4.copy()
    out[:3, 3] += np.asarray(shift_xyz, dtype=out.dtype)
    return out


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------
# Directories that are symlinked (unchanged by the shift)
DIRS_TO_LINK = [
    'images',
    'extrinsics',
    'intrinsics',
    'track',
    'dynamic_mask',
    'sky_mask',
    'lidar_depth',
    'intensity',
]

# Files that are copied verbatim
FILES_TO_COPY = [
    'timestamps.json',
    'timestamps_specific.json',
    'run.log',
]


def create_shifted_dataset(
    source_dir: str,
    output_root: str,
    shift_name: str,
    shift_vector,
    frame_poses: dict,
    cam_poses: dict,
    dry_run: bool = False,
):
    """
    Build one shifted variant of the source dataset.

    Parameters
    ----------
    source_dir : str
        Original dataset root.
    output_root : str
        Parent folder where the shifted variant subdirectory is created.
    shift_name : str
        Subdirectory name for this variant (e.g. 'shift_right').
    shift_vector : sequence of 3 floats
        [dx, dy, dz] in meters.
    frame_poses, cam_poses : dict
        Pre-loaded original poses.
    dry_run : bool
        If True, only print what would happen.
    """
    shift_output_dir = os.path.join(output_root, shift_name)
    shift_vector = [float(v) for v in shift_vector]

    if dry_run:
        print(f"\n[DRY RUN] Would create: {shift_output_dir}")
        print(f"          Shift vector: {shift_vector} m  (dx, dy, dz)")
        return shift_output_dir

    os.makedirs(shift_output_dir, exist_ok=True)
    print(f"\n[create] {shift_output_dir}")
    print(f"         shift = {shift_vector} m  (dx, dy, dz)")

    # ---- symlink directories that don't change --------------------------------
    for d in DIRS_TO_LINK:
        src = os.path.join(source_dir, d)
        dst = os.path.join(shift_output_dir, d)
        if not os.path.exists(src):
            continue
        if os.path.exists(dst) or os.path.islink(dst):
            continue
        # Use relative symlinks for portability
        rel_src = os.path.relpath(src, os.path.dirname(dst))
        os.symlink(rel_src, dst)
        print(f"  ln -s {rel_src}  →  {d}/")

    # ---- copy small JSON / text files ----------------------------------------
    for f in FILES_TO_COPY:
        src = os.path.join(source_dir, f)
        dst = os.path.join(shift_output_dir, f)
        if not os.path.exists(src):
            continue
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  cp {f}")

    # ---- write shifted ego poses ---------------------------------------------
    ego_out = os.path.join(shift_output_dir, 'ego_pose')
    os.makedirs(ego_out, exist_ok=True)

    for stem, pose in {**frame_poses, **cam_poses}.items():
        shifted = shift_pose(pose, shift_vector)
        np.savetxt(os.path.join(ego_out, f'{stem}.txt'), shifted, fmt='%.16e')

    print(f"  wrote {len(frame_poses) + len(cam_poses)} shifted ego-pose files")

    # ---- metadata ------------------------------------------------------------
    metadata = {
        'shift_name': shift_name,
        'shift_vector_meters': shift_vector,
        'source_path': source_dir,
        'output_dir': shift_output_dir,
        'available_shift_types': {k: v for k, v in SHIFT_TYPES.items()},
    }
    meta_path = os.path.join(shift_output_dir, 'shift_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  wrote metadata → shift_metadata.json")

    return shift_output_dir


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate shifted-trajectory datasets for reconstruction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--exp_dir', type=str, default=None,
        help='Experiment directory that contains cfg_args (used to locate '
             'source data and model). Mutually exclusive with --source_dir.')
    parser.add_argument(
        '--source_dir', type=str, default=None,
        help='Source dataset root directly (must contain ego_pose/, images/, '
             'etc.). Use this when you have the dataset path but no experiment '
             'cfg_args. Mutually exclusive with --exp_dir.')
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output root directory. One subdirectory per shift variant is '
             'created inside it.')
    parser.add_argument(
        '--shift', type=str, default='all',
        choices=['all'] + list(SHIFT_TYPES.keys()),
        help='Which predefined shift to apply (default: all).')
    parser.add_argument(
        '--shift_vector', type=float, nargs=3, default=None, metavar=('DX', 'DY', 'DZ'),
        help='Custom shift vector in meters (overrides --shift).')
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Print plan without writing anything.')

    args = parser.parse_args()

    # ---- 0. Validate input source --------------------------------------------
    if args.source_dir and args.exp_dir:
        parser.error('--exp_dir and --source_dir are mutually exclusive.')
    if not args.source_dir and not args.exp_dir:
        parser.error('Either --exp_dir or --source_dir must be provided.')

    # ---- 1. Resolve source_path ----------------------------------------------
    if args.source_dir:
        # Use the given dataset root directly
        source_path = os.path.abspath(args.source_dir)
        exp_dir = None
        model_path = None
    else:
        # Parse cfg_args from experiment directory
        exp_dir = os.path.abspath(args.exp_dir)
        source_path, model_path = parse_cfg_args(exp_dir)

    print("=" * 64)
    if exp_dir:
        print(f"Experiment dir : {exp_dir}")
    print(f"Source data    : {source_path}")
    if model_path:
        print(f"Model path     : {model_path}")
    print(f"Output root    : {args.output_dir}")
    print("=" * 64)

    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    # ---- 2. Create output folder ---------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 3. Persist experiment info to output --------------------------------
    exp_info = {
        'source_path': source_path,
        'shift_types': SHIFT_TYPES,
    }
    if exp_dir:
        exp_info['exp_dir'] = exp_dir
    if model_path:
        exp_info['model_path'] = model_path
    info_path = os.path.join(args.output_dir, 'experiment_info.json')
    with open(info_path, 'w') as f:
        json.dump(exp_info, f, indent=2)

    # ---- 4. Load original ego poses ------------------------------------------
    frame_poses, cam_poses = load_ego_poses(source_path)

    # ---- 5. Decide which shifts to apply -------------------------------------
    if args.shift_vector is not None:
        shifts_to_apply = {'custom_shift': list(args.shift_vector)}
    elif args.shift == 'all':
        shifts_to_apply = SHIFT_TYPES
    else:
        shifts_to_apply = {args.shift: SHIFT_TYPES[args.shift]}

    print(f"\nShifts to apply: {list(shifts_to_apply.keys())}")
    print(f"Dry run: {args.dry_run}")

    # ---- 6. Generate datasets ------------------------------------------------
    created = []
    for name, vec in shifts_to_apply.items():
        out = create_shifted_dataset(
            source_dir=source_path,
            output_root=args.output_dir,
            shift_name=name,
            shift_vector=vec,
            frame_poses=frame_poses,
            cam_poses=cam_poses,
            dry_run=args.dry_run,
        )
        created.append(out)

    # ---- 7. Summary ----------------------------------------------------------
    print("\n" + "=" * 64)
    if args.dry_run:
        print("[DRY RUN] No files were written.")
    else:
        print("Created datasets:")
        for p in created:
            print(f"  {p}")
    print("Done.")


if __name__ == '__main__':
    main()
