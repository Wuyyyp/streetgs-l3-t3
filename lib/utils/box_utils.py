import numpy as np
import cv2

# def get_bound_2d_mask(corners_3d, K, pose, H, W):
#     corners_3d = np.dot(corners_3d, pose[:3, :3].T) + pose[:3, 3:].T
#     corners_3d[..., 2] = np.clip(corners_3d[..., 2], a_min=1e-3, a_max=None)
#     corners_3d = np.dot(corners_3d, K.T)
#     corners_2d = corners_3d[:, :2] / corners_3d[:, 2:]
#     corners_2d = np.round(corners_2d).astype(int)
#     mask = np.zeros((H, W), dtype=np.uint8)
#     cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
#     cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
#     cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
#     cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
#     cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
#     cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
#     return mask

def get_bound_2d_mask(corners_3d, K, pose, H, W):
    corners_cam = np.dot(corners_3d, pose[:3, :3].T) + pose[:3, 3:].T

    znear = 1e-3       # threshold for "behind camera"
    safe_z = 1.0        # minimum depth for stable projection (meters)

    if np.all(corners_cam[..., 2] <= znear):
        return np.zeros((H, W), dtype=np.uint8)

    if np.any(corners_cam[..., 2] <= znear):
        # some corners behind camera → clip the cuboid
        # but first, check if the visible portion is too close for reliable projection
        if np.max(corners_cam[..., 2]) < safe_z:
            return np.zeros((H, W), dtype=np.uint8)
        return _draw_clipped_bbox(corners_cam, K, H, W, znear=znear, safe_z=safe_z)

    if np.any(corners_cam[..., 2] <= safe_z):
        # object fully in front but very close to camera → skip
        return np.zeros((H, W), dtype=np.uint8)

    corners_cam[..., 2] = np.clip(corners_cam[..., 2], a_min=safe_z, a_max=None)
    corners_proj = np.dot(corners_cam, K.T)
    corners_2d = corners_proj[:, :2] / corners_proj[:, 2:]
    corners_2d = np.round(corners_2d).astype(int)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def _draw_clipped_bbox(corners_cam, K, H, W, znear=1e-3, safe_z=0.5):
    """Clip cuboid against the near plane (z=znear), then clamp depth to safe_z
    before projection to avoid numerical explosion."""
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if bin(i ^ j).count('1') == 1]

    visible_pts = []

    for i, j in edges:
        zi, zj = corners_cam[i, 2], corners_cam[j, 2]
        if zi >= znear and zj >= znear:
            continue  # both in front, handled by corner pass below
        if zi < znear and zj < znear:
            continue  # both behind, skip entirely

        # edge crosses the near plane: interpolate to z=znear
        t = (znear - zi) / (zj - zi)
        pt = corners_cam[i] + t * (corners_cam[j] - corners_cam[i])
        pt[2] = znear
        visible_pts.append(pt)

    # add original corners that are in front
    for i in range(8):
        if corners_cam[i, 2] >= znear:
            visible_pts.append(corners_cam[i])

    if len(visible_pts) < 3:
        return np.zeros((H, W), dtype=np.uint8)

    visible_pts = np.array(visible_pts)
    # clamp depth to safe_z for stable 2D projection
    visible_pts[:, 2] = np.clip(visible_pts[:, 2], a_min=safe_z, a_max=None)
    corners_proj = np.dot(visible_pts, K.T)
    corners_2d = corners_proj[:, :2] / corners_proj[:, 2:]
    corners_2d = np.round(corners_2d).astype(np.int32)

    # clamp 2D points to image bounds so the convex hull represents
    # the visible portion that actually intersects the image
    corners_2d[:, 0] = np.clip(corners_2d[:, 0], -1, W + 1)
    corners_2d[:, 1] = np.clip(corners_2d[:, 1], -1, H + 1)

    hull = cv2.convexHull(corners_2d)
    if hull is None or len(hull) < 3:
        return np.zeros((H, W), dtype=np.uint8)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [hull[:, 0, :]], 1)
    return mask

def scale_to_corrner(scale):
    min_x, min_y, min_z = -scale, -scale, -scale
    max_x, max_y, max_z = scale, scale, scale
    corner3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corner3d

def bbox_to_corner3d(bbox):
    min_x, min_y, min_z = bbox[0]
    max_x, max_y, max_z = bbox[1]
    
    corner3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corner3d

def points_to_bbox(points):    
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    bbox = np.array([min_xyz, max_xyz])
    return bbox

def inbbox_points(points, corner3d):
    min_xyz = corner3d[0]
    max_xyz = corner3d[-1]
    return np.logical_and(
        np.all(points >= min_xyz, axis=-1),
        np.all(points <= max_xyz, axis=-1)
    )

