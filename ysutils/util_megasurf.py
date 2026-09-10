import open3d as o3d
import numpy as np

def get_bbox_from_pcd(pcd, clip=False, min_len=0.001,scale=1.1):
    # obb = pcd.get_oriented_bounding_box()
    # obb.color = (0, 1, 0)
    aabb = pcd.get_axis_aligned_bounding_box()
    # aabb.color = (1, 0, 0)
    # o3d.visualization.draw_geometries([pcd, aabb])
    obb = aabb.get_oriented_bounding_box()
    # obb.color = (0, 1, 0)
    if clip:
        obb.extent = obb.extent * scale
        obb.extent = obb.extent.clip(min=min_len)
    aabb2 = obb.get_axis_aligned_bounding_box()
    # aabb2.color = (1, 0, 0)
    # o3d.visualization.draw_geometries([pcd, aabb2])
    cube = o3d.geometry.TriangleMesh.create_box(width=aabb2.get_extent()[0],height=aabb2.get_extent()[1],depth=aabb2.get_extent()[2])
    cube.translate(
        aabb2.get_center(),
        relative=False,
    )
    # o3d.visualization.draw_geometries([cube, aabb2])
    return cube, aabb2


def get_bbox1():
    p1 = np.asarray([1,1,1])
    p2 = np.asarray([-1,1,1])
    p3 = np.asarray([1,-1,1])
    p4 = np.asarray([1,1,-1])
    p5 = np.asarray([-1,-1,1])
    p6 = np.asarray([1,-1,-1])
    p7 = np.asarray([-1,1,-1])
    p8 = np.asarray([-1,-1,-1])
    bbox_point = np.stack([p1,p2,p3,p4,p5,p6,p7,p8])
    #
    #
    bbox = o3d.geometry.LineSet()
    bbox_line_idx = []
    for i in range(8):
        for j in range(i+1,8):
            bbox_line_idx.append([i,j])
    bbox_line_idx = np.asarray(bbox_line_idx)
    bbox.points = o3d.utility.Vector3dVector(bbox_point)
    bbox.lines = o3d.utility.Vector2iVector(bbox_line_idx)
    bbox.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([[255,0,0]]),(bbox_line_idx.shape[0],1)))
    return bbox