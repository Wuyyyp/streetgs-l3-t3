print("ysutils ver 20240719")
print("warning: some open3d api err when using (windows, python>=3.9)")

# block change:
# pose rotation -- extrinsic rotation extrin[:3,:3]
# center -- pose[:,3]
#
# colmap
# qvec -- extrin[:3,:3]
# tvec -- extrin[:3,3]