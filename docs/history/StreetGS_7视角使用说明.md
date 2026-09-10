# 修复后的 L3 数据 → StreetGS 7 视角

服务器输出根目录：`/data/l3data-reconstruction-bingxing/tem-test/streetGS`

训练代码：`/data/l3_data_test/street_gaussians-main-local-v2/train.py`

输入：`/data/l3data-reconstruction-bingxing/samples-restored/l3-data`

本次三个 clip 分别有 299、297、297 帧；按用户选择，各取 PKL 索引 **0–100（含首尾）**，每段 101 帧、707 张图，共 2121 张图。

| 相机 ID | 输入名称 |
|---|---|
| 0 | left_front_camera |
| 1 | right_front_camera |
| 2 | rear_camera |
| 3 | left_rear_camera |
| 4 | right_rear_camera |
| 9 | front_camera_fov30 |
| 10 | front_camera_fov120 |

## 已实现的适配

1. 使用 `ego2global_transformation_matrix_camera_reference_offset_optimized`。相机到世界位姿为 `优化后的 ego2world @ inverse(camera.extrinsic)`；`camera.extrinsic` 在此 PKL 中是 ego→camera。
2. 点云使用交付目录下的 `compensated_perception/*.pcd`，不回退到旧点云。保持点云和目标框的 ego 坐标，由训练读取器统一变换到世界坐标。
3. 从 `/data` 下解析原图，使用 PKL 的 `cam_intrinsic`、`d` 和 `cam_intrinsic_resize` 去畸变到 **1600×900**。训练内参采用同一个 `cam_intrinsic_resize`，LiDAR 深度按该内参重新投影。
4. 动态掩码先在原始图像坐标上绘制，再使用同一去畸变映射变换；目标可见性和轨迹转换复用旧 M18 脚本。轨迹时间采用交付的 `dynamic_compensation_reference_time_s`。
5. 两个前视名称映射到旧读取器的 `center_camera_fov30/120`，对应 ID 9/10。
6. `lib/utils/m18_utils.py` 改为根据标定文件名查找相机参数，移除 `frame * 11 + cam` 和图像数除以 11 的假设，并校验内参、外参、相机位姿文件名一致。`train.py` 无需修改。

适配器复用 `/data/mnt/yswang-wan22/code/data_proc_local/M18proc/M18_converter_xirang_pkl_parallel_v2.py` 的点云投影、深度、轨迹和 NPZ 合并逻辑；没有改动原转换器或原始数据。

## 输出结构

```text
streetGS/
  convert_restored_l3_streetgs.py
  train_restored_l3_7view.sh
  m18_7view_reader.patch
  m18_utils.before.py               # 本次修改前的读取器备份
  reader_validation.json
  smoke_f0_2/                      # 3 帧 × 7 路的链路验证
  clip_M18-2_07_..._DF/
    converted/
      images/                      # 000000_00.jpg … 000100_10.jpg
      intrinsics/                  # 3×3 pinhole 内参
      extrinsics/                  # ego→camera 的 4×4 矩阵
      ego_pose/                    # 每帧 ego2world 和逐图 camera2world
      lidar_depth/                 # NPY：mask + value
      dynamic_mask/
      sky_mask/
      track/
      pointcloud.npz               # ego 点云 + 相机投影
      timestamps.json
      timestamps_specific.json
      conversion.complete.json    # 原始帧映射、来源、转换参数
    train.yaml
    sky.log
    sky.complete
    train.log
    model/
      input_ply/
      point_cloud/
      trained_model/
    train.complete                 # 25000 次迭代成功结束后生成
```

`conversion.complete.json` 只在所有转换检查通过后写入；转换器发现输出目录已存在会拒绝覆盖。失败的部分输出需要先移走，再重新执行。

## 重新转换或训练

以下命令在 Pod 中运行。转换输出需使用尚不存在的目录：

```bash
BASE=/data/l3data-reconstruction-bingxing/tem-test/streetGS
CLIP=clip_M18-2_07_20251128074133_DF
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/python \
  "$BASE/convert_restored_l3_streetgs.py" \
  --restored-root "/data/l3data-reconstruction-bingxing/samples-restored/l3-data/$CLIP" \
  --output "$BASE/$CLIP/converted" --frame-start 0 --frame-count 101 --num-workers 4

# 转换完成后，生成/复用天空掩码并开始训练；第二个参数是物理 GPU ID。
bash "$BASE/train_restored_l3_7view.sh" "$CLIP" 4
```

正式配置沿用 `L3_front.yaml` 的优化参数和 25000 次迭代，覆盖 `source_path`、`model_path`、`record_dir`、GPU、`selected_frames: [0,100]`、`cameras: [0,1,2,3,4,9,10]`。图像先存 CPU，训练时传入 GPU，控制 707 张图带来的常驻显存占用。脚本的 GPU 配置写入 YAML，避免原配置中的 `gpus: [5]` 覆盖外部 GPU 选择。

天空掩码使用现有 DA3 环境 `/data/mnt/yswang-wan22/code/Depth-Anything-3/env/bin/python`，配合 `PYTHONPATH=.../Depth-Anything-3/src`；旧实验的 `/data/envs/reconstruction_env/bin/python` 在当前 Pod 上指向缺失的 Python 3.9，不能直接照搬。

## 验证范围

已用 3 帧 × 7 路完整跑过转换、天空掩码、初始化和 30 次训练迭代，保存了 `smoke_f0_2/model/trained_model/iteration_30.pth`。转换时逐图检查 camera2world 与优化 PKL 公式一致、内参与 K_resize 一致、深度尺寸/有效值正确，并检查 NPZ 帧索引。

短训练仅用于验证程序链路；重建质量需要查看正式训练结果。原 `L3_front.yaml` 没有独立验证帧，本次沿用该设置，训练 PSNR 不能当作独立测试集指标。
