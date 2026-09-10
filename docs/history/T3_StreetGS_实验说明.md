# T3 StreetGS 修复前后对比

服务器结果目录：`/data/l3data-reconstruction-bingxing/tem-test/streetGS/T3-test`

| Clip | 原始帧范围（含首尾） | 每组图像数 |
|---|---|---:|
| 2026_04_25_06_06_15_dlp_pilotGtParser | 0–100 | 707 |
| 2026_04_25_06_11_15_dlp_pilotGtParser | 0–100 | 707 |
| 2026_04_25_06_26_16_dlp_pilotGtParser | 0–100 | 707 |
| 2026_04_25_14_32_57_dlp_pilotGtParser | 1–101（用户确认） | 707 |

第四个 clip 的第 0 帧前视 FOV30 时间早于位姿轨迹起点，before/after CSV 均缺对应记录，因此整体顺延一帧。转换后的帧编号统一为 0–100。

- `before/<clip>/`：修复前；`after/<clip>/`：修复后。只有这两种位姿。
- 每组包含 `converted/`、`train.yaml`、`input_manifest.json`、训练后生成的 `model/`、`train.log`、`state.json`。
- `logs/` 保存每对实验的转换/运行状态；`checks/` 保存验证、原代码备份和补丁；`scripts/` 保存执行脚本。
- 最终 PLY：`model/point_cloud/iteration_25000/point_cloud.ply`；检查点：`model/trained_model/iteration_25000.pth`。

7 个视角为左前、右前、后视、左后、右后、前视 FOV30、前视 FOV120，对应相机编号 0、1、2、3、4、9、10。每组训练 25,000 步；背景贡献度裁切在 2,500、5,000、7,500、10,000、12,500 步执行，CDF 阈值沿用代码默认值 0.99。第 12,500 步裁切后停止增密和透明度重置，继续优化参数。保存第 12,500 和 25,000 步 PLY，运行脚本在结束后逐模型核对点数保持一致，并写入 `pruning.validation.json`。

相机世界位姿直接读取交付包各自的 `image_camera_to_world_before.csv` 和 `image_camera_to_world_after.csv`。帧级 ego 位姿分别使用 `ego2global_transformation_matrix_camera_reference_offset` 与其 `_optimized` 字段；相机相对该帧 ego 的外参由世界位姿反推。两组共用图像与已补偿的 ego 坐标系点云，分别计算投影、深度和物体可见性。

原始图像从 `/root/t3-data/Preproduction_Data` 读取，按 PKL 微秒时间戳乘 1000 精确匹配原图纳秒文件名。遵循 `images_undistorted=True`，仅缩放至 1600×900 并同步缩放内参。天空掩码复用上次实验的 Depth Anything 3 脚本生成，两组共享。物体轨迹来自 `gt_boxes/track_id/vels`，通过三维包围盒投影生成动态区域和视角可见性。

其余训练参数及 M18 训练器的固定图像遮挡区域沿用上次配置。交付点云主要覆盖前向视角；没有有效点云投影的相机仍参与图像训练，深度文件为空掩码。`train.py` 仅额外修复了空深度损失：有效深度点不足 2 个时跳过该项，避免空数组均值导致 NaN。修改前代码和差异保存在 `checks/train_before_empty_depth_fix.py` 与 `checks/empty_depth_fix.patch`。

本文件说明配置与产物位置；实时进度以服务器 `logs/*.state.json` 和各组 `state.json` 为准。

Checkpoint 保存间隔已改为 5,000 步：5,000、10,000、15,000、20,000、25,000。已存在的第 1,000 步检查点保留；第 12,500 步仍保存 PLY 用于核对裁切后点数。中断后可在确认旧进程已退出的情况下执行 `scripts/run_t3_pair.py --clip <clip名> --stage train --resume`；脚本跳过完成的训练，保留旧日志并从最新检查点继续。恢复时会把晚于该检查点的裁切日志归档，防止重复计数。

当前 train.py 直接按每 5,000 步及训练最后一步保存 checkpoint；旧 checkpoint_iterations 列表不再控制频率。恢复检查点加载失败时会明确报错，避免静默跳过恢复。
