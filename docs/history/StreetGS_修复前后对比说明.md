# StreetGS：修复前后 ego pose 对比

服务器根目录：`/data/l3data-reconstruction-bingxing/tem-test/streetGS`

本次仅做**修复前、修复后两组**，三个 clip 各取前 101 帧，7 路相机 `[0,1,2,3,4,9,10]`，每组每个 clip 训练 25000 次迭代。

## 目录

```text
streetGS/
  before/<clip>/                 # 修复前
    converted/                  # 单独的 ego_pose，其余输入共享 after
    train.yaml
    train.log
    launch.json
    model/                      # 从头训练，不加载 after 的模型
    train.complete              # 成功完成训练后产生
  after/<clip>/                 # 修复后，原先已启动的三组训练
    converted/
    train.yaml
    train.log
    launch.json
    model/
    train.complete
  comparison/<clip>/
    pair.json                   # 字段、参数一致性检查、位姿差值
    pipeline.log
    pipeline_state.json         # 训练/等待/渲染/对比/完成/失败阶段
    metrics.json                # 完成后：总体及分相机 PSNR、SSIM
    per_view.json               # 完成后：每张图的指标
    comparison.html            # 完成后：GT、修复前、修复后并排展示
    comparison.complete        # 对比完成标记
  scripts/                     # 转换、准备对照、训练及对比脚本
  checks/                      # 3 帧验证、读取器回归检查、源代码备份/补丁
  logs/                        # 初始转换和验证日志
  README.md
```

根目录另保留三个旧 clip 路径的符号链接和 `train_restored_l3_7view.sh` 入口链接，用于兼容已经运行的进程及旧配置。它们指向整理后的目录，不是重复数据；不要在这些进程运行时删除。

## 两种位姿的定义

| 组别 | PKL 字段 |
|---|---|
| before | `ego2global_transformation_matrix_camera_reference_offset` |
| after | `ego2global_transformation_matrix_camera_reference_offset_optimized` |

这两个字段分别是相同相机参考时间基准下的优化前后位姿，与交付数据已有 A/B 准备脚本的字段定义一致。更早的原始 `ego2global_transformation_matrix` 不属于此次对照。

每帧写入对应的 ego→world 矩阵；逐图 camera→world 使用 `ego_to_world @ inverse(camera.extrinsic)`。三个 clip 均检查了 101 个 ego 位姿及 707 个相机位姿。

只切换 ego pose。图像、相机内外参、补偿后的 ego 坐标系点云、LiDAR 深度、动态/天空掩码、轨迹、时间戳都共享同一份文件。两组各自在自己的 `model/` 内用相应位姿重新构建世界坐标系初始化点云，不共享训练权重或初始化 PLY。这样可以观察 ego pose 优化本身带来的变化；并非整个感知预处理流程的原始/修复对比。

优化参数、101 帧范围、相机集合一致，沿用代码的随机种子 0；配置仅允许路径、实验名称和 GPU 不同。模型初始化会因位姿变化而改变点云合并和点数，因此不能保证后续随机操作逐步完全一致。

## GPU 分配

| Clip 时间标识 | before | after |
|---|---|---|
| 20251128074133 | GPU 1 | GPU 2 |
| 20251215104909 | GPU 3 | GPU 4 |
| 20251220130904 | GPU 7 | GPU 6 |

`after` 延续已经开始的训练；新加的 `before` 从零开始。每个对比进程在 before 训练结束后等待对应 after 成功结束，再顺序渲染两组 25000 次迭代模型，生成指标和 HTML。失败原因写入 `pipeline_state.json`，详细信息见对应日志。

## 如何查看

```bash
BASE=/data/l3data-reconstruction-bingxing/tem-test/streetGS
CLIP=clip_M18-2_07_20251128074133_DF
tail -f "$BASE/before/$CLIP/train.log"
tail -f "$BASE/after/$CLIP/train.log"
cat "$BASE/comparison/$CLIP/pipeline_state.json"
```

完成后，在浏览器打开 `comparison/<clip>/comparison.html`。图片链接引用对应 `before/after` 模型渲染目录，移动或下载报告时应保持相对目录结构。

## 比较口径

两组均在相同的 707 张训练图像上计算全图 PSNR 和 SSIM，数值越高越好，并输出 after−before 差值。HTML 展示每路相机首帧、中间帧和末帧的 GT/Before/After。指标反映训练视角的重建拟合效果，不是独立测试集或新视角泛化分数。

小样本验证覆盖了 7 路转换、30 次训练迭代，以及同一模型重复渲染的零差值检查。正式两组训练完成之前，不能据此判断修复是否提升了重建质量。
