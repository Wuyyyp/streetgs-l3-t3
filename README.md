# StreetGS · L3 / T3 重建项目

整理日期：2026-09-10。基于当前服务器 `street_gaussians-main-local-v2` 的代码快照，整合 L3 / T3 数据匹配、格式转换、修复前后对比、7 视角训练、可选背景贡献度裁切、原位渲染和全量 PSNR。

## 代码入口

| 功能 | 入口 |
|---|---|
| 训练 / 恢复训练 | `train.py` |
| L3 图像、标定、点云适配 | `tools/data/convert_restored_l3_streetgs.py` |
| L3 修复前后配对 | `tools/data/prepare_pose_comparison.py` |
| T3 图像时间戳、CSV 位姿匹配 | `tools/data/convert_t3_streetgs.py` |
| 内置 M18 转换依赖 | `tools/data/M18proc/` |
| 生成训练配置 | `tools/make_config.py` |
| 天空掩码 | `tools/infer_sky.py`（独立 DA3 环境） |
| 背景贡献度裁切 | `lib/utils/static_prune.py`、`docs/static_prune.md` |
| 原位 / 偏移渲染 | `render.py`、`render_shifted_v2.py` |
| 全量 PSNR | `tools/eval_streetgs_full_psnr.py` |
| 原位三视角视频 | `tools/render_original_triptych.py`（旧 25k / 4 路模型流程） |
| 历史批处理脚本 | `tools/legacy_runs/`，保留原实验路径与参数，迁移时需修改 |
| 可选 EIG 工具 | `compute_eig.py`、`render_eig_shift.py`、`FaithFusion-main/diff/` |

仅打包源码、配置、必要 CUDA / GLM 源码及许可证。不包含训练数据、点云、权重、checkpoint、视频、编译产物或凭据。原实验目录和运行中的训练不受影响。

## 环境

当前训练服务器：Linux、Python 3.8.10、PyTorch 1.13.1+cu116、CUDA 11.6。需要 NVIDIA GPU 和与 PyTorch 匹配的 CUDA 编译工具链，macOS 不用于训练。

```bash
conda create -n streetgs python=3.8
conda activate streetgs
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization ./submodules/simple-knn ./submodules/simple-waymo-open-dataset-reader
# 启用贡献度裁切时安装：
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization_ms
```

DA3 天空掩码使用单独安装的 Depth Anything 3 环境及 DA3MONO-LARGE 权重；本项目提供推理脚本，不包含 DA3 模型源码/权重。EIG 可选工具还需安装 `FaithFusion-main/diff` 与 nvdiffrast。上游说明见 [README.upstream.md](README.upstream.md)。

## 数据匹配规则

默认每个 clip 取 101 帧，转换后编号为 0–100，共 707 张图，1600×900。

| ID | 相机 |
|---|---|
| 0 / 1 | 左前 / 右前 |
| 2 / 3 / 4 | 后视 / 左后 / 右后 |
| 9 / 10 | 前视 FOV30 / FOV120 |

- **L3**：按 PKL 的图像路径查原图，使用 `cam_intrinsic`、`d`、`cam_intrinsic_resize` 去畸变；修复后 camera2world = 优化 ego2world × inverse(ego2camera)。前视名称映射为 M18 的 `center_camera_fov30/120`。`prepare_pose_comparison.py` 生成修复前组，共享非位姿输入，仅切换 ego 与 camera 世界位姿。
- **T3**：PKL 微秒时间戳 ×1000 精确匹配原图纳秒文件名，校验 CSV 的帧号、相机、文件名和时间戳。相机世界位姿分别来自 `image_camera_to_world_before.csv` / `after.csv`；外参由各图像时刻的 camera2world 与帧级 ego2world 反推。原图已去畸变，只缩放图像和内参。两组分别重算点云投影、深度和动态可见性。
- 两类输入均使用交付的 `compensated_perception` ego 点云。T3 无 LiDAR 投影的相机写空深度掩码，训练跳过不足 2 个有效点的深度损失。
- 仅有 **before / after** 两种 pose。匹配失败会报错，不用最近邻偷偷补位姿。已知 T3 第四个历史 clip 的第 0 帧缺 CSV 对应位姿，需显式设置 `--frame-start 1`。

输入目录需包含 `result_interpolation_optimized.pkl` 和 PKL 引用的补偿点云；T3 还需 `image_extrinsics/image_camera_to_world_{before,after}.csv`，原图根目录下为 `camera/<camera_name>/<timestamp_ns>.jpg`。仅加载可信来源的 PKL。

## 转换与训练

从仓库根目录执行，路径替换为自己的实际路径，转换输出必须不存在。

```bash
# L3 修复后
python tools/data/convert_restored_l3_streetgs.py \
  --restored-root /path/to/restored-l3/clip \
  --source-prefix /path/to/raw-prefix \
  --output /path/to/run/after/clip/converted --frame-count 101 --num-workers 4

# T3 修复前 / 后，两次使用同一原始帧范围
python tools/data/convert_t3_streetgs.py \
  --restored-root /path/to/restored-t3/clip --raw-root /path/to/raw-t3/clip \
  --output /path/to/run/before/clip/converted --variant before --frame-count 101
python tools/data/convert_t3_streetgs.py \
  --restored-root /path/to/restored-t3/clip --raw-root /path/to/raw-t3/clip \
  --output /path/to/run/after/clip/converted --variant after --frame-count 101 \
  --shared-images /path/to/run/before/clip/converted/images

# 在 DA3 环境中运行，对 before / after 均生成或共享同一图像对应的掩码
python tools/infer_sky.py --root /path/to/run/after/clip/converted \
  --out /path/to/run/after/clip/converted --weights /path/to/DA3MONO-LARGE/model.safetensors

# 100k：10k/20k/30k/40k/45k 裁切，稠密化到 <50k
python tools/make_config.py --source /path/to/run/after/clip/converted \
  --model /path/to/run/after/clip/model --output /path/to/run/after/clip/train.yaml \
  --gpu 0 --iterations 100000 --densify-until 50000 --prune-interval 10000 --prune-until 45000
python train.py --config /path/to/run/after/clip/train.yaml
```

不额外裁切：生成另一个配置，设 `--prune-interval 0`。50k 方案使用 `--iterations 50000 --densify-until 25000 --prune-interval 5000 --prune-until 25000`。贡献度裁切只作用于背景；关闭后仍保留 StreetGS 原有 opacity/size pruning。停止稠密化后仍继续优化参数。

L3 before 配对需先准备 `after/<clip>/converted`、`train.yaml`、天空掩码，并在确认掩码完整后写入 `after/<clip>/sky.complete`，再运行：

```bash
python tools/data/prepare_pose_comparison.py --base /path/to/run --clip clip --gpu 0
python train.py --config /path/to/run/before/clip/train.yaml
```

默认每 5000 步及训练末步保存 checkpoint。恢复时确保旧进程已退出，再用同一配置启动；默认 `resume: true`。核心训练失败时不会静默忽略恢复错误。新实验使用新的 model 目录。

输出包含 `point_cloud/iteration_<N>/point_cloud.ply`、`trained_model/iteration_<N>.pth` 和可选 `static_prune_events.jsonl`。

## 渲染与评估

```bash
# 原始相机直接渲染（推荐新 7 路模型使用，保留 train.yaml 的模型配置）
python render.py --config /path/to/run/after/clip/train.yaml mode evaluate loaded_iter 100000

# 全部 707 张训练视图，逐张 PSNR(dB) 的算术平均
python tools/eval_streetgs_full_psnr.py --sample /path/to/run/after/clip \
  --iteration 100000 --gpu 0 --output /path/to/run/after/clip/psnr_all_100000.json
```

全量 PSNR 工具约定 sample 下有 `train.yaml`、`converted/` 和 `model/`，当前面向 7×101、1600×900；它是训练视图重建指标，不是独立测试集指标。视频用的机盖遮罩不参与此项计算。

旧偏移渲染器原位使用 `--shift_vector 0 0 0`。历史三视角视频按左前(0)｜前视 FOV120(10)｜右前(1) 排列。`tools/legacy_runs/render_comparison.py` 保存了本次上下拼接视频的实际实现与时间戳对齐逻辑，属于实验脚本，需调整其 job 路径。

## 检查与来源

```bash
python tests/test_static_prune_schedule.py
python tests/test_packaging.py
# CUDA 环境下回归裁切与正常渲染反传
CUDA_VISIBLE_DEVICES=0 python tests/test_static_prune_switch.py
```

打包时执行了语法、配置生成、匹配规则和裁切调度检查；没有在新环境重跑完整训练或重编译所有 CUDA 扩展。主训练/渲染代码为当前服务器快照；新增整理内容集中在 `tools/`、README、打包检查与示例配置。原训练器保留现有固定遮挡区域等实验行为。

本项目基于 Street Gaussians，**并非全部代码均为原创或可任意商用**。保留根目录 `LICENSE` 及第三方组件原许可证；上游 StreetGS 对研究/非商业用途及衍生修改有明确限制，详见原文。额外组件来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
