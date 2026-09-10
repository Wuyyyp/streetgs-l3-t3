# 来源与许可证

- Street Gaussians: https://github.com/zju3dv/street_gaussians ，保留根目录 LICENSE 和 README.upstream.md。
- 原始与贡献度裁切光栅器：submodules/diff-gaussian-rasterization、diff-gaussian-rasterization_ms；保留各自 LICENSE.md、版权头。后者来源为服务器 GSAPro 分支，并已接入正常 StreetGS 的可选开关。
- simple-knn、simple-waymo-open-dataset-reader：保留各子目录许可证。
- GLM：随三个 CUDA 光栅器保留所需源码和 copying.txt；移除生成文档、测试及示例资产。
- FaithFusion：仅保留 compute_eig 所需 diff 光栅器以及原 LICENSE / README，未打包其完整独立训练流水线。
- tools/data/M18proc：来自现用 data_proc_local/M18proc 转换器及常量定义，与 L3/T3 适配脚本一同保存。
- Depth Anything 3：外部安装依赖；仅保存现用推理逻辑，移除未使用的服务器私有 minors 导入，权重改为命令行参数。

此整理未改写或替换上游许可证。历史脚本中的绝对路径用于追溯原实验，新的通用入口见根 README。
