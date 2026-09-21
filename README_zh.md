# SPATEX 开源候选版

**Spatial Preservation for Array-Flexible Target Speech Extraction**

[English](README.md) | [简体中文](README_zh.md)

论文：**SPATEX: Array-Flexible Multichannel-to-Multichannel Target Speech Extraction
with Spatial Cue Preservation**。

SPATEX 根据多通道混合语音、麦克风坐标、有效通道掩码和目标方位角，为每个麦克风重建目标说话人的混响声像。
模型原名为 PACT；仓库名称、Python 包 `pact` 和配置文件名保持不变，以兼容现有使用方式。

本目录是供作者审阅的独立代码包。完整安装和运行命令见 [README](README.md)。

## 模型架构

[![SPATEX 架构：逐麦克风编码、几何感知 TAC、DOA 条件提取、时间 co-attention 和逐麦克风重建。](docs/figures/spatex_architecture.png)](docs/figures/spatex_architecture.pdf)

**图 1. SPATEX 模型架构。** 几何感知 TAC 在麦克风之间交换信息，DOA 条件指定提取目标。
时间 co-attention 共享注意力图，同时保留各麦克风自己的 value 特征流；共享解码器分别重建各麦克风的目标声像。
[矢量 PDF](docs/figures/spatex_architecture.pdf)。

## 先核对这一处差异

当前论文写 `PCM + 0.15 IPD + 0.03 Coh`；与表格结果对应的归档配置仍启用
`0.1 Act`。`doa_only` 关闭的是分离路径中的活动门控，并不自动关闭活动辅助损失。

- `configs/archived/` 保留归档损失设置，仅替换本机数据路径。
- `configs/manuscript/` 将 Act 权重设为零，尚未据此重训，不能宣称能复现论文数值。

论文表 1 的 PCM 行来自 `abl_m2_coatt_logit_mean_pcm/99.pt`，不是另一个
`pure_pcm` 实验。主模型为 `abl_m2_coatt_logit_mean_full/85.pt`，聚合方式为
`mean`，不是旧 `sqrt_count`。

## 已整理内容

SPATEX 与独立注意力对照、损失消融、固定四麦克风配置，训练/数据生成/评测入口，
无需 enrollment 的推理接口，以及回归测试。保留原网络参数名以兼容 checkpoint。
语音数据、权重、论文 PDF、内部会话、集群脚本和训练日志不在本候选仓库中。

外部基线的覆盖范围见 [BASELINES.md](docs/BASELINES.md)；复现限制见
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。作者审阅说明位于本目录上一级。

## 目标 DOA 偏差实验

[![目标 DOA 偏差为 0–15 度时，变量阵列与固定四麦克风 SPATEX 及外部基线的 SI-SNRi 和 DOA drift。](docs/figures/spatex_doa_robustness.png)](docs/figures/spatex_doa_robustness.pdf)

**图 2. 固定四麦克风线性阵列上的目标 DOA 偏差鲁棒性。** 相同偏差幅度的正负方向结果在同一组 500 个场景上取平均。
灰色区域表示四个外部基线的结果范围，灰色虚线表示各偏差幅度下的最佳外部结果。
左图为 SI-SNRi（越高越好），右图为输出与目标之间的 DOA drift（越低越好）。
[矢量 PDF](docs/figures/spatex_doa_robustness.pdf) · [带符号偏差的结果数据](results/figure2_signed.csv)。
