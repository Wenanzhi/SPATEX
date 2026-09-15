# PACT 开源候选版

本目录是供作者审阅的独立代码包。完整安装和运行命令见 [README](README.md)。

## 先核对这一处差异

当前论文写 `PCM + 0.15 IPD + 0.03 Coh`；与表格结果对应的归档配置仍启用
`0.1 Act`。`doa_only` 关闭的是分离路径中的活动门控，并不自动关闭活动辅助损失。

- `configs/archived/` 保留归档损失设置，仅替换本机数据路径。
- `configs/manuscript/` 将 Act 权重设为零，尚未据此重训，不能宣称能复现论文数值。

论文表 1 的 PCM 行来自 `abl_m2_coatt_logit_mean_pcm/99.pt`，不是另一个
`pure_pcm` 实验。主模型为 `abl_m2_coatt_logit_mean_full/85.pt`，聚合方式为
`mean`，不是旧 `sqrt_count`。

## 已整理内容

PACT 与独立注意力对照、损失消融、固定四麦克风配置，训练/数据生成/评测入口，
无需 enrollment 的推理接口，以及回归测试。保留原网络参数名以兼容 checkpoint。
语音数据、权重、论文 PDF、内部会话、集群脚本和训练日志不在本候选仓库中。

外部基线的覆盖范围见 [BASELINES.md](docs/BASELINES.md)；复现限制见
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。作者审阅说明位于本目录上一级。
