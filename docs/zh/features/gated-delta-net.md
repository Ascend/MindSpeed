# GatedDeltaNet（GDN）

MindSpeed 在 Megatron-Core 的 GatedDeltaNet 基础上提供 `chunk_gated_delta_rule` 和 `l2norm` 的 NPU 实现，因果卷积使用 MA 的基础适配。非确定性模式使用 MindSpeed 的 Triton gated delta rule。

## 使用方法

**后端限制：** GDN/混合模型必须设置 `--transformer-impl transformer_engine`，不支持 `local`。

按[软件安装](../user-guide/install_guide.md)准备配套环境并导入 `megatron_adaptor`。设置 `--transformer-impl transformer_engine`，模型使用 Megatron 原生 `--hybrid-layer-pattern` 描述层结构，其中 `G` 表示 GDN 层；例如 `G*` 表示一个 GDN 层和一个 Attention 层。具体维度和层排列沿用模型配置。

使用混合模型流水线时，通过 `--hybrid-layer-pattern` 中的 `|` 划分 stage，不与 `--pipeline-model-parallel-layout`、`--num-layers-per-virtual-pipeline-stage` 或 `--num-virtual-stages-per-pipeline-rank` 同时配置。

本文描述训练路径；GDN 尚不支持推理路径。
