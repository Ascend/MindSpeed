# DeepSeek Sparse Attention（DSA）

MindSpeed 基于 Megatron-Core 的 DSA 实现，提供 NPU 稀疏注意力、Indexer、KL loss 融合算子、矩阵吸收和 KVAllGather 上下文并行适配。

## 使用方法

**后端限制：** 必须设置 `--transformer-impl transformer_engine`，CP=1 时也不支持 `local`。

在模型配置中启用 `--transformer-impl transformer_engine --experimental-attention-variant dsa --multi-latent-attention --qk-layernorm`。MLA 维度配置见 [MLA](multi-head-latent-attention.md)，Indexer 使用 Megatron 原生的 `--dsa-indexer-head-dim`、`--dsa-indexer-n-heads`、`--dsa-indexer-topk`、`--dsa-indexer-loss-coeff` 参数。

以下 MindSpeed 开关默认关闭；全部关闭时使用 Megatron 原生 DSA 路径。

| 参数 | 作用与约束 |
| --- | --- |
| `--use-fused-sparse-flash-attention`（P1） | NPU 稀疏注意力；可单独开启，但此时要求 CP=1。 |
| `--use-fused-lightning-indexer`（P2） | NPU Indexer，必须与 P1、P3 一起开启。 |
| `--use-fused-lightning-indexer-kl-loss`（P3） | NPU Indexer KL loss，必须与 P1、P2 一起开启；`--num-attention-heads` 仅支持 32、64、128。 |
| `--apply-rope-in-complex` | Indexer 使用复数形式 RoPE，可独立于 P1/P2/P3 开启。 |
| `--use-dsa-absorb` | 训练时使用矩阵吸收，可独立于 P1/P2/P3 开启。 |
| `--apply-rope-fusion` | 非 packed Indexer 使用 Ascend fused RoPE；同时开启复数 RoPE 时，Indexer 优先使用复数路径。 |

CP>1 时设置 `--transformer-impl transformer_engine --cp-comm-type all_gather`，见 [KVAllGather](kvallgather-context-parallel.md)。融合路径必须同时开启 P1/P2/P3；不支持 P1 单开与 CP 组合。不支持 `--eod-mask-loss`。

## 检查点与训练

矩阵吸收路径在训练时拆分 KV 权重，在检查点中合并为 Megatron 原生 KV 布局。Indexer loss 按 micro-batch size 归一化。PP/VPP 无需增加 DSA 专用流水线开关。
