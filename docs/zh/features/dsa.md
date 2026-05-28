# Megatron DeepSeek Sparse Attention

## 背景与挑战

标准注意力机制的计算复杂度为 O(S²)，在长序列（如 128K token）场景下，注意力计算成为训练的主要瓶颈。为降低计算量，DeepSeek-V3.2 提出 DeepSeek Sparse Attention（DSA），通过学习一个轻量级 indexer 模型预测 token 之间的重要性分数，仅对 top-k 个最重要的 token pair 计算真实注意力，从而将计算量从 O(S²) 降至 O(S × K)。

## 解决方案

DSA（DeepSeek Sparse Attention）通过引入可学习的 Indexer 网络，为每个 query token 选择重要的 top-k key token，仅在稀疏位置计算注意力，从而降低长序列场景下的注意力计算量。DSA 的算法原理和 DeepSeek-V3.2 模型设计细节可参考 [DeepSeek-V3.2 论文](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)。

### MindSpeed 适配内容

MindSpeed 基于 Megatron 的 DSA 原生实现，面向昇腾 NPU 训练场景补充了以下适配：

- **DSA 特性接入**：支持在昇腾 NPU 上使用 Megatron DSA 注意力变体。
- **NPU 融合算子接入**：支持 `npu_lightning_indexer`、`npu_sparse_flash_attention` 和 `npu_sparse_lightning_indexer_grad_kl_loss` 三个融合算子。
- **TND 变长序列支持**：支持 DSA 开启融合算子场景下使用 TND 数据格式。
- **Tensor Parallel 支持**：支持 DSA 在 TP 并行场景下训练。
- **Context Parallel 支持**：支持 DSA 使用 `kvallgather_cp_algo` 进行 CP 并行训练。

## 使用方法

### DSA 最小配置

开启 DSA 时，需要启用 MLA，并配置 MLA 和 Indexer 的关键参数：

```bash
DSA_ARGS="
    --multi-latent-attention \
    --experimental-attention-variant dsa \
    --qk-layernorm \
    --qk-head-dim 128 \
    --qk-pos-emb-head-dim 64 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --dsa-indexer-n-heads 64 \
    --dsa-indexer-head-dim 128 \
    --dsa-indexer-topk 2048 \
    --dsa-indexer-loss-coeff 1.0 \
    --dsa-indexer-use-sparse-loss \
"
```

### MLA 配置

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--multi-latent-attention` | 启用 MLA（Multi-head Latent Attention） | 必须开启 |
| `--experimental-attention-variant dsa` | 选择 DSA 注意力变体 | 必须配置 |
| `--qk-layernorm` | 启用 Q/K LayerNorm | 必须开启 |
| `--qk-head-dim` | Q/K 非位置编码部分的 head 维度 | 128 |
| `--qk-pos-emb-head-dim` | Q/K 位置编码部分的 head 维度 | 64 |
| `--q-lora-rank` | Query 低秩压缩维度 | 1536 |
| `--kv-lora-rank` | KV 低秩压缩维度 | 512 |
| `--v-head-dim` | Value 每个 head 的维度 | 128 |

### Indexer 配置

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--dsa-indexer-n-heads` | Indexer 注意力头数 | 64 |
| `--dsa-indexer-head-dim` | Indexer 每个头的维度 | 128 |
| `--dsa-indexer-topk` | 稀疏注意力保留的 top-k 数量 | 2048 |
| `--dsa-indexer-loss-coeff` | Indexer KL 辅助 loss 系数 | 1.0 |
| `--dsa-indexer-use-sparse-loss` | 仅在 top-k 位置计算 loss | 建议开启 |

### 融合/吸收路径开关

> [!CAUTION]
> DSA 的四个融合/吸收路径开关必须**同时全部开启**或**同时全部关闭**，不支持部分开启。

```bash
--use-dsa-absorb                       # 启用矩阵吸收
--use-fused-lightning-indexer          # 启用融合 Indexer 算子
--use-fused-sparse-flash-attention     # 启用融合稀疏注意力算子
--use-fused-lightning-indexer-kl-loss  # 启用融合 Indexer KL loss 算子
```

### TND 变长序列支持

DSA 在开启融合算子场景下支持 TND 数据格式：

```bash
--reset-attention-mask                 # 启用打包变长序列
--reset-position-ids                   # 配合使用
--variable-seq-lengths                 # CP + causal reset attention mask 场景必须开启
```

### Tensor Parallel 支持

DSA 支持张量并行训练，可通过以下参数配置 TP 并行度：

```bash
--tensor-model-parallel-size N
```

### Context Parallel 支持

多卡长序列并行时，DSA 支持 `kvallgather_cp_algo` 算法：

```bash
--context-parallel-size N                    # CP 并行度
--context-parallel-algo kvallgather_cp_algo  # CP 算法（仅支持此算法）
```

### 完整示例

完整训练脚本可以参考 `tests_extend/system_tests/feature_tests/deepseek_dsa.sh`。

## 注意事项

1. **版本限制**：DSA 融合算子依赖 `torch_npu` 版本支持，低版本 `torch_npu` 可能不包含相关算子接口。
2. **MLA 依赖**：DSA 必须配合 `--multi-latent-attention` 使用，不支持普通 MHA/GQA 模型。
3. **融合/吸收路径约束**：四个开关（`--use-dsa-absorb`、`--use-fused-lightning-indexer`、`--use-fused-sparse-flash-attention`、`--use-fused-lightning-indexer-kl-loss`）必须同时全部开启或全部关闭。
4. **TND 仅限融合路径**：TND 变长序列格式仅在开启融合开关时支持，unfused 路径不支持 TND。
5. **CP 算法限制**：长序列并行仅支持 `kvallgather_cp_algo`，不支持 `ring_cp_algo` 或 `ulysses_cp_algo`。
6. **qk-layernorm**：当前实现要求 DSA 启用 `--qk-layernorm`，否则参数校验阶段会报错。
