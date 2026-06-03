# Megatron Gated Delta Net

## 背景与挑战

标准 Transformer 注意力机制在序列维度上的计算和显存复杂度通常随序列长度呈二次增长，长序列训练时容易成为性能和显存瓶颈。Gated Delta Net（GDN）是一类线性注意力结构，通过 Gated Delta Rule 使用递归状态和门控更新替代完整的 Softmax 注意力矩阵，在长序列场景下降低注意力计算与中间激活开销。Qwen 系列模型从 Qwen-Next 起引入 GDN 作为线性注意力变体之一，因此训练这类模型时需要框架侧支持 GDN 层、核心算子、并行切分以及变长序列输入。

## 解决方案

GDN 层将输入 `hidden_states` 投影为 query、key、value、output gate、beta、alpha，经过 QKV 因果卷积、Q/K L2Norm、门控衰减计算和 Gated Delta Rule 核心计算后，再通过 Gated RMSNorm 与输出投影返回到隐藏维度。相比标准注意力，GDN 避免显式构造完整注意力矩阵，更适合长序列线性注意力训练。

GDN 的算法原理和模型设计细节可参考 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)。

### MindSpeed 适配内容

MindSpeed 基于 Megatron Core 的 GatedDeltaNet 模块，面向昇腾 NPU 训练场景补充了以下适配：

- **GDN 特性接入**：支持通过 `--experimental-attention-variant gated_delta_net` 使用 Megatron GDN 线性注意力变体。
- **NPU 核心算子接入**：优先使用 `fla_npu` 提供的 Gated Delta Rule NPU 算子路径；未安装 `fla_npu` 时回退到 Torch-native 路径。
- **Triton/Torch 回退实现**：提供 `chunk_gated_delta_rule` 和 `torch_chunk_gated_delta_rule`，支持确定性训练场景回退。
- **因果卷积回退**：安装 `causal_conv1d` 时使用融合因果卷积路径；未安装或开启确定性路径时使用 PyTorch Conv1d。
- **Context Parallel 支持**：在 GDN 层中通过 All-to-All 完成序列切分与头切分转换，支持 GDN 的 CP 并行训练。
- **Packed Sequence 支持**：支持带 `cu_seqlens` 的 Packed Sequence/THD 变长序列输入。
- **Tensor Parallel 支持**：支持输入投影、输出投影、Conv1d、`A_log` 和 `dt_bias` 参数在 TP 场景下分片训练。
- **分布式 Checkpoint 支持**：在 `sharded_state_dict` 中处理 GDN 投影、Conv1d 与门控参数的分片保存。

## 使用方法

### GDN 最小配置

开启 GDN 时，需要选择 `gated_delta_net` 注意力变体，并配置线性注意力层的 head 维度、head 数量、卷积核大小和插入频率：

```bash
GDN_ARGS="
    --experimental-attention-variant gated_delta_net \
    --linear-attention-freq 4 \
    --linear-conv-kernel-dim 4 \
    --linear-key-head-dim 128 \
    --linear-value-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 32
"
```

### GDN 配置

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--experimental-attention-variant gated_delta_net` | 选择 GDN 注意力变体 | 必须配置 |
| `--linear-attention-freq` | GDN 线性注意力层的插入频率 | 4 |
| `--linear-conv-kernel-dim` | QKV 因果卷积核大小 | 4 |
| `--linear-key-head-dim` | GDN query/key 每个 head 的维度 | 128 |
| `--linear-value-head-dim` | GDN value 每个 head 的维度 | 128 |
| `--linear-num-key-heads` | GDN key head 数量 | 16 |
| `--linear-num-value-heads` | GDN value head 数量 | 32 |
| `--use-naive-l2norm` | 使用朴素 L2Norm 路径替代 FLA L2Norm | 按需开启 |

当 `linear-num-value-heads` 大于 `linear-num-key-heads` 时，当前实现会对 query/key 按 head 维度执行 `repeat_interleave`，以匹配 value head 数量。

### 并行训练配置

GDN 支持 TP、PP 训练。若模型中仍包含普通注意力层，可继续按原有方式配置 FlashAttention 等普通注意力相关参数：

```bash
--tensor-model-parallel-size N
--pipeline-model-parallel-size N
--use-flash-attn
```

### Context Parallel 支持

GDN 的 CP 路径采用 Ulysses 风格的 All-to-All 通信：计算前将本地序列切分转换为完整序列上的部分 head，计算后再转换回序列切分布局。

```bash
--context-parallel-size N
--context-parallel-algo ulysses_cp_algo
```

使用 CP 时，需要满足：

- `seq-length` 能被 `context-parallel-size` 整除。
- 注意力头数能被 `tensor-model-parallel-size * context-parallel-size` 整除。
- GDN 的 key/value head 数量在 TP 和 CP 切分后仍能均匀分配。

### Packed Sequence 变长序列支持

GDN 支持 Packed Sequence/THD 变长序列输入，训练脚本可配合 reset attention mask 相关参数生成 `packed_seq_params`：

```bash
--reset-attention-mask
--reset-position-ids
--variable-seq-lengths
```

Packed Sequence 场景下，当前实现要求：

- batch 维度必须为 1。
- `cu_seqlens_q` 与 `cu_seqlens_kv` 必须一致。
- Packed Sequence 不支持 `deterministic_mode=True`。

### 确定性模式

当 Megatron `TransformerConfig` 中的 `deterministic_mode=True` 时，GDN 会使用 `torch_chunk_gated_delta_rule` 作为 Gated Delta Rule 回退实现：

```bash
--deterministic-mode
```

该路径主要用于确定性复现，性能通常低于 NPU/FLA 融合路径，并且不支持 Packed Sequence。若使用 MindSpeed 的全局 NPU 确定性开关，可参考 `docs/zh/features/npu_deterministic.md` 中的 `--npu-deterministic` 说明。

### 完整示例

完整训练脚本可以参考 `tests_extend/system_tests/feature_tests/qwen_gdn.sh`。

## 注意事项

1. **依赖限制**：GDN 初始化依赖 `flash-linear-attention`（FLA）库；使用 NPU 高性能路径需要安装并正确编译 `fla_npu` 相关算子；`causal_conv1d` 为可选依赖。
2. **精度限制**：Gated Delta Rule 核心算子不支持 float32 输入，推荐使用 bf16 训练。
3. **推理限制**：当前 `GatedDeltaNet.forward` 对推理模式会抛出 `NotImplementedError`，暂不支持 GDN 推理。
4. **Packed Sequence 限制**：Packed Sequence 要求 batch 为 1，且 `cu_seqlens_q` 与 `cu_seqlens_kv` 相同；`deterministic_mode=True` 时不支持 Packed Sequence。
5. **CP 算法限制**：GDN 的 CP 实现是 All-to-All 头序转换路径，建议使用 `ulysses_cp_algo`，暂不支持 Ring CP 对 GDN 递归状态的专门处理。
6. **Head 维度限制**：当前核心融合算子面向常用 head 维度实现，建议 `linear-key-head-dim` 和 `linear-value-head-dim` 不超过 256。
7. **attention_mask**：GDN 层当前不使用 `attention_mask` 参数，模型中普通注意力层的 mask 行为仍按原有注意力实现处理。
