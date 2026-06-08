# flash-attn-npu Batch Invariant Attention

## 背景

`--use-flash-attn-npu-batch-invariant` 用于在 MindSpeed 中接入 `flash-attn-npu` attention 后端，服务于训练/推理一致性相关的 batch invariant 场景。

该后端来自 `MinghuasLab/flash-attention-npu` 仓库提供的 Python 包 `flash_attn_npu`，使用其中的 v2 接口：

```python
from flash_attn_npu import flash_attn_func, flash_attn_varlen_func
```

该后端作为可选 attention 实现接入，通过独立开关 `--use-flash-attn-npu-batch-invariant` 启用。

## 安装

在训练环境中先安装 `flash-attention-npu`，环境准备和依赖说明请参考 `MinghuasLab/flash-attention-npu` 仓库的 `README.md`。构建 v2 接口时可使用：

```bash
cd /path/to/flash-attention-npu
FLASH_ATTN_BUILD_VERSION=v2 python setup.py install
```

安装后可用以下命令确认 Python 包可导入：

```bash
python - <<'PY'
from flash_attn_npu import flash_attn_func, flash_attn_varlen_func
print("flash_attn_npu import ok")
PY
```

## 使用方式

启动 Megatron-LM / MindSpeed 训练时增加：

```bash
--use-flash-attn-npu-batch-invariant
```

不要与 MindSpeed 现有 flash attention 开关同时使用：

```bash
--use-flash-attn
--use-fusion-attn-v2
```

## 接口差异

### MindSpeed 现有 FlashAttention

MindSpeed 现有路径使用 NPU fusion attention 接口：

```python
torch_npu.npu_fusion_attention(
    query,
    key,
    value,
    head_num,
    input_layout,
    atten_mask=attention_mask,
    scale=scale,
    pre_tockens=pre_tockens,
    next_tockens=next_tockens,
    keep_prob=keep_prob,
    sparse_mode=sparse_mode,
    actual_seq_qlen=actual_seq_qlen,
    actual_seq_kvlen=actual_seq_kvlen,
)
```

该接口使用 Ascend 原生参数语义，支持 `SBH` / `TND` 等 layout，并通过 `actual_seq_qlen` / `actual_seq_kvlen` 描述 reset attention mask 场景中的实际序列结束位置。

### flash-attn-npu

dense attention 使用：

```python
flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=scale,
    causal=causal,
    window_size=(-1, -1),
    alibi_slopes=None,
)
```

varlen / packed attention 使用：

```python
flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=scale,
    causal=causal,
)
```

主要差异如下：

| 维度 | MindSpeed 现有 FlashAttention | flash-attn-npu |
| --- | --- | --- |
| 后端接口 | `torch_npu.npu_fusion_attention` | `flash_attn_npu.flash_attn_func` / `flash_attn_varlen_func` |
| dense 输入 | Megatron `[S, B, H, D]`，调用前转换为 NPU FA 所需布局 | `flash_attn_func` 需要 `[B, S, H, D]` |
| packed 输入 | `TND`，并传 `actual_seq_qlen` / `actual_seq_kvlen` | `[T, H, D]`，并传 `cu_seqlens` / `max_seqlen` |
| 序列边界 | `[s0, s0+s1, ...]`，表示结束位置 | `[0, s0, s0+s1, ...]`，表示累积边界 |
| `cu_seqlens` dtype | 由 NPU FA 接口处理 | 需要 `torch.int32` |
| dropout | 使用 `keep_prob` | 当前接入要求 `attention_dropout == 0` |
| mask | 支持 MindSpeed 现有 `sparse_mode` 等语义 | 当前接入支持 `causal` 和 `no_mask` |

## 当前限制

当前接入优先覆盖训练侧 batch invariant attention 场景，限制如下：

- `attention_dropout == 0`
- 支持 `causal` / `no_mask`
- 支持 fp16 / bf16
- `head_dim <= 256`
- 不支持 `attention_bias`
- 不支持 alibi / pse / softcap
- 不支持 2D TP
- 当前不支持 context parallel，需要 CP attention 路径单独适配
