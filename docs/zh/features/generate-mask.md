# MindSpeed Mask归一实现阐述

## 背景与挑战

### 1. Megatron源码阐述

[1] 各device通过 `pretrain_gpt.py`-`def get_batch` 去获取各项数据，包括AttnMask。

[2] PP的首尾节点通过 `megatron/core/utils.py`-`def get_batch_on_this_tp_rank` 去获取各项数据，包括AttnMask。其他节点直接返回None。

[3] TP的首节点通过 `megatron/core/datasets/gpt_dataset.py`-`def _get_ltor_masks_and_position_ids` 生成AttnMask。

[4] TP其他节点，直接生成与首节点相同shape的empty矩阵，通过broadcast获取首节点生成的AttnMask。

Tips: 以上操作默认开启，生成的AttnMask全部为下三角形状，可以通过 `--no-create-attention-mask-in-dataloader` 关闭。

### 2. 问题发现

[1] 昇腾的FA需要外部生成AttnMask，所以除了基础下三角模式，需要额外接口生成自定义AttnMask。

[2] 非PP首尾节点的AttnMask为None，无法使能FA加速。

[3] AttnMask生成、拷贝及广播操作，在每个micro_step都会跟随get_batch重复。

[4] 长序列下，生成的AttnMask占用显存过大。

## 解决方案

### 解决思路

[1] 提供统一AttnMask生成接口，同一进程复用一个全局变量AttnMask，避免重复生成和拷贝。

[2] 适配AttnMask压缩模式，减少显存占用。

## 使用场景

**后端限制：** 本文 mask 缓存接口仅用于 `--transformer-impl local`。

| 配置 | 支持范围 |
| --- | --- |
| `--attention-mask-type causal` | 默认值，因果注意力；支持 Ring、Ulysses、KVAllGather 和混合 CP。 |
| `--attention-mask-type general` | 普通训练支持 Ring、Ulysses 和混合 CP。EOD Reset 仅 CP=1 可用。 |
| `--use-flash-attn --sparse-mode 0` | 默认模式，根据传入的 mask 计算。 |
| `--use-flash-attn --sparse-mode 2` | 左上对齐的因果压缩 mask。 |

CP>1 需使用 `--transformer-impl transformer_engine`，与 EOD Reset 组合时仅支持 `p2p/all_gather + causal`，具体约束见 [EOD Reset](eod-reset.md)。`attention-mask-type` 与 `sparse-mode` 是不同参数，TENPU 内部使用的算子模式不等同于启动参数的可选范围。

本地 DotProductAttention 路径在 causal 且传入 mask 为 None 时生成并缓存 mask；FA 默认生成 [2048,2048] 的压缩 mask，`multi_head_latent_attention` 配置使用完整序列长度。非 FA 使用完整 mask。

## 使用方法

[1] 针对以上问题和思路，在MindSpeed中，直接默认使能AttnMask，不再使用原生mask生成方式。

[2] 在 `mindspeed/core/transformer/flash_attention/generate_mask/generate_mask.py` 中提供全局变量 `_GLOBAL_ATTN_MASK`。

[3] 提供 `--sparse-mode` 传参，配合FA多种模式调用。`--sparse-mode`的不同模式信息可以参考[torch_npu.npu_fusion_attention算子文档](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_001095.html)。

[4] 提供 `mindspeed/core/transformer/flash_attention/generate_mask/generate_mask.py` 中的 `set_attention_mask`、`get_attention_mask`、`generate_attention_mask` 三个额外接口，实现正常流程外的设置、获取和生成功能。

[5] 本地 DotProductAttention 通过 `mindspeed/core/transformer/flash_attention/generate_mask/adaptor.py` 中的 `dot_product_attention_forward_wrapper` 在首次需要时生成 mask；CP 注意力由 TENPU 处理。

## 使用效果

例如下三角模式，压缩模式下设sparse_mode=2，mask.shape固定为[2048,2048]，将大幅提升性能并降低显存。

## 注意事项

自动生成的 FA mask 为因果 mask；设置 `general` 不会自动生成任意自定义 mask。`set_attention_mask` 仅用于设置本地缓存，不改变 CP/EOD 或 `--sparse-mode` 的支持限制。
