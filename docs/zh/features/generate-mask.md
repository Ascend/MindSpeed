# MindSpeed Mask归一实现阐述

## 背景与挑战

**Megatron源码阐述**

- 各device通过 `pretrain_gpt.py#L93`-`def get_batch`去获取各项数据，包括AttnMask。

- PP的首尾节点通过 `megatron/training/utils.py#L276`-`def get_batch_on_this_tp_rank`去获取各项数据，包括AttnMask。其他节点直接返回None。

- TP的首节点通过 `megatron/core/datasets/gpt_dataset.py#L675`-`def _get_ltor_masks_and_position_ids`生成AttnMask。

- TP其他节点，直接生成与首节点相同shape的empty矩阵，通过broadcast获取首节点生成的AttnMask。

注：以上操作默认开启，生成的AttnMask全部为下三角形状，可以通过`--no-create-attention-mask-in-dataloader`关闭。

**问题发现**

- 昇腾的FA需要外部生成AttnMask，所以除了基础下三角模式，需要额外接口生成自定义AttnMask。

- 非PP首尾节点的AttnMask为None，无法启用FA加速。

- AttnMask生成、拷贝及广播操作，在每个micro_step都会跟随get_batch重复。

- 长序列下，生成的AttnMask占用显存过大。

## 解决方案

- 提供统一AttnMask生成接口，同一进程复用一个全局变量AttnMask，避免重复生成和拷贝。

- 适配AttnMask压缩模式，减少显存占用。

## 使用场景

- 目前支持FA和非FA的Mask生成，传入AttnMask可以为None和下三角模式。

- FA场景，当序列长度大于2048或使用ring_cp_algo时，默认走压缩模式。

- 其他场景使用完整Mask。

## 使用方法

- 针对以上问题和思路，在MindSpeed中，直接默认启用AttnMask，不再使用原生mask生成方式。

- 提供全局变量`mindspeed/model/transformer.py`-`global _GLOBAL_ATTN_MASK`。

- 提供`--sparse-mode` 传参，配合FA多种模式调用。`--sparse-mode`的不同模式信息可以参考[torch_npu.npu_fusion_attention算子文档](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_001095.html)。

- 提供`mindspeed/model/transformer.py`-`def set_attention_mask`、`def get_attention_mask`、`def generate_attention_mask`三个额外接口，实现正常流程外的设置、获取和生成功能。

- 提供`mindspeed/model/transformer.py`-`def parallel_transformer_forward_wrapper` 接口，避开get_batch的缺陷，在第一次正向的时候启用generate_attention_mask来生成mask。

## 使用效果

例如下三角模式，压缩模式下设sparse_mode=2，mask.shape固定为[2048,2048]，将大幅提升性能并降低显存。

> [!NOTE]
>
> 当前FA场景仅支持下三角及Band模式，其他自定义AttnMask模式需要手动set_attention_mask，或修改get_attention_mask逻辑。
