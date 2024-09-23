# MindSpeed Mask归一实现阐述

## 问题分析
### 1. Megatron源码阐述
[1] 各device通过 `pretrain_gpt.py#L93`-`def get_batch` 去获取各项数据，包括AttnMask。

[2] PP的首尾节点通过 `megatron/training/utils.py#L276`-`def get_batch_on_this_tp_rank` 去获取各项数据，包括AttnMask。其他节点直接返回None。

[3] TP的首节点通过 `megatron/core/datasets/gpt_dataset.py#L675`-`def _get_ltor_masks_and_position_ids` 生成AttnMask。

[4] TP其他节点，直接生成与首节点相同shape的empty矩阵，通过broadcast获取首节点生成的AttnMask

Tips: 以上操作默认开启，生成的AttnMask全部为下三角形状，可以通过 `--no-create-attention-mask-in-dataloader` 关闭

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
目前支持FA和非FA的Mask生成，传入AttnMask可以为None和下三角模式。

FA场景，当序列长度大于2048或使用ring_cp_algo时，默认走压缩模式。

其他场景使用完整Mask。


## 使用方法
[1] 针对以上问题和思路，在MindSpeed中，直接默认使能AttnMask，不再使用原生mask生成方式。

[2] 提供全局变量 `mindspeed/model/transformer.py`-`global _GLOBAL_ATTN_MASK`。

[3] 提供 `--sparse_mode` 传参，配合FA多种模式调用。

[4] 提供 `mindspeed/model/transformer.py`-`def set_attention_mask`、`def get_attention_mask`、`def generate_attention_mask` 三个额外接口，实现正常流程外的设置、获取和生成功能。

[5] 提供 `mindspeed/model/transformer.py`-`def parallel_transformer_forward_wrapper` 接口，避开get_batch的缺陷，在第一次正向的时候使能generate_attention_mask来生成mask。


## 使用效果
例如下三角模式，压缩模式下设sparse_mode=2，mask.shape固定为[2048,2048]，大幅提升性能并降低显存。


## 注意事项
当前FA场景仅支持下三角及Band模式，其他自定义AttnMask模式需要手动set_attention_mask，或修改get_attention_mask逻辑。
