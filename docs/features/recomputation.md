# Megatron 重计算
## 问题分析

大模型训练过程中，通常要求保留前向计算的激活值用于后续的反向梯度计算，并且需要保存结果的数量会随着模型层数的增加线性增加，大大增加芯片的内存压力。

## 解决思路

在前向过程和loss计算时直接删除激活值，反向梯度计算需要用时再重新计算一遍激活值，从而有效缩短激活值的生命周期，缓解内存压力。

## 使用场景
主要用于训练场景，重计算分为：选择性重计算和完全重计算。

选择性重计算（推荐使用）：只重计算transformer中的core_attention部分，将占用较少内存存储空间且重计算开销较高的激活保留在内存中，并将占用较多内存存储空间但重新计算开销相对较低的激活重新计算。

完全重计算：对于内存非常有限场景，仅将输入保存，重新计算所有激活值。

## 使用方法

选择性重计算：脚本中添加`--recompute-activations`开启选择性重计算。

完全重计算：脚本中配置`--recompute-granularity full`开启完全重计算，开启完全重计算时使用`--recompute-method uniform/block` 确认具体重计算方式。

`--recompute-method uniform`：将Transformer层均匀划分组（每组大小`--recompute-num-layers`），按组存储输入和激活值。

`--recompute-method block`：将前`--recompute-num-layers`个transformer层重计算，剩余层不进行重计算。

同时配置`--recompute-activations` 、`--recompute-granularity full`生效选择性重计算。

## 使用影响
显存开销降低、性能降低。
