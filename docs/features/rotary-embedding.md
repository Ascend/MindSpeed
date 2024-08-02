# Rotary Postion Embedding 融合优化

## 问题分析

Rotary Position Embedding（RoPE）是一种大模型文本位置信息编码（Position Embedding）的解决方案。RoPE通过绝对位置编码的形式实现了相对位置信息的注入，融合了绝对和相对位置编码的优点，同时具备较好的长度外推性。目前RoPE方案已经被较多的大模型采用，例如LLaMA和GLM。

然而，目前torch并没有针对RoPE做特定的实现和优化，在模型侧通常是通过自定义的方式实现，且Rotary Embedding的计算方式较为复杂，实现方式的计算和内存开销需要优化。

## 解决方案
`torch_npu`侧将Rotary Embedding操作合并成一个算子，减少数据传输和临时储存，优化模型训练性能。MindSpeed调用`torch_npu`侧接口实现算子融合。

## 使用场景

模型侧使用了Rotary Embedding作为Position Embedding解决方案。

## 使用方法

首先确保`--position-embedding-type`选项设置为`rope`。

同时开启`--use-fused-rotary-pos-emb`选项，以启用融合算子。

## 使用效果

使用融合算子可以提升训练性能。
