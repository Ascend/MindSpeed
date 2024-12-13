# flash attention 

## 问题分析

由于self-attention的时间计算复杂度和内存复杂度与序列长度成二次方关系，因此transformer在长序列上的处理时间、内存开销较大。近似的注意力方法可以优化这一问题，但会降低模型质量。

## 解决方案

加速注意力的关键在于优化IO访存，即降低片上内存的读/写次数。

### 解决思路:

Flash Attention 是一种优化IO访存开销的精确注意力方法，原理如下图所示[1]，通过Tiling切片、重计算、Kernel Fusion等方式来减少高带宽内存（片上内存）和SRAM之间的内存读/写次数。NPU上提供了相同解决方案，可参考[fusion attention 对外接口](../ops/fusion_attention.md) 。

a. Tiling切片：利用更高速的SRAM代替片上内存，但SRAM的内存容量较少，无法一次性完成所有数据的完整注意力计算，因此需要进行分块计算。

b. 重计算：放弃中间结果写回，需要使用时重新计算，用计算换访存。

c. Kernel Fusion：将多个操作融合为一个操作，基于Tiling利用一个kernel完成整个计算。

<p align="center"> <img src="../../sources/images/flash_attention.png" height="400px" width="880px"></p>

[原文链接](https://arxiv.org/pdf/2205.14135)

## 使用场景

本方法适用于self-attention相关模型，尤其适用于长序列输入场景。开启长序列并行时该特性默认开启。

## 使用方法

设置`--use-flash-attn`即可调用该算法。

## 使用效果

在模型训练时间、模型质量等方面可以提升性能。
