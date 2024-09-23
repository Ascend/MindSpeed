# TFLOPS计算

## 问题分析

当前大模型在计算MFU依靠框架理论打印值TFLOPS/有效算力得到，但是理论值计算适用于一般模型，如果针对模型结构进行变动，将不再适用，同时HFU的计算目前需要再手动计算，不太方便。

## 解决方案

提供接口可以统计所有涉及MatMul计算的算子的浮点计算次数，同时能统计到模型正反向训练以及重计算的总浮点计算次数。

### 解决思路:

目前支持的涉及MatMul的算子有MatMul、BatchMatMul、FlashAttention、MC2相关融合算子、GEMM融合算子。

## 使用方法

（1）对于开启此功能，设置`--op-cal-tflops`即可调用。

（2）由于此功能统计的是各卡的TFOPS信息，在CP/EP/PP场景下，各卡计算量不同，因此在最后需要汇总各卡的信息进行平均，额外增加一个通信，使用gloo通信。

（3）使用此功能由于会增加一个额外通信以及计算各算子的浮点计算次数，可能影响性能。

## 使用效果

通过打印值`actual throughput per NPU (TFLOP/s/NPU)`和`actual throughput per NPU with recompute (TFLOP/s/NPU)`可以方便计算MFU
和HFU。

## 注意事项

由于在Ring Attention长序列并行方案中，在causal场景下，由于算法优化缘故，会有部分计算减少，因此会导致理论值和实际统计值不符合的现象，理论计算上FA
的计算减少值为`(CP-1)/2CP`
