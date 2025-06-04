# 参数副本复用

## 问题分析

当前所有大模型训练客户场景都采用混精训练，导致在训练过程中持续保存了计算权重与状态权重。实际上，两者所使用的生命周期并没有重叠，故可通过数值变换的方式**等价**去除冗余。


## 解决方案

基于大模型混合精度训练中BF16计算参数（负责前后向计算）及FP32参数副本（负责参数更新）不需同时存在的特点，和两者在数值上的对应关系，设计内存共用算法。



### 解决思路

具体算法步骤如下：
1. FP32 = BF16 + Residual；
2. 前向计算开始前将FP32转换为BF16并保存Residual；
3. 优化器更新前基于BF16和Residual恢复FP32参数并进行更新；
4. 使用int32加减法来等价模拟原始逻辑中FP32<->BF16的相互转换（IEEE754向偶数舍入）。



参数副本复用流程如下图所示：
<p align="center"> <img src="../../sources/images/reuse_fp32_param_a.png" height="400px" width="600px"></p>

数值变化的详细逻辑如下图所示：
<p align="center"> <img src="../../sources/images/reuse_fp32_param_b.png" height="400px" width="750px"></p>

## 使用场景

1. 该特性主要用于使用BF16的训练场景。

## 使用方法

设置`--reuse-fp32-param`，即可调用该算法。

## 使用效果

1. 对于Float16OptimizerWithFloat16Params，整体能够节省`sizeof(bfloat16)*模型参数量`的静态内存，性能劣化在多个模型上测试小于1%。
2. 对于开启分布式优化器的训练，整体能够节省`sizeof(bfloat16)*模型参数量 / DP`的静态内存，性能劣化在多个模型上测试小于1%。

## 注意事项

1. 使用legacy model训练时，`reuse_fp32_param`暂不支持和`--overlap-param-gather`一起使用。
2. 使用fused_ema_adamw优化器时，不支持同时开启`reuse_fp32_param`。