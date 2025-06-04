# Token 重排性能优化

## 问题分析

DeepSpeed MoE的token重排采用了两个BatchMatmul实现，时间复杂度为O(s^2)，而token重排进行计算时由于矩阵的稀疏性导致一些不必要的计算，存在优化空间。

## 解决方案

重排操作可以通过等价的pytorch API: index_select来实现，降低计算时间复杂度到O(s)，从而提高训练性能。

### 解决思路:

1. 重排过程：top1gating/top2gating 函数计算出每个专家选择的token的索引：expert_select_token_idx，shape为: [E*C]，MoE前向过程中根据此索引通过index_select API实现token的重排；

2. 反重排过程：top1gating/top2gating 函数同时需要计算每个token在各个专家输出的索引位置：token_rearrange_ec_idx，shape为：[S]。在MoE前向过程中，token经过专家输出后通过index_select API 从[E*C, M]的专家输出中恢复token的输出：[S, M]，最后乘以token选择对应专家的权重，得到MoE layer的输出。

## 使用场景

进MoE层时实际序列长度8K以上。

## 使用方法

设置`--enable-token-rearrange-opt`，即可调用该算法。

## 使用效果

预期性能收益在2%~3%左右。

