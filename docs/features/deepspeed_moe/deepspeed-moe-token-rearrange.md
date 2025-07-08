# Token 重排性能优化

## 背景与挑战

在DeepSpeed MoE（Mixture of Experts，混合专家）框架中，Token重排操作原通过两个BatchMatmul实现，其时间复杂度为O(s2)。然而，由于矩阵的稀疏特性，原方法在重排计算中产生了不必要的计算负担，存在潜在的优化空间。

## 解决方案
通过采用等价且更加高效的PyTorch API : index_select，重排操作的时间复杂度得以降低至O(s)，从而显著提升了模型训练的整体性能。

* 重排过程：在top1gating或top2gating函数中，计算出每个专家选择的Token索引expert_select_token_idx，其形状为[E*C]。在MoE前向传播过程中，依据此索引，通过index_select API实现Token的重排。
+ 反重排过程：top1gating或top2gating函数还需计算每个Token在各专家输出中的索引位置token_rearrange_ec_idx，形状为[S]。在MoE前向传播过程中，Token经过专家处理后，通过index_select API从[E*C,M]形状的专家输出中恢复出Token的输出[S,M]，最后乘以Token对应专家的选择权重，以得到MoE层的最终输出。

## 使用场景

当进入MoE层的实际序列长度达到或超过8K时，此优化策略将展现出显著的效果。

## 使用方法

设置`--enable-token-rearrange-opt`，即可调用该算法。

## 使用效果

预期性能提升幅度大约在2%至3%之间，具体收益取决于模型的具体配置和数据集特性。

