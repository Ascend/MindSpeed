# Ampipe流水通信隐藏

## 问题分析

MoE模型中引入了alltoall通信算子，用于在ep组中不同rank间交换token。在MoE层前向过程中，专家mlp部分前后各有一个alltoall通信算子，且计算与通信为串行执行，需要减少这部分通信的时间，提升训练性能。


## 解决方案

ampipe将transformer模型中从attention到mlp部分的通信和计算的输入切分为多份，每一份数据之间互相独立不存在依赖，使得各个部分的计算和通信可以循环流水并行，同时调整计算和通信的算子执行顺序，实现计算和通信并行达到掩盖通信的目的。

![原理图](../../sources/images/ampipe.png)

论文参考：
https://openreview.net/pdf?id=yLgr02IsXY

## 解决思路
1. 从attention的输入开始切分，q和attention_mask在seq序列维度进行切分, k, v保持完整输入，可以使得切分attention后再拼接结果等价。 
2. attention之后的dropout、残差、norm归一化以及mlp等计算在seq序列维度上均独立，切分后再拼接结果同样可以等价，所以在中间各个部分不需要拼接，直到所有计算完成后再拼接结果即可。
3. 切分后重新编排各个切分副本循环流水的顺序，使得计算和通信并行。
4. 针对主流的Megatron的序列并行sequence-parallel以及长序列并行的context-parallel进行适配，可以实现sp开启时mlp部分的all-gather和reduce-scatter通信隐藏。

## 使用场景

在训练MoE模型时，可以开启ampipe特性。
推荐在`--seq-length`序列长度较长时开启特性，可以获得更好的性能提升。

## 使用方法

1. 在训练脚本中添加`--ampipe-degree N`即可使能ampipe特性，N为切分数。
2. 推荐开启`--ampipe-tp-sp-comm-overlap`，额外掩盖mlp中tp域内通信以达到最佳性能提升。
3. 支持同时开启ampipe特性（包含1,2中两个特性开关）以及mlp通信隐藏特性`--use-pipe-experts`，单独或同时设置`--pipe-experts-multi-stream`和`--pipe-experts-multi-data N`来叠加使用“多流水线”和“多副本”的特性。

限制条件：
1. 需要开启`--moe-model-type deepspeed_moe`以及`--use-flash-attn`的前提下使用特性
2. 暂不支持`--use-ascend-mc2`、`--overlap-grad-reduce`、`--overlap-param-gather`以及nanopipe `--use-nanopipe`、ripipe `--recompute-in-bubble` `--recompute-in-advance`和自适应选择重计算。
3. 需要保证设置的`--seq-length`即序列长度可以被`--ampipe-degree`整除，如果需要设置`--sequence-parallel`以及`--context-parallel-size > 1`，需要额外保证设置的`--seq-length`可以被tp和cp整除
4. 同时开启ampipe特性以及mlp通信隐藏特性时，`--pipe-experts-multi-data N`多副本数量N必须被`--ampipe-degree M`ampipe切分数M整除且N>M，否则`--use-pipe-experts`不生效；同时额外设置`--pipe-experts-multi-stream`时，此限制可以放宽至N>=M

## 使用效果

使用该特性可以提升性能。

场景：双机16P, sequence_len = 128k, num_layers = 2, num_experts = 4, recompute_method = block, recompute_granularity = full, recompute_num_layers = 2, hidden_size = 12288, moe_router_topk = 2, ep = 2, tp = 8, dp = 1, cp = 2, pp = 1, sp = True


|          对比场景           | ampipe-degree | ampipe-tp-sp-comm-overlap | multi-stream | multi-data | 平均TFLOPs | 提升幅度  |
|:-----------------------:|:-------------:|:-------------------------:|:------------:|:----------:|:--------:|:-----:|
|        baseline         |       1       |             关             |      关       |     1      |  120.56  |   /   |
| pipe-experts（baseline2） |       1       |             关             |      开       |     2      |  124.85  | 3.56% |
|         ampipe          |       2       |             开             |      关       |     1      |  127.29  | 5.58% |
|   ampipe&pipe-experts   |       2       |             开             |      开       |     4      |  126.87  | 5.23% |


## 注意事项

- 在开启`--ampipe-degree N`时，若`N`过大，导致输入数据切分过细，会引入多余的 cast 和 add 算子，导致额外的开销，引起性能劣化。 目前仅推荐开启`--ampipe-degree 2`，在开启`--context-parallel-size` > 1的场景下，仅支持设置`--ampipe-degree 2`。
- 推荐开启`--ampipe-tp-sp-comm-overlap`，尤其在开启`--sequence-parallel`时，可额外掩盖mlp中tp域内通信以达到最佳性能提升。
- 与部分通信隐藏特性冲突，暂时不支持，参考使用方法中的限制条件。
