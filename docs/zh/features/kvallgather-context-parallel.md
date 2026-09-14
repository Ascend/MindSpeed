# KVAllGather长序列并行

## 背景与挑战

从生成性AI到科研模型，长序列训练正在变得非常重要。 在生成性AI领域，会话式AI、长文档摘要和视频生成等任务都需要在空间和时间层面对长上下文进行推理。 同样，章节和书籍级别的摘要（数万甚至数十万字）在会话式AI和摘要任务中也非常重要。现有的数据、张量和流水线等并行方法无法在序列维度进行切分。当序列维度(S)增长时，训练内存开销会以 $O$($S^2$) 的速度增长。因此需要针对长序列场景进行特定的优化解决长训练场景的训练需求。

## 解决方案

在KVAllGather长序列并行方案中，各计算设备首先在序列维度上分别持有不同片段的数据样本。随后，在注意力计算开始前，通过对已分片的key和value执行all-gather通信操作，使每个设备都能获取完整的key和value序列。最后，每个设备利用本地的query与完整的key、value进行注意力计算，得到对应的输出结果。

更多关于KVAllGather长序列并行方案的细节请参见原文：

> Section 3.3.2 of "The Llama 3 Herd of Models" <https://arxiv.org/abs/2407.21783>

## 使用场景

**后端限制：** CP>1 必须设置 `--transformer-impl transformer_engine`。

KVAllGather通过收集完整的key、value计算本地query对应的attention。支持的 mask 范围见下文注意事项。

在GQA（Grouped-Query Attention）、MQA（Multi-Query Attention）等场景下，由于对key和value进行all-gather通信所需的时间远少于整体计算时间，因此该方案带来的收益更为显著。

该方案可与FlashAttention完全兼容，目前已默认启用FlashAttention进行加速计算。

## 使用方法

| 训练类型 | 支持的 mask 类型 |
| --- | --- |
| 普通训练 | causal |
| EOD Reset | causal，需满足 [EOD Reset 配置约束](eod-reset.md) |

`--cp-comm-type allgather` 与 `all_gather` 等价。

使用 `--cp-comm-type` 选择通信方式。

| 重要参数                                               | 参数说明                                                    |
|----------------------------------------------------|---------------------------------------------------------|
| --context-parallel-size [int]                      | 开启CP对应的数量，默认为1，根据用户需求配置。                                |
| --cp-comm-type <b>all_gather</b> | 长序列并行算法选项，设置为`all_gather`, 开启KVAllGather长序列并行。 |
| --seq-length [int]                                 | 输入序列的长度。                                                |

## 使用效果

利用多个计算设备对输入序列进行并行切分，降低单设备的内存消耗，相比不开启序列并行单步耗时增加，相比重计算计算效率提升。

## 注意事项

1. 开启KVAllGather长序列并行时需要同时设置`--transformer-impl transformer_engine`，否则特性不支持。
2. 普通训练和 EOD Reset 均仅支持 `--attention-mask-type causal`。
3. 对于定长padding训练场景，采用负载均衡的序列切分方式，`--seq-length`要求能被 2 * context-parallel-size整除。 
4. EOD Reset 需设置 `--variable-seq-lengths --micro-batch-size 1`，各层统一使用 `all_gather`；不使用 Ring 的逐文档在线 pad。序列长度仍需满足原生 CP 切分要求。
