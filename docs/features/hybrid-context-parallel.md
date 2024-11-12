# 混合长序列并行

## 问题分析

从生成性AI到科研模型，长序列训练正在变得非常重要。 在生成性AI领域，会话式AI、长文档摘要和视频生成等任务都需要在空间和时间层面对长上下文进行推理。 同样，章节和书籍级别的摘要（数万甚至数十万字）在会话式AI和摘要任务中也受到重视。现有的数据、张量和流水线等并行方法无法在序列维度进行切分。当序列维度(S)增长时，训练内存开销会以 $O$($S^2$) 的速度增长。因此需要针对长序列场景进行特定的优化解决长训练场景的训练需求。

目前流行的序列并行方案，Ulysses和Ring Attention存在各自的局限性。

Ulysses需要确保attention head数可以被序列并行维度整除，在GQA、MQA场景下序列并行的大小有限制，导致序列长度的扩展有限。

Ring Attention的并行维度不受attention head数限制，因此理论上序列长度可以无限拓展。但相比于Ulysses，Ring Attention不能充分利用通信和计算带宽，在序列块大小较低时性能劣于Ulysses。

## 解决方案
对Ulysses和Ring Attention做融合，实现混合序列并行，以此解决两个方案各自缺陷。

## 使用场景

可兼容FlashAttention，目前已默认开启FlashAttention。

序列并行维度被分为Ulysses维度和ring attention维度，Ulysses维度和ring attention维度乘积即为序列并行维度。

## 使用方法

设置`--context-parallel-size`，默认为1，根据用户需求配置。

设置`--context-parallel-algo hybrid_cp_algo`，以使能混合序列并行。

设置`--ulysses-degree-in-cp`，需要确保`--context-parallel-size`可以被该参数整除且大于1。例如当设置`--context-parallel-size=8`时，可以设置`--ulysses-degree-in-cp=2`或`--ulysses-degree-in-cp=4`。

同时需要确保`--ulysses-degree-in-cp`可以被attention head数整除。

混合长序列并行支持Ring Attention长序列并行相关特性，包括send receive overlap功能、Mask计算类型配置和Double Ring Attention算法。

## 使用效果

利用多个计算设备对输入序列进行并行切分，降低单设备的内存消耗，相比不开启序列并行单步耗时增加，相比重计算计算效率提升。

## 鸣谢

1. GitHub项目地址：
https://github.com/feifeibear/long-context-attention

2. 论文预印本地址：
USP: A Unified Sequence Parallelism Approach for Long Context Generative AI
https://arxiv.org/abs/2405.07719
