# 混合长序列并行

## 背景与挑战

从生成性AI到科研模型，长序列训练正在变得非常重要。 在生成性AI领域，会话式AI、长文档摘要和视频生成等任务都需要在空间和时间层面对长上下文进行推理。 同样，章节和书籍级别的摘要（数万甚至数十万字）在会话式AI和摘要任务中也受到重视。现有的数据、张量和流水线等并行方法无法在序列维度进行切分。当序列维度(S)增长时，训练内存开销会以 $O$($S^2$) 的速度增长。因此需要针对长序列场景进行特定的优化解决长训练场景的训练需求。

目前流行的序列并行方案，Ulysses和Ring Attention存在各自的局限性。

Ulysses需要确保attention head数可以被序列并行维度整除，在GQA、MQA场景下序列并行的大小有限制，导致序列长度的扩展有限。

Ring Attention的并行维度不受attention head数限制，因此理论上序列长度可以无限拓展。但相比于Ulysses，Ring Attention不能充分利用通信和计算带宽，在序列块大小较低时性能劣于Ulysses。

## 解决方案

对Ulysses和Ring Attention做融合，实现混合序列并行，以此解决两个方案各自缺陷。
具体细节可参见文献[USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719)。
 
## 使用场景

**后端限制：** CP>1 必须设置 `--transformer-impl transformer_engine`。

可兼容FlashAttention，目前已默认开启FlashAttention。

序列并行维度被分为Ulysses维度和Ring Attention维度，Ulysses维度和Ring Attention维度乘积即为序列并行维度。

## 使用方法

| 训练类型 | 支持的 mask 类型 |
| --- | --- |
| 普通训练 | causal、general |
| EOD Reset | 不支持 |

使用 `--cp-comm-type` 选择通信方式。

| 重要参数 | 参数说明 |
| --- | --- |
| `--transformer-impl transformer_engine` | CP>1 的必需后端。 |
| `--context-parallel-size C` | 长序列并行大小，默认 1。 |
| `--cp-comm-type a2a+p2p` | 开启混合长序列并行。 |
| `--hierarchical-context-parallel-sizes A R` | A 为 Ulysses 度数，R 为 Ring 度数；A×R=C。注意力头数应能被 TP×A 整除。 |

`--attention-mask-type` 可选择 `causal`（默认）或 `general`。支持 `--use-cp-send-recv-overlap`、`--use-fused-ring-attention-update` 和 `--cp-window-size`；窗口大小须小于 Ring 度数 R 且能整除 R。不支持 EOD Reset。

## 使用效果

利用多个计算设备对输入序列进行并行切分，降低单设备的内存消耗，相比不开启序列并行单步耗时增加，相比重计算计算效率提升。

## 鸣谢

1. GitHub项目地址：
<https://github.com/feifeibear/long-context-attention>

2. 论文预印本地址：
USP: A Unified Sequence Parallelism Approach for Long Context Generative AI
<https://arxiv.org/abs/2405.07719>
