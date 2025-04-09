# 非对齐PP和VPP切分

## 背景与挑战

类Megatron-LM框架已成为大模型训练的主流方案之一，PP(流水并行 Pipeline Parallelism) 和VPP（虚拟流水并行 Virtual Pipeline Parallelism）是大模型训练的基本并行范式，该范式在部分场景存在计算不均衡问题。具体而言：

- **嵌入层（Embedding Layer）**：在处理文本或分类数据时，嵌入层将高维稀疏特征转换为低维稠密向量表示。此过程涉及索引查找和潜在的大规模矩阵乘法，特别是在自然语言处理应用中，词汇表可能包含数十万乃至数百万词条。高维度的查找与转换操作会消耗大量计算资源。

- **对数几率层（Logits Layer）**：位于网络末端的对数几率层通常是一个全连接层，其功能是将最后一层的隐藏状态映射到输出空间，为后续损失函数计算提供未归一化的预测值。如果分类任务具有大量类别，那么该层的权重矩阵将非常庞大，导致矩阵乘法运算成为性能瓶颈。

- **多token预测（Multi-Token Prediction）**：位于网络末端的一个模块，将预测范围扩展到每个位置的多个未来token。扩展了对数几率层的预估能力，同时对计算资源提出了更高的要求。

上述操作的计算复杂度随着输入特征数量和类别数量的增加而上升，可能导致训练速度降低，并且在计算资源有限的环境中形成性能瓶颈。

## 解决方案

为应对上述挑战，我们引入了“非对齐PP和VPP切分”功能，允许用户通过指定每个PP层和VPP层分布的transformer layer的层数，来动态调整模型在训练流水线中的计算负载。此机制有助于在多个计算节点间更均匀地分配工作负载，从而优化整体计算资源的利用。


## 使用场景

- 当用户遇到由于计算资源分配不均导致的性能瓶颈时，此功能尤为适用。通过对计算密集型任务进行重新分配，可以有效减少流水线中的空闲时间（即空泡），从而提高系统的吞吐量和效率。

## 使用方法

在模型参数中添加 --pipeline-num-transformer-layers 参数，使用二维矩阵来表示PP层和VPP层中transformer layer的层数，横轴坐标表示pp rank，纵轴坐标表示vpp rank。
假设：pipeline_num_transformer_layers = [[0,1],[1,1]*4,[1,0]]，pp_rank = 0， vpp_rank = 1，则pipeline_num_transformer_layers[pp_rank][vpp_rank]中只有1层layer。
假设没有设置VPP，则可以设置为pipeline_num_transformer_layers = [[1],[2]*4,[1]]，如果pp_rank = 1，则pipeline_num_transformer_layers[pp_rank]有2层layer。

**注意事项：**
- 需要保证数组中的层数之和与配置项--num-layers中一致；
- 由于框架中默认VPP是等切的，需要满足单个PP层的内容能够被配置项--num-layers-per-virtual-pipeline-stage整除，实际生效以数组中层数控制为准；

**设置训练脚本参数**
```shell
# 开启非对齐PP/VPP
--pipeline-num-transformer-layers [[0,1],[1,1]*4,[1,0]] \
```

## 使用效果
通过实施自定义PP和VPP层中transformer-layers层数控制策略，预期能够显著减少流水线中的空泡现象，从而优化计算流程并提升系统性能。这不仅有助于加速模型训练过程，还能最大化硬件资源的利用率。

