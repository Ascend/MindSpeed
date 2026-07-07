# Allgather Dispatcher 分支优化

## 背景与挑战

**gather & scatter 算子替换**

在Megatron MoE中的Allgather分支，存在使用gather/scatter操作。gather/scatter功能为沿dim轴根据索引逐元素进行取值/赋值操作，此操作会有大量的随机地址，对性能造成巨大影响。

在Megatron MoE中对gather/scatter的调用主要是以下调用方式，通过对index做expand操作对维度进行扩展，再通过扩展后index对hidden_states进行逐元素取值/赋值。

```python
self.global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
local_hidden_states = torch.gather(global_hidden_states, 0, self.global_local_map)
```

**异步通信**

在Allgather dispatcher分支中，permute函数开头会分别对hidden_states、max_ind、max_prob三个数据做allgather通信，这些操作为串行操作，但各计算任务之间并非串行依赖关系。

## 解决方案

**gather & scatter 算子替换**

由于index是通过expand进行扩展的，因此它的每一行中的内容都是一致的，所以无需使用gather/scatter进行逐元素的操作，可通过index算子以及indexput算子进行逐行操作，对gather/scatter进行等价替换。

**异步通信**

通过对通信任务进行重新排序，并使用async=True参数进行异步下发，达到计算和通信并行的目的。

## 使用场景

本优化策略适用于部署了Mcore MoE（Mixture of Experts）架构的深度学习模型，并开启`--moe-token-dispatcher-type allgather`。

## 使用方法

开启参数`--moe-permutation-async-comm`。

## 使用效果

根据实际测试数据显示，类DeepSeek-V2十亿参数级别的MoE模型，采用上述优化措施后，端到端训练过程中的性能提升了约10%。这意味着不仅加快了模型收敛速度，同时也降低了达到相同精度水平所需的计算资源消耗。
