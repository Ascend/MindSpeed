# Allgather Dispatcher 分支优化

## 背景与挑战
### 1. gather & scatter 算子替换

在 Megatron MoE 中的 Allgather 分支，存在使用 gather/scatter 操作。gather/scatter 功能为沿 dim 轴根据索引逐元素进行取值/赋值操作，此操作会有大量的随机地址，对性能造成巨大影响。

在 Megatron MoE 中对 gather/scatter 的调用主要是以下调用方式，通过对 index 做 expand 操作对维度进行扩展，再通过扩展后 index 对 hidden_states 进行逐元素取值/赋值。

```python
self.global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
local_hidden_states = torch.gather(global_hidden_states, 0, self.global_local_map)
```

### 2. 异步通信
在 Allgather dispatcher 分支中，permute 函数开头会分别对 hidden_states、max_ind、max_prob 三个数据做 allgather 通信，这些操作为串行操作，但各计算任务之间并非串行依赖关系。


## 解决方案
### 1. gather & scatter 算子替换
由于 index 是通过 expand 进行扩展的，因此它的每一行中的内容都是一致，而我们没有必要使用 gather/scatter 进行逐元素的操作，可通过 index 算子以及 indexput 算子进行逐行操作，对 gather/scatter 进行等价替换。


### 2. 异步通信
通过对通信任务进行重新排序，并使用 async=True 参数进行异步下发，达到计算和通信并行的目的。

## 使用场景
本优化策略适用于部署了Mcore MoE（Mixture of Experts）架构的深度学习模型,
并开启 `--moe-token-dispatcher-type allgather`。

## 使用方法
开启参数 `--moe-permutation-async-comm`。

## 使用效果
根据实际测试数据显示，类DeepSeekV2十亿参数级别的MoE模型，采用上述优化措施后，端到端训练过程中的性能提升了约10%。这意味着不仅加快了模型收敛速度，同时也降低了达到相同精度水平所需的计算资源消耗。