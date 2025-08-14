# 非对齐Ring长序列并行

## 背景与挑战

随着生成式AI和科研模型领域的发展，长序列训练变得越来越重要。然而，传统的Ring序列并行设计要求序列长度（sequence length）必须能够被长序列并行大小（Context Parallel size, CP size）整除。这在处理动态或不规则输入时带来了限制，特别是在多模态应用中，输入数据的序列长度可能无法预测且经常变化。因此，需要一种机制来支持这些非对齐情况下的操作，以适应更广泛的应用场景。


## 解决方案

为了解决传统Ring序列并行设计在处理非对齐序列长度时的局限性，“非对齐 Ring”机制通过建立形状协商协议，通信前先交换有效长度信息，并通过get_unaligned_cp_shapes隔离变化，用以获取当前rank长度和目标rank长度，在分块计算和通信时传递非均匀的序列，实现非均匀切分的RingAttention计算。


## 使用场景

“非对齐 Ring”功能适用于以下几种典型场景：

- **多模态学习**：当处理图像、视频、文本等多种类型的数据时，由于不同类型数据的序列长度差异较大，难以统一到固定的CP size。
- **实时数据分析**：在处理流数据时，数据到达的时间不确定，导致每次处理的序列长度也可能不同。
- **个性化推荐系统**：用户行为数据的序列长度通常各不相同，这种情况下也需要支持非对齐的操作。

## 使用方法

为了使用“非对齐 Ring”功能，用户需要在调用`ringattn_context_parallel`接口时，如下面示例代码所示，传入`shapes`参数。

```python
# 示例代码
 output = ringattn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, shapes=shapes)

```

`get_unaligned_cp_shapes`是非对齐Ring序列并行中的重要函数，`shapes`参数的主要作用是获取序列的非对齐切分后的子序列长度信息，函数中通过 shapes[block_id] 和 shapes[next_block_id] 的方式访问元素，所以`shapes`参数可以是任何支持通过 [] 操作符进行索引访问的数据结构，包括但不限于：
1. 列表（list）：如 [100, 100, 20]
2. 元组（tuple）：如 (100, 100, 20)
3. 字典（dict）：如 {0: 100, 1: 100, 2: 20}

`get_unaligned_cp_shapes`函数最终会返回一个包含两个元素的列表：[shapes[block_id], shapes[next_block_id]]，这两个元素分别对应 block_id 和 next_block_id 索引处的值。

```python
def get_unaligned_cp_shapes(shapes, block_id, next_block_id):
    if shapes is None:
        return None
    unaligned_cp_shapes = [shapes[block_id], shapes[next_block_id]]
    return unaligned_cp_shapes
```

## 使用效果

通过引入“非对齐 Ring”，系统提升了对不同输入长度的适应能力。这不仅解决了传统Ring序列并行在处理动态或不规则输入序列时遇到的问题，而且保持了良好的扩展能力。

## 注意事项

1. 非对齐Ring长序列并行当前仅支持--attention-mask-type为general的场景。