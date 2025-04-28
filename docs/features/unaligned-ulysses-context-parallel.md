# 非对齐Ulysses长序列并行

## 背景与挑战

随着生成式AI和科研模型领域的发展，长序列训练变得越来越重要。然而，传统的Ulysses设计要求序列长度（sequence length）必须能够被长序列并行大小（Context Parallel size, CP size）整除。这在处理动态或不规则输入时带来了限制，特别是在多模态应用中，输入数据的序列长度可能无法预测且经常变化。因此，需要一种机制来支持这些非对齐情况下的操作，以适应更广泛的应用场景。


## 解决方案

为了解决传统Ulysses设计在处理非对齐序列长度时的局限性，“非对齐 Ulysses”机制通过引入一个抽象基类 `GatherSizeCalculator` 来提供计算 gather size 的接口。Gather size 通常指的是经过 （Ulysses 机制中的）all-to-all 通信后，输出张量在 `gather_idx` 维度上的大小。该基类定义了任何具体实现都必须提供的 `calculate()` 方法，用于返回整数形式的 gather size 或者 None。

基于此接口，实现了两种具体的策略：`DefaultGatherSizeCalculator` 和 `DynamicGatherSizeCalculator`。前者默认返回 None，意味着使用对齐的Ulysses长序列并行；后者则根据当前批次的注意力掩码序列长度动态计算 gather size。这种设计使得系统能够灵活应对不同场景的需求，尤其是在多模态领域中处理 sequence length 不能被 CP size 整除的情况时尤为重要。

此外，在 `UlyssesContextAttention` 类中，允许用户注入一个 `gather_size_calculator` 实例，使得系统能够灵活地选择不同的 gather size 计算方法，从而适应不同场景的需求。

## 使用场景

“非对齐 Ulysses”功能适用于以下几种典型场景：

- **多模态学习**：当处理图像、视频、文本等多种类型的数据时，由于不同类型数据的序列长度差异较大，难以统一到固定的CP size。
- **实时数据分析**：在处理流数据时，数据到达的时间不确定，导致每次处理的序列长度也可能不同。
- **个性化推荐系统**：用户行为数据的序列长度通常各不相同，这种情况下也需要支持非对齐的操作。

## 使用方法

为了利用“非对齐 Ulysses”功能，用户可以根据业务需求传入基于 `GatherSizeCalculator` 基类的自定义 Calculator，或者直接使用预定义的 `DynamicGatherSizeCalculator`。以下是基本步骤：

1. 启动脚本中配置长序列并行大小大于1`--context-parallel-size [int]`。 同时配置`--context-parallel-algo ulysses_cp_algo`。
2. 创建一个继承自 `GatherSizeCalculator` 的自定义计算器类，并实现 `calculate()` 方法。在初始化 `UlyssesContextAttention` 对象时，通过构造函数参数传入自定义的 `gather_size_calculator` 实例。
3. 如果不需要复杂的自定义逻辑，可以直接使用 `DynamicGatherSizeCalculator`，它会自动根据当前批次的注意力掩码序列长度计算 gather size。

```python
# 示例代码
import megatron.core.parallel_state as ps
from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import UlyssesContextAttention, GatherSizeCalculator, DynamicGatherSizeCalculator
from your_library import FlashSelfAttention

# 自定义 GatherSizeCalculator
class CustomGatherSizeCalculator(GatherSizeCalculator):
    def calculate(self, *args, **kwargs):
        # 示例逻辑
        return kwargs.get("gather_size", None)


core_attention = FlashSelfAttention()
# 根据实际情况，使用预定义DynamicGatherSizeCalculator()或自定义CustomGatherSizeCalculator()
calculator = DynamicGatherSizeCalculator()
ulysses_attention = UlyssesContextAttention(core_attention, ps.get_context_parallel_group(),
                                            gather_size_calculator=calculator)

```

## 使用效果

通过引入“非对齐 Ulysses”，系统提升了对不同输入长度的适应能力。这不仅解决了传统 Ulysses 在处理动态或不规则输入序列时遇到的问题，而且保持了良好的扩展能力。

## 注意事项

1. 非对齐Ulysses长序列并行不支持在legacy分支使用，即不支持和`--use-legacy-models`同时开启。
2. 非对齐Ulysses长序列并行暂不兼容Ulysses长序列并行KV缓存优化，即启动脚本设置了--context-parallel-kv-cache-policy为full或者half，系统将自动切换回使用对齐的Ulysses长序列并行机制。