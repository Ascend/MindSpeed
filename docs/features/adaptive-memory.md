# 自适应内存优化

## 问题分析

在大模型训练中，重计算特性可以有效地减少显存使用，但是策略较为固定，无法最大限度使用显存资源。

## 解决方案

为了在最大限度地利用NPU显存的同时，提高模型训练的性能，我们支持通过自动调整训练内存大小来自动选择计算策略。这一特性称为自适应内存优化。

### 解决思路

自适应内存优化设计主要包括：自适应无损swap、 策略生成、策略搜索、SwapManager功能以及内存管理等几部分。

- 策略生成依赖自适应无损swap去生成策略

- 策略搜索依赖SwapManager功能及时将tensor换到CPU，避免OOM导致训练中断。

自适应内存优化策略流程如下图所示：

<p align="center"> <img src="../../sources/images/adaptive_memory_a.png.png" height="500px" width="400px"></p>

SwapManager功能需要内存管理适配PTA的NPUPluggableAllocator接口拦截OOM，让SwapManager功能可以介入，流程如下图所示：
<p align="center"> <img src="../../sources/images/adaptive_recompute_b.png" height="300px" width="500px"></p>

## 使用场景

该特性主要用于训练场景，如果用户发现开启了全重计算功能后， NPU显存剩余较多，此时若想充分利用显存，从而提高训练性能，可以考虑开启该特性。

## 使用方法

在训练脚本中添加`--adaptive-memory-optimization`

注意：
1. 当前自适应内存优化与全重计算、自适应选择重计算、预取特性swap-attention、 recompute-in-bubble等不兼容。 
2. 目前自适应内存优化已能够管理一部分使用torch.autograd.Function修饰的auto_function类

   - 在调用auto_function的文件中 添加 `from mindspeed.core.memory.adaptive_memory.adaptive_memory_function import adapt_mem_func_wrapper`
   - 将 `auto_function.apply(*args)` 修改为 `adapt_mem_func_wrapper(auto_function, *args)`
   - 以mindspeed.moe.pipe_experts中的PipeExpert类的调用为例，在mindspeed.moe.moe_layer文件中添加`from mindspeed.core.memory.adaptive_memory.adaptive_memory_function import adapt_mem_func_wrapper`，将`expert_output = PipeExpert.apply(*args)`修改为`expert_output = adapt_mem_func_wrapper(PipeExpert, *args)`

## 使用效果

这里的gpt-175B是经过裁剪后的

gpt-175B:

| 特性         | 参数                                                                                                                        | NPU卡数    | TFLOPs      | 收益        |
|------------|---------------------------------------------------------------------------------------------------------------------------|----------|-------------| -------------|
| adaptive-memory-optimization    | seq-length=8192、micro-batch-size=10、global-batch-size=40、TP=8、PP=1、DP=1、CP=1、NL=8、hidden-size=12288                        | 8卡（单机）   | 165.90      | - |
| 全重计算     | seq-length=8192、micro-batch-size=10、global-batch-size=40、TP=8、PP=1、DP=1、CP=1、NL=3、hidden-size=12288、recompute-num-layers=3 | 8卡（单机）   | 145.93      | 13.68% |


| 特性         | 参数                                                                                                                        | NPU卡数    | TFLOPs | 收益     |
|------------|---------------------------------------------------------------------------------------------------------------------------|----------|--------|--------|
| adaptive-memory-optimization    | seq-length=8192、micro-batch-size=3、global-batch-size=9、TP=2、PP=4、DP=1、CP=1、NL=8、hidden-size=12288                          | 8卡（单机）   | 76.30  | -      |
| 全重计算     | seq-length=8192、micro-batch-size=3、global-batch-size=9、TP=2、PP=4、DP=1、CP=1、NL=8、hidden-size=12288、recompute-num-layers=1 | 8卡（单机）   | 66.50  | 14.17% |

| 特性         | 参数                                                                                                                              | NPU卡数    | TFLOPs | 收益     |
|------------|---------------------------------------------------------------------------------------------------------------------------------|----------|--------|--------|
| adaptive-memory-optimization    | seq-length=8192、micro-batch-size=2、global-batch-size=8、TP=2、PP=4、VPP=2、DP=1、CP=1、NL=8、hidden-size=12288                          | 8卡（单机）   | 86.10  | -      |
| 全重计算     | seq-length=8192、micro-batch-size=2、global-batch-size=8、TP=2、PP=4、VPP=2、DP=1、CP=1、NL=8、hidden-size=12288、recompute-num-layers=1 | 8卡（单机）   | 75.10  | 14.65% |

## 注意事项

1. 由于自适应内存优化与内存碎片优化两个特性都修改了PyTorch内存管理模块，这两个特性都打开会存在冲突，mindspeed进行了assert判断。
2. 由于自适应内存优化依赖cpu的绑核，因此需要保证运行环境内含有npu-smi以及lspci命令。
安装命令：yum install pciutils



