# 自适应选择重计算

## 问题分析

重计算特性可以有效的减少显存使用，但是策略较为固定，无法最大限度使用显存资源。

## 解决方案

为了在最大限度地利用 NPU 显存的同时，提高模型训练的性能，我们支持通过自动调整训练内存大小来自动选择重新计算策略。这一特性称为自适应选择重计算。

### 解决思路

自适应选择重计算设计主要包括重计算策略搜索、SwapManager 功能和内存管理三大部分。

其中重计算策略搜索依赖 SwapManager 功能及时将 tensor 换到 CPU，避免 OOM 导致训练中断。

自动选择重计算策略流程如下图所示：

<p align="center"> <img src="../../sources/images/adaptive_recompute_a.png" height="500px" width="400px"></p>

SwapManager 能需要内存管理适配 PTA 的 NPUPluggableAllocator 接口拦截 OOM，让 SwapManager 功能可以介入，流程如下图所示：
<p align="center"> <img src="../../sources/images/adaptive_recompute_b.png" height="300px" width="500px"></p>

## 使用场景

该特性主要用于训练场景，如果用户发现开启了全重计算功能后， NPU显存剩余较多，此时若想充分利用显存，从而提高训练性能，可以考虑开启该特性。

## 使用方法

1. 在训练脚本中添加`--adaptive-recompute-device-swap`。注意：当前自适应选择性重计算与全重计算、重计算独立调度流水线ripipe、激活函数重计算、预取特性swap-attention等不兼容。
2. （可选）支持手动调整训练内存大小来自动选择重计算策略，请使用`--adaptive-recompute-device-size`进行设置来指定自适应选择重计算策略的训练内存大小（单位：MB）。内存>0为有效内存，最大内存限度为device最大内存。在该范围内自适应重计算才可以进行最优策略搜寻，不在有效内存范围内会使用读取到的device最大内存信息作为默认值。需要注意的是内存设定较小时，性能会与全重计算一致。该方式如果发生OOM，您需要重新选择一个新的内存值来重启模型训练。您也可以通过二分法的方式获得最优解，对该特性不熟悉请勿使用此选项。
3. （可选）支持设置停止profiling的训练step，请使用`--adaptive-recompute-profiling-step`进行设置。该参数需要设置为>0的整数。默认在第10步停止profiling。若该值<=0，则采用默认值10，推荐设置该值>5。当step<5或者>总步数的1/10时，会有告警信息，但不影响正常训练，不会对性能和精度有任何影响。

## 使用效果

相比全重计算，Llama2-7B场景下，性能提升约 16.29%，Llama2-13B 性能提升约12.05%。

## 注意事项

由于自适应选择重计算与内存碎片优化两个特性都修改了PyTorch内存管理模块，这两个特性都打开会存在冲突，mindspeed进行了assert判断。
