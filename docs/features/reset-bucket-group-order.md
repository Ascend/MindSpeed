# bucket重排算法

## 背景与挑战

在大模型的训练过程中，模型的定义顺序和执行不一致是非常常见的问题，尤其是重定义常见的transformer组件或者使用多模态大模型时。这直接导致overlap-param-gather参数时会出现精度问题和计算通信串行的问题。
目前megatron 0.12.1的方案解决了精度问题，然而不可避免地会出现计算和通信串行的问题。
## 解决方案

为解决上述问题，引入了针对于参数分桶重排的策略，通过记录第一次迭代的bucket_group顺序来实现后续迭代的计算与通信的流水掩盖，有效提升资源利用率。

### 在开启overlap-param-gather的条件下，打开 `--reset-bucket-group-order`参数
在第一次迭代过程中，我们记录bucket-group触发的顺序，第一次前向结束后，将bucket的顺序记录完整。第二次迭代开始时，除了第一个bucket触发无法和计算重叠，后续每一次预取下一个桶的通信都会和当前的计算进行重叠。


## 使用场景

该特性适用于采用数据并行策略的训练场景，特别适合模型定义顺序非常混乱的时候，此时桶的通信是无序的，计算和通信存在大量串行，开启overlap-param-gather效果不够显著，开启reset-bucket-group-order参数可以在盘古上提升吞吐0.85%
，在手动调乱模型定义顺序的llama2上吞吐提升1%左右。
## 使用方法

* 要启用bucket重排算法功能，需在训练配置中加入以下参数：
    `--reset-bucket-group-order`
* 确保同时开启了以下三个参数。
    `--use-distributed-optimizer`
    `--overlap-grad-reduce`
    `--overlap-param-gather`




