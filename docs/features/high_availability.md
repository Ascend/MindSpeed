# 高可用特性

## 问题分析

当前大模型在训练时，尤其是在多卡训练时一旦发生故障，可能没有及时保存权重，从而导致距离上一次保存权重之间的训练无效，浪费资源以及时间。

## 解决方案

高可用特性提供临终遗言的功能，可以在触发报错时在当前步保存ckpt，在之后拉起训练时load这个ckpt，保证训练框架的可用性。

### 解决思路:

分布式优化器的思想是通过将优化器状态均匀地分布在数据并行组中来节省内存。基于该思想，设计了将数据并行组切分成两个副本数据并行组的方案，副本优化器将优化器状态均匀分布在副本数据并行组，实现优化器状态均有备份。结合华为自研的高可用框架，可实现以下功能：
1. 训练过程中，支持故障场景保存临终checkpoint，训练结果0损失。

2. 训练过程中，支持HBM的UCE故障检测，并完成在线修复，达到Step级重计算。

## 使用方法

（1）安装华为自研高可用框架mindio_ttp.whl mindio_ttp相关说明：https://www.hiascend.com/document/detail/zh/mindx-dl/60rc1/mindio/mindiottp

（2）对于开启此功能，设置`--enable-high-availability`使能即可调用。

（3）设置`enable-optimizer-state-local-copy`使能保存上一步优化器状态，内存会进一步增加，默认可关闭。

## 使用效果

使用后在故障后触发UCE流程，方便继续训练。

开启高可用特性时，副本优化器使用的静态内存有所增加，每个参数的理论字节数为（其中“d”是数据并行大小）：

|                                  | Non-distributed optim | Distributed optim | Replica optim |
|----------------------------------| ------ | ------ |---------------|
| fp16/bf16 param, fp16/bf16 grads | 20 | 4 + 16/d | 4 + 32/d       |
| fp16/bf16 param, fp32 grads      | 18 | 6 + 12/d | Supporting      |
| fp32 param, fp32 grads           | 16 | 8 + 8/d  | Supporting      |
