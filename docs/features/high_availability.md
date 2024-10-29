# 高可用特性

## 问题分析

当前大模型在训练时，尤其是在多卡训练时一旦发生故障，可能没有及时保存权重，从而导致距离上一次保存权重之间的训练无效，浪费资源以及时间。

## 解决方案

高可用特性提供临终遗言的功能，可以在触发报错时在当前步保存ckpt，在之后拉起训练时load这个ckpt，保证训练框架的可用性。

### 解决思路:

分布式优化器的思想是通过将优化器状态均匀地分布在数据并行组中来节省内存。基于该思想，设计了将数据并行组切分成两个副本数据并行组的方案，副本优化器将优化器状态均匀分布在副本数据并行组，实现优化器状态均有备份。结合华为自研的高可用框架，可实现以下功能：
1. 训练过程中，支持故障场景保存临终checkpoint，训练结果0损失。

2. 训练过程中，支持片上内存故障检测，并完成在线修复，实现Step级重计算。

## 使用方法

（1）正确安装华为自研高可用框架mindio_ttp.whl mindio_ttp下载地址：[MindIO TTP 下载软件包-昇腾社区](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc2/mindio/mindiottp/mindiottp009.html)

（2）检测HDK版本，需支持使能devmem。

（3）对于开启此功能，设置总开关`--enable-high-availability`使能即可调用。

（4）设置`--enable-optimizer-state-local-copy`使能保存上一步优化器状态，内存会进一步增加，默认可关闭。

（5）设置`--enable-hbmfault-repair`使能进行片上内存故障在线修复功能的开关。

（6）设置`--enable-worker-reboot`使能进行ARF（空中加油）功能的开关。

## 使用效果

开启高可用特性时，副本优化器使用的静态内存有所增加，每个参数的理论字节数为（其中“d”是数据并行大小）：

|                                  | Non-distributed optim | Distributed optim | Replica optim | Local copy |
|----------------------------------|-----------------------|-------------------|---------------|------------|
| fp16/bf16 param, fp16/bf16 grads | 20                    | 4 + 16/d          | 4 + 32/d      | 40         |
| fp16/bf16 param, fp32 grads      | 18                    | 6 + 12/d          | 6 + 24/d      | 36         |
| fp32 param, fp32 grads           | 16                    | 8 + 8/d           | 8 + 16/d      | 32         |

使能`--enable-hbmfault-repair`后，在片上内存故障后触发修复流程，迅速继续训练；

使能`--enable-worker-reboot`，并配合支持相关功能的 MindX DL 组件共同使能后，在发生一般性故障时，进行进程级重启修复，继续训练。

## 适配限制

当前mindio_ttp仅支持Megatron框架通过MindSpeed使能高可用特性，在MindSpeed中均被`--enable-high-availability`总开关隔离，
详情查看项目官网，其他训练框架需进行额外的适配修改。

## 鸣谢

1. MindIO TTP 项目官网地址：
[MindIO TTP 项目介绍-昇腾社区](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc2/mindio/mindiottp)
