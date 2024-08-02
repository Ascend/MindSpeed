# 内存碎片优化

## 问题分析

在模型训练的过程中，需要大量的显存来存储模型参数、中间计算结果、优化器状态以及批量输入数据等。频繁地申请和释放内存空间容易引发内存碎片问题。

## 解决方案

通过对不同生命周期的tensor进行分别管理，以减少内存碎片。

### 解决思路:

#### **1、识别不同生命周期的tensor**

一次训练过程中，长生命周期的tensor主要有四种：
（1）模型初始化创建的权重、梯度等。
（2）在前向时产生的激活值。
（3）用户没有进行垃圾回收的tensor会保留到下一个step。
（4）梯度收敛后产生的优化器状态tensor。

#### **2、使用不同的内存池隔离管理**

（1）将识别出的长短生命周期tensor放入不同的内存池分别管理。
（2）对长生命周期的大tensor精准分配与tensor大小相同的block，并采取优化后的多级分配策略，以避免长生命周期tensor对应的内存池产生碎片。

## 使用场景

该特性主要用于训练场景，如果用户发现NPU报内存不足(out of memory)的错误，reserved和allocated的内存差距过大时(如reserved-allocated>1G)，则
说明torch中可能产生了较多的内存碎片，此时可考虑开启该特性以减少内存碎片，避免内存不足的问题。

**示例** ：
Tried to allocated 3384.00 MiB (NPU 2; 61.22 GiB total capacity; 53.87 GiB already allocated; 53.87 GiB current
activate; 1.59 GiB free;
56.60 GiB reserved in total by PyTorch), 发现reserved-allocated=2.73G，碎片较多，可以考虑开启该特性。

## 使用方法

脚本中设置参数`--memory-fragmentation`，即开启内存碎片优化特性。

## 使用效果

主要收益场景及配置：

| 模型           | 参数                                                                          | NPU卡数    | 显存收益        |
|--------------|-----------------------------------------------------------------------------|----------|-------------|
| llama2-7B    | seq-length=4096、mico-batch-size=4、global-batch-size=16、TP=8、PP=1、DP=1、开启FA  | 8卡（单机）   | 3%（1.71G）   |
| llama2-7B    | seq-length=6144、mico-batch-size=4、global-batch-size=16、TP=8、PP=1、DP=1、开启FA  | 8卡（单机）   | 2.4%（1.22G） |
| llama2-13B   | seq-length=8192、mico-batch-size=4、global-batch-size=16、TP=8、PP=1、DP=1、开启FA  | 8卡（单机）   | 3.8%（2.19G） |
| llama2-13B   | seq-length=4096、mico-batch-size=2、global-batch-size=16、TP=8、PP=1、DP=1、开启FA  | 16卡（双机）  | 1.2%（0.67G） |
| llama2-13B   | seq-length=6144、mico-batch-size=2、global-batch-size=16、TP=8、PP=1、DP=1、开启FA  | 16卡（双机）  | 0.5%（0.31G） |
| llama2-13B   | seq-length=8192、mico-batch-size=2、global-batch-size=16、TP=8、PP=1、DP=1、开启FA  | 16卡（双机）  | 3.1%（1.71G） |
| llama2-70B   | seq-length=4096、mico-batch-size=2、global-batch-size=1024、TP=8、PP=4、DP=1、开启FA | 32卡（4机）  | 2.2%（1.28G） |
| llama2-70B   | seq-length=6144、mico-batch-size=2、global-batch-size=1024、TP=8、PP=4、DP=1、开启FA | 32卡（4机）  | 2.5%（1.4G）  |
| llama2-70B   | seq-length=8192、mico-batch-size=2、global-batch-size=1024、TP=8、PP=4、DP=1、开启FA | 32卡（4机）  | 2%（1.21G）  |

## 注意事项：

由于该特性在内存充足时倾向于新申请内存，而非将已申请的内存空间碎片化，因此在少量情况下可能和hccl抢占内存，hccl在内存不足时无法通过torch释放额外预留的空闲空间，
从而报hccl内存不足的错误。此问题可以通过设置torch_npu.npu.set_per_process_memory_fraction接口来设置允许torch占用的内存上限来解决该问题

**接口设置**：
位置：MindSpeed/mindspeed/core/memory/memory_fragmentation/memory_recorder.py
添加：torch_npu.npu.set_per_process_memory_fraction(x)，其中x为想要限制torch占用内存的最高比例，例如x设置为0.94，表示torch最多占用"单卡内存*0.94"的内存。

**注**：由于内存碎片优化与自适应选择重计算两个特性都修改了PyTorch内存管理模块，这两个特性都打开会存在冲突，mindspeed进行了assert判断。
