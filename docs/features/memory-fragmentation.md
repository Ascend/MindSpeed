# 内存碎片优化

## 问题分析

在模型训练的过程中，需要大量的显存来存储模型参数、中间计算结果、优化器状态以及批量输入数据等。频繁地申请和释放内存空间容易引发内存碎片问题。<br />

## 解决方案

通过对不同生命周期的tensor进行分别管理，以减少内存碎片。

### 解决思路:

#### **1、识别不同生命周期的tensor**

   一次训练过程中，长生命周期的tensor主要有四种：  
   （1）模型初始化创建的权重、梯度等。  
   （2）在前向时产生的激活值。  
   （3）用户没有进行垃圾回收的tensor会保留到下一个step。  
   （4）梯度收敛后产生的优化器状态tensor。  

####  **2、使用不同的内存池隔离管理**
   （1）将识别出的长短生命周期tensor放入不同的内存池分别管理。  
   （2）对长生命周期的大tensor精准分配与tensor大小相同的block，并采取优化后的多级分配策略，以避免长生命周期tensor对应的内存池产生碎片。<br />

## 使用场景
该特性主要用于训练场景，如果用户发现NPU报内存不足(out of memory)的错误，reserved和allocated的内存差距过大时(如reserved-allocated>1G)，则 <br />
说明torch中可能产生了较多的内存碎片，此时可考虑开启该特性以减少内存碎片，避免内存不足的问题 <br />
**示例** ：<br />
Tried to allocated 3384.00 MiB (NPU 2; 61.22 GiB total capacity; 53.87 GiB already allocated; 53.87 GiB current activate; 1.59 GiB free; <br />
56.60 GiB reserved in total by PyTorch), 发现reserved-allocated=2.73G，碎片较多，可以考虑开启该特性

## 使用方法
设置环境变量'MEMORY_FRAGMENTATION = 1'，即开启内存碎片优化特性。

## 使用效果
主要收益场景及配置： <br />

|       模型       |                                      参数                                      |  NPU卡数  |
|:--------------:|:----------------------------------------------------------------------------:|:-------:|
|   llama2-7B    |  seq-length=4096、mico-batch-size=4、global-batch-size=16、TP=8、PP=1、DP=1、开启FA  | 8卡（单机）  |
|   llama2-7B    |  seq-length=6144、mico-batch-size=4、global-batch-size=16、TP=8、PP=1、DP=1、开启FA  | 8卡（单机）  |
|   llama2-13B   |  seq-length=8192、mico-batch-size=4、global-batch-size=16、TP=8、PP=1、DP=1、开启FA  | 8卡（单机）  |
|   llama2-13B   |  seq-length=4096、mico-batch-size=2、global-batch-size=16、TP=8、PP=1、DP=1、开启FA  | 16卡（双机） |
|   llama2-13B   |  seq-length=6144、mico-batch-size=2、global-batch-size=16、TP=8、PP=1、DP=1、开启FA  | 16卡（双机） |
|   llama2-13B   |  seq-length=8192、mico-batch-size=2、global-batch-size=16、TP=8、PP=1、DP=1、开启FA  | 16卡（双机） |
|   llama2-70B   | seq-length=4096、mico-batch-size=2、global-batch-size=1024、TP=8、PP=4、DP=1、开启FA | 32卡（4机） |
|   llama2-70B   | seq-length=6144、mico-batch-size=2、global-batch-size=1024、TP=8、PP=4、DP=1、开启FA | 32卡（4机） |
|   llama2-70B   | seq-length=8192、mico-batch-size=2、global-batch-size=1024、TP=8、PP=4、DP=1、开启FA | 32卡（4机） |


## 注意事项：<br />
由于该特性在内存充足时倾向于新申请内存，而非将已申请的内存空间碎片化，因此在少量情况下可能和hccl抢占内存，hccl在内存不足时无法通过torch释放额外预留的空闲空间，<br />
从而报hccl内存不足的错误。此问题可以通过设置torch_npu.npu.set_per_process_memory_fraction接口来设置允许torch占用的内存上限来解决该问题 <br />
**接口设置**：<br />
位置：MindSpeed/mindspeed/core/memory/memory_fragmentation/memory_recorder.py <br />
添加：torch_npu.npu.set_per_process_memory_fraction(x)，其中x为想要限制torch占用内存的最高比例，例如x设置为0.94，表示torch最多占用"单卡内存*0.94"的内存 <br />
