# 重计算流水线独立调度
## 问题分析

在目前的流水线调度中，重计算由反向计算触发，与反向计算绑定在一起调度，意味着重计算需要等待下一个stage返回梯度才可以开始计算。然而重计算并不需要用到反向计算的梯度，这导致bubble的增多和性能的下降。

## 解决方案

为了将重计算和反向计算独立调度，需要将重计算的调度修改为由调度器主动触发，并修改调度器，将重计算作为一个调度单元加入到调度器中，这使我们获得了自由地插入或去除部分重计算的能力，进而可以在内存和性能方面做出优化。

### 解决思路
通过torch的saved_tensors_hooks实现一种新的重计算方法，在反向计算前合适的时机主动触发或者直接去除部分重计算，从而实现对内存或性能的优化。

## 使用场景

在pipelining_with_interleaving调度中，若用户未开启重计算，则可以利用bubble主动插入重计算，以极小的性能代价换取内存峰值的降低，将需要保留激活值的前向计算块的个数减少到pp * vp。
<p align="center"> <img src="../../sources/images/ripipe_a.png" height="154px" width="972px"></p>

在pipelining_with_interleaving调度中，若用户已开启重计算，则可以通过解除重计算与后一个stage的反向计算间的依赖关系从而提前重计算，以及去除模型最后一层的重计算，实现计算性能的提升。
<p align="center"> <img src="../../sources/images/ripipe_b.png" height="122px" width="954px"></p>

## 使用方法

脚本中添加： --recompute-in-bubble 可开启利用bubble进行重计算功能，实现内存节省。
使用条件：必须开启虚拟流水并行特性，使用此功能前不能开启重计算，recompute_num_layers参数需为None或0。

脚本中添加： --recompute-in-advance 可开启提前重计算以及去除不必要重计算功能，实现训练性能提升。
使用条件：必须开启虚拟流水并行特性，使用此功能前需要开启重计算，且不支持recompute_method为uniform，recompute_num_layers不能为None或0。

#### 注意：
两者不可同时开启