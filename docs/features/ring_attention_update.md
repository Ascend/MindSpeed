# Ascend Ring Attention Update 融合优化

## 问题分析
Ring Attention 是一种优化注意力机制的技术，它借鉴了分块 Softmax 的原理，通过分块的方式执行自注意力和前馈网络计算，从而在不需要获取整个序列的完整矩阵的情况下进行注意力计算。
Ring Attention 的核心思想是在进程之间构建注意力计算块的环状通信结构（Ring），每个进程具有一个切分后的本地 QKV 块。在计算完本地的 attention 后，
通过向后发送和向前获取 KV 块，遍历进程设备环，以逐块的方式进行注意力和前馈网络计算。在这个过程中，需要对在不同计算设备上计算得到的注意力结果，根据当前的 softmax 最大值和总和进行更新。这个更新过程涉及多个小算子，包括但不限于计算 softmax 的最大值等。面临计算延迟增加、内存使用效率低下、并行处理效率下降等问题。

## 解决方法
MindSpeed将注意力更新操作融合成一个算子，显著提高注意力更新计算的性能，优化内存使用，并简化代码结构。算子接口见[link](../ops/npu_ring_attention_update.md)。

## 使用方法
### 前提条件
开启Ring Attention长序列并行
`--context-parallel-size ${CP}  \
--context-parallel-algo megatron_cp_algo 
`
其中，CP大于1

设置`--use-fused-ring-attention-update`即可调用Ring Attention update融合算子。

## 使用效果 
启用融合算子后，能够显著提高注意力更新计算的性能，优化内存使用，并简化代码结构。
