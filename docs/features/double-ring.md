# Double Ring Attention长序列并行

## 问题分析

在训练长序列的大语言模型时需要大量的内存资源和计算资源。业界已经提出了基于头并行和基于上下文并行的序列并行方法。在Attention块中，头并行方法保留整个序列，并行计算不同头的注意力，而上下文并行方法将QKV张量沿序列维度分成块。然而，当应用于大规模极长序列的大语言模型时，这两种方法都面临限制。首先，头并行遇到可扩展性问题。在头并行中，序列并行的程度本质上不能超过注意力头的数量。因此，头并行能够扩展的程度存在一个上限。其次，上下文并行遇到了通信效率问题。上下文并行采用了点对点（P2P）通信原语。然而，P2P遇到了低节点内带宽利用率和低节点间网络资源利用率的问题。这一瓶颈使得在扩展上下文并行维度时难以将通信与计算重叠。

## 解决方案

支持Double Ring Attention算法，进一步加速原始Ring Attention实现。算法细节参见原文：
> LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism (https://arxiv.org/pdf/2406.18485)

### 解决思路

原有的Ring Attention借鉴了分块Softmax原理，在不需要获取整个序列的完整矩阵情况下进行分块attention计算。 以分块方式执行自注意力和前馈网络计算，跨多个设备分布序列维度。具体地，该方法在进程之间构建注意力计算块的环状通信结构（Ring），每个进程具有一个切分后的本地QKV块。在计算完本地的attention后，通过向后发送和向前获取KV块，遍历进程设备环，以逐块的方式进行注意力和前馈网络计算。同时，本地的attention计算和KV块的通信理想情况下可以互相掩盖，从而消除了额外引入的通信开销。另外该方案在计算attention的过程中全程不需要数据拼接，支持的序列长度理论上可以无限拓展。 在此基础上Double Ring Attention算法采用分布式注意力机制，通过双环结构（Double Ring Attention）来优化计算和内存使用。

## 使用场景

已开启Ring Attention的训练场景

Ring Attention使能方式参考[此处](ring-attention-context-parallel.md)


## 使用方法

开启Ring Attention的训练场景中，将`--cp-window-size`设置为大于1的整数，即可使能Double Ring Attention算法，优化原始Ring Attention性能。

| 重要参数                   | 参数说明                                                                                                                                        |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| --cp-window-size [int] | 默认为`1`，即使用原始的Ring Attention算法,将`--cp-window-size`设置为大于1的整数，即可使能Double Ring Attention算法,该参数为Double Ring Attention算法中双层Ring Attention的内层窗口大小； |

## 使用效果

利用多个计算设备对输入序列进行并行切分，通过双环结构（Double Ring Attention）提升计算效率。


## 注意事项

1. 需要确保`--context-parallel-size`能被`--cp-window-size`整除。
2. 内层窗口`--cp-window-size`增大时，通信与计算并发程度更高，但是计算、通信并发时可能由于片上内存带宽抢占，整体效率下降，需要结合实际场景进行调试，例如Llama2裁剪模型32k序列长度，cp为16且无其他并行切分时，实测内层窗口大小为2时性能最优。