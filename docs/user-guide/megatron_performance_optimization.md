# 基于 Megatron 并行策略的性能优化

## 概述

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)是NVIDIA提出的一种分布式训练加速库，支持数据并行、模型并行和序列并行等特性，在大模型训练中得到广泛应用。经过[MindSpeed](https://gitee.com/ascend/MindSpeed)昇腾平台的兼容性适配，现已在昇腾平台上支持原生并行策略。
虽然适配了众多并行策略，但是模型在长文本场景依旧存在空间和时间复杂度较高的问题。MindSpeed 从序列维度出发，实现了众多序列并行方法，解决了序列维度扩展问题。
本手册从序列并行的角度进行介绍，指导用户使用 MindSpeed 进行 Megatron 性能优化。本手册主要介绍以下四种序列并行算法及使用方法：
- Ulysses长序列并行
- Ring Attention长序列并行
- Double Ring Attention长序列并行
- 混合长序列并行

## Ulysses长序列并行

### 算法思路

Ulysses 将各个样本在序列维度上分割给参与的计算设备。然后，在 attention 计算之前，它对已分割的查询(Q)、键(K)和值(V)执行 all-to-all 通信操作，以使每个计算设备接收完整的序列，但仅用于注意力头的非重叠子集。这使得参与的计算设备可以并行计算不同的注意力头。最后，Ulysses 还使用另一个 all-to-all 来在注意力头上收集结果，同时重新在序列维度上进行分区。


### 使用场景

num_head 要能被 tp_size*cp_size 整除。

### 使用方法

设置`--context-parallel-size`，默认为1，根据用户需求配置。
同时设置`--context-parallel-algo ulysses_cp_algo`。

#### 执行脚本
拷贝 MindSpeed 目录下的 tests_extend 目录到 Megatron 目录， 并在 Megatron 目录下执行如下命令：

```
bash tests_extend/system_tests/feature_tests/ulysses.sh
```
### 使用效果

利用多个计算设备对输入序列进行并行切分，降低单设备的内存消耗，相比不开启序列并行单步耗时增加，相比重计算计算效率提升。


## Ring Attention长序列并行


### 算法思路

Ring Attention借鉴了分块Softmax原理，在不需要获取整个序列的完整矩阵情况下进行分块attention计算。因此作者提出以分块方式执行自注意力和前馈网络计算，跨多个设备分布序列维度。具体地，该方法在进程之间构建注意力计算块的环状通信结构（Ring），每个进程具有一个切分后的本地QKV块。在计算完本地的attention后，通过向后发送和向前获取KV块，遍历进程设备环，以逐块的方式进行注意力和前馈网络计算。同时，本地的attention计算和KV块的通信理想情况下可以互相掩盖，从而消除了额外引入的通信开销。另外该方案在计算attention的过程中全程不需要数据拼接，支持的序列长度理论上可以无限拓展。



### 使用场景

当使用GPT类模型进行训练，同时数据进MoE层时实际序列长度8K以上。

不同于Ulysses方案，该方案不需要确保head_size被cp_size整除。

可兼容FlashAttention，目前已默认开启FlashAttention。

如果想要使得计算和通信可以互相掩盖，理论上需要确保每个计算块分到的序列长度$c \geq F/B$。其中F是每个device的FLOPS，B是每个device间的带宽。具体推导过程参见原文。在实践中，需要确保每个计算块分到的序列长度足够大，才能较好掩盖。


### 使用方法

| 重要参数                                     | 参数说明                                                                                                                                                         |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --context-parallel-size [int]            | 开启CP对应的数量，默认为1，根据用户需求配置。                                                                                                                                     |
| --seq-length [int]                       | 输入序列的长度。                                                                                                                                                     |
| --use-cp-send-recv-overlap               | 建议开启，开启后支持send receive overlap功能。                                                                                                                            |
| --attention-mask-type [general/causal]   | 可选，设置Mask计算类型，默认是causal（倒三角）Mask计算，设置general代表全量计算。                                                                                                          |
| --context-parallel-algo megatron_cp_algo | 长序列并行算法选项，默认项为`ulysses_cp_algo`，当设置为`megatron_cp_algo`时开启Ring Attention。                                                                                     |
| --megatron-cp-in-bnsd                    | 开启后，FA使用BNSD计算。                                                          |
| --cp-window-size [int]                   | 可选，默认为`1`，即使用原始的Ring Attention算法；当设置为大于`1`时，即使用Double Ring Attention算法，优化原始Ring Attention性能，--cp-window-size即为算法中双层Ring Attention的内层窗口大小，需要确保cp_size能被该参数整除。|

#### 执行脚本
首先拷贝 MindSpeed 目录下的 tests_extend 目录到 Megatron 目录，然后在该目录下将`tests_extend/system_tests/feature_tests/ring_attention.sh`文件中`cp-window-size`参数改为1，最后执行如下命令：

```
bash tests_extend/system_tests/feature_tests/ring_attention.sh
```

### 使用效果

利用多个计算设备对输入序列进行并行切分，降低单设备的内存消耗，相比不开启序列并行单步耗时增加，相比重计算计算效率提升。


### 注意事项

1. 开启Context Parallel时需要同时开启Flash Attention特性，否则特性不支持。
2. 在使用GPT类模型进行训练的场景下，建议`attention-mask-type`设置为`causal`。
3. 在8k的序列长度情况下，由于计算的时间缩短，cp功能分割之后的send receive的时间反而会长于计算时间，造成性能的下降，所以建议配置seq-length / context-parallel-size> 8k以获取最佳效果。具体公式参考：S/(Talpha) >= 1/(Wbeta)，其中，S=seq-length / context-parallel-size， T表示芯片的理论算力，alpha表示计算效率，W表示理论通信带宽，beta表示带宽利用率。
4. 内层窗口`--cp-window-size`增大时，通信与计算并发程度更高，但是计算、通信并发时可能由于片上内存带宽抢占，整体效率下降，需要结合实际场景进行调试，例如llama2裁剪模型32k序列长度，cp为16且无其他并行切分时，实测内层窗口大小为2时性能最优。


## Double Ring Attention长序列并行
### 算法思路

原有的Ring Attention借鉴了分块Softmax原理，在不需要获取整个序列的完整矩阵情况下进行分块attention计算。 以分块方式执行自注意力和前馈网络计算，跨多个设备分布序列维度。具体地，该方法在进程之间构建注意力计算块的环状通信结构（Ring），每个进程具有一个切分后的本地QKV块。在计算完本地的attention后，通过向后发送和向前获取KV块，遍历进程设备环，以逐块的方式进行注意力和前馈网络计算。同时，本地的attention计算和KV块的通信理想情况下可以互相掩盖，从而消除了额外引入的通信开销。另外该方案在计算attention的过程中全程不需要数据拼接，支持的序列长度理论上可以无限拓展。 在此基础上Double Ring Attention算法采用分布式注意力机制，通过双环结构（Double-Ring-Attention）来优化计算和内存使用。

### 使用场景

已开启Ring Attention的训练场景

Ring Attention使能方式参考[此处](ring-attention-context-parallel.md)


### 使用方法

开启Ring Attention的训练场景中，将`--cp-window-size`设置为大于1的整数，即可使能Double Ring Attention算法，优化原始Ring Attention性能。

| 重要参数                   | 参数说明                                                                                                                                        |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| --cp-window-size [int] | 默认为`1`，即使用原始的Ring Attention算法,将`--cp-window-size`设置为大于1的整数，即可使能Double Ring Attention算法,该参数为Double Ring Attention算法中双层Ring Attention的内层窗口大小； |

#### 执行脚本
拷贝 MindSpeed 目录下的 tests_extend 目录到 Megatron 目录， 并在 Megatron 目录下执行如下命令：

```
bash tests_extend/system_tests/feature_tests/ring_attention.sh
```
### 使用效果

利用多个计算设备对输入序列进行并行切分，通过双环结构（Double-Ring-Attention）提升计算效率。


### 注意事项

1. 需要确保`--context-parallel-size`能被`--cp-window-size`整除。
2. 内层窗口`--cp-window-size`增大时，通信与计算并发程度更高，但是计算、通信并发时可能由于片上内存带宽抢占，整体效率下降，需要结合实际场景进行调试，例如llama2裁剪模型32k序列长度，cp为16且无其他并行切分时，实测内层窗口大小为2时性能最优。




## 混合长序列并行

目前流行的序列并行方案，Ulysses和Ring Attention存在各自的局限性。

Ulysses需要确保attention head数可以被序列并行维度整除，在GQA、MQA场景下序列并行的大小有限制，导致序列长度的扩展有限。

Ring Attention的并行维度不受attention head数限制，因此理论上序列长度可以无限拓展。但相比于Ulysses，Ring Attention不能充分利用通信和计算带宽，在序列块大小较低时性能劣于Ulysses。

### 算法思路
对Ulysses和Ring Attention做融合，实现混合序列并行，以此解决两个方案各自缺陷。

### 使用场景

可兼容FlashAttention，目前已默认开启FlashAttention。

序列并行维度被分为Ulysses维度和ring attention维度，Ulysses维度和ring attention维度乘积即为序列并行维度。

### 使用方法

设置`--context-parallel-size`，默认为1，根据用户需求配置。

设置`--context-parallel-algo hybrid_cp_algo`，以使能混合序列并行。

设置`--ulysses-degree-in-cp`，需要确保`--context-parallel-size`可以被该参数整除且大于1。例如当设置`--context-parallel-size=8`时，可以设置`--ulysses-degree-in-cp=2`或`--ulysses-degree-in-cp=4`。

同时需要确保`--ulysses-degree-in-cp`可以被attention head数整除。

混合长序列并行支持Ring Attention长序列并行相关特性，包括send receive overlap功能、Mask计算类型配置。

#### 执行脚本
拷贝 MindSpeed 目录下的 tests_extend 目录到 Megatron 目录， 并在 Megatron 目录下执行如下命令：

```
bash tests_extend/system_tests/feature_tests/hybrid.sh
```

### 使用效果

利用多个计算设备对输入序列进行并行切分，降低单设备的内存消耗，相比不开启序列并行单步耗时增加，相比重计算计算效率提升。

