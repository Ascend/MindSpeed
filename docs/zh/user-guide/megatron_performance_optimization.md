# 基于Megatron并行策略的性能优化

## 概述

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)是NVIDIA提出的一种分布式训练加速库，支持数据并行、模型并行和序列并行等特性，在大模型训练中得到广泛应用。经过MindSpeed昇腾平台的兼容性适配，现已在昇腾平台上支持原生并行策略。

在长文本场景下，模型训练面临空间和时间复杂度较高的问题。MindSpeed从序列维度出发，实现了多种序列并行方法，解决了序列维度扩展问题。本手册从性能诊断到优化实践，全面指导用户使用MindSpeed进行Megatron性能优化。

## 性能诊断方法论

### 性能指标定义

性能优化的第一步是理解性能指标。对于一个batch而言，时间主要由以下部分构成：

```text
单batch总时间 = 数据加载时间 + 模型前反向时间 + 优化器时间 + 模型后处理时间 + 通信时间 + 调度时间
```

各组成成分介绍如下：

- **数据加载时间**：模型在加载自身所需要的数据（如图片、视频和文本等）的时间，包括将数据从硬件存储设备读取到CPU（Central Processing Unit）中、CPU中数据的预处理（编解码等操作）、CPU数据放到device上的时间。对于一些需要切分在若干张卡上的模型，数据加载还包括从数据加载卡广播到其他卡上的时间。
- **模型前反向时间**：深度学习模型的前向过程和反向过程的时间，即Forward和Backward过程，包含前向的数据计算和反向的数据微分求导的时间。
- **优化器时间**：模型参数更新时间。
- **模型后处理时间**：优化器更新后的时间，包括数据的后处理或者一些必要的同步操作，通常取决于模型特有操作。
- **通信时间**：单节点时卡之间和多节点时节点之间的通信时间。由于PyTorch的特殊机制，在通信和计算可以并行的情况下，表示未被计算掩盖的通信时间。
- **调度时间**：模型从CPU的指令到调用NPU侧的核（Kernel）所需要的时间。

### 调优流程

性能调优一般遵循以下五步流程：

```text
采集profiling数据 → 分析算子耗时 → 分析通信耗时 → 分析内存使用 → 选择优化策略
```

1. **采集profiling数据**：运行训练脚本并启用profiling功能。
2. **分析算子耗时**：识别耗时最长的算子，定位计算瓶颈。
3. **分析通信耗时**：查看通信时间占比，判断是否存在通信瓶颈。
4. **分析内存使用**：检查显存占用情况，判断是否存在内存瓶颈。
5. **选择优化策略**：根据瓶颈类型选择合适的优化方案。

### 性能数据采集

采集性能数据是分析性能问题、找到性能瓶颈的关键步骤。MindSpeed支持基于昇腾芯片采集profiling数据。

在训练脚本中启用profiling功能，常用参数如下：

```shell
# 启用profiling采集
python your_train_script.py \
    --profile \
    --profile-step-start 5 \
    --profile-step-end 6 \
    --profile-ranks 0 \
    --profile-level level1 \
    --profile-with-cpu \
    --profile-record-shapes \
    --profile-save-path ./profile_dir
```

**参数说明**：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--profile` | 启用性能数据采集 | False |
| `--profile-step-start` | 开始采集的step数（包含） | 0 |
| `--profile-step-end` | 结束采集的step数（不包含），设置为-1表示采集到训练结束 | -1 |
| `--profile-ranks` | 指定采集的rank，设置为-1时表示采集所有rank的profiling数据 | [0] |
| `--profile-level` | 采集级别：level0（仅算子耗时）、level1（算子+通信耗时）、level2（完整数据） | level0 |
| `--profile-with-cpu` | 是否采集CPU数据 | False |
| `--profile-record-shapes` | 是否采集计算shape（用于分析显存和计算量） | False |
| `--profile-save-path` | 采集数据保存路径 | ./profile_dir |

### 性能分析流程

采集到性能数据后，可通过[MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md)对性能数据进行可视化分析，定位性能瓶颈。

#### 分析维度

MindStudio Insight支持多维度性能分析：

| 分析维度 | 分析内容 | 定位目标 |
| --- | --- | --- |
| 算子耗时分析 | 识别耗时较长的算子 | 计算瓶颈 |
| 通信耗时分析 | 分析通信与计算的时间占比 | 通信瓶颈 |
| 内存分析 | 查看显存使用情况 | 显存瓶颈 |
| 流水线分析 | 分析流水线并行的空泡比例 | 并行效率 |

#### 分析流程

1. **数据导入**：将采集到的profiling数据导入MindStudio Insight。
2. **可视化分析**：查看算子耗时分布图、通信时间占比等。
3. **瓶颈定位**：根据分析结果定位性能瓶颈。
4. **优化建议**：根据瓶颈类型选择合适的优化策略。

### 瓶颈类型判断

根据分析结果，可将性能瓶颈分为以下几类：

| 瓶颈类型 | 判断依据 | 典型表现 |
| --- | --- | --- |
| **计算瓶颈** | 算子耗时占比高 | 单卡训练速度慢，GPU/NPU利用率低 |
| **通信瓶颈** | 通信时间占比高 | 多卡训练加速比不理想 |
| **内存瓶颈** | 显存占用接近上限 | 训练过程中出现OOM错误 |
| **数据加载瓶颈** | 数据加载时间占比高 | 训练过程中GPU/NPU空闲等待数据 |

## 序列并行优化方案

### Ascend Ulysses长序列并行

#### 算法思路

Ulysses 将各个样本在序列维度上分割给参与的计算设备。然后，在 attention 计算之前，它对已分割的查询(Q)、键(K)和值(V)执行 all-to-all 通信操作，以使每个计算设备接收完整的序列，但仅用于注意力头的非重叠子集。这使得参与的计算设备可以并行计算不同的注意力头。最后，Ulysses 还使用另一个 all-to-all 来在注意力头上收集结果，同时重新在序列维度上进行分区。

#### 使用场景

num_head要能被tp_size*cp_size整除。适合head数较多且能被并行维度整除的场景。

#### 使用方法

设置`--context-parallel-size`，默认为1，根据用户需求配置。
同时设置`--context-parallel-algo ulysses_cp_algo`。

具体使用方式参考如下示例：

1. 拷贝`MindSpeed`目录下的`tests_extend`文件夹到`Megatron`目录中，并进入`Megatron`目录。

2. 修改`tests_extend/system_tests/feature_tests/ulysses.sh`文件中`TOKENIZER_MODEL`和`DATA_PATH`为本地路径。

3. 执行如下命令：

    ```shell
    bash tests_extend/system_tests/feature_tests/ulysses.sh
    ```

#### 使用效果

利用多个计算设备对输入序列进行并行切分，降低单设备的内存消耗，相比不开启序列并行单步耗时增加，相比重计算计算效率提升。

### Ascend Ring Attention长序列并行

#### 算法思路

Ring Attention借鉴了分块Softmax原理，在不需要获取整个序列的完整矩阵情况下进行分块attention计算。因此作者提出以分块方式执行自注意力和前馈网络计算，跨多个设备分布序列维度。具体地，该方法在进程之间构建注意力计算块的环状通信结构（Ring），每个进程具有一个切分后的本地QKV块。在计算完本地的attention后，通过向后发送和向前获取KV块，遍历进程设备环，以逐块的方式进行注意力和前馈网络计算。同时，本地的attention计算和KV块的通信理想情况下可以互相掩盖，从而消除了额外引入的通信开销。另外该方案在计算attention的过程中全程不需要数据拼接，支持的序列长度理论上可以无限拓展。

#### 使用场景

当使用GPT类模型进行训练，同时数据进MoE层时，实际序列长度8k以上。

不同于Ulysses方案，该方案不需要确保head_size被cp_size整除。

可兼容FlashAttention，目前已默认开启FlashAttention。

如果想要使得计算和通信可以互相掩盖，理论上需要确保每个计算块分到的序列长度$c \geq F/B$。其中F是每个device的FLOPS，B是每个device间的带宽。具体推导过程参见原文。在实践中，需要确保每个计算块分到的序列长度足够大，才能较好掩盖。

#### 使用方法

| 重要参数 | 参数说明 | 是否可选 | 取值范围 |
| --- | --- | --- | --- |
| --context-parallel-size [int] | 开启CP对应的数量，根据用户需求配置。 | 是 | 默认为1 |
| --seq-length [int] | 输入序列的长度。 | 否 | - |
| --use-cp-send-recv-overlap | 建议开启，开启后支持send receive overlap功能。 | 是 | 默认为True |
| --attention-mask-type | 设置Mask计算类型。 | 是 | 默认是causal（倒三角）Mask计算，设置general代表全量计算 |
| --context-parallel-algo | 长序列并行算法选项，当设置为`megatron_cp_algo`时开启Ring Attention。 | 是 | 默认值为ulysses_cp_algo，其他取值可为megatron_cp_algo，hybrid_cp_algo，adaptive_cp_algo，hybrid_adaptive_cp_algo |
| --megatron-cp-in-bnsd | 开启后，FA使用BNSD计算。 | 是 | 默认为True |
| --cp-window-size [int] | 使用原始的Ring Attention算法；当设置为大于`1`时，即使用Double Ring Attention算法，优化原始Ring Attention性能，--cp-window-size即为算法中双层Ring Attention的内层窗口大小，需要确保cp_size能被该参数整除。 | 是 | 默认为1 |

具体使用方式参考如下示例：

1. 拷贝`MindSpeed`目录下的`tests_extend`文件夹到`Megatron`目录中，并进入`Megatron`目录
2. 修改`tests_extend/system_tests/feature_tests/ring_attention.sh`文件中`TOKENIZER_MODEL`和`DATA_PATH`为本地路径，并设置`--cp-window-size`为1
3. 执行如下命令：

```shell
bash tests_extend/system_tests/feature_tests/ring_attention.sh
```

#### 使用效果

利用多个计算设备对输入序列进行并行切分，降低单设备的内存消耗，相比不开启序列并行单步耗时增加，相比重计算计算效率提升。

#### 注意事项

+ 开启Context Parallel时需要同时开启FlashAttention特性，否则特性不支持。
+ 在使用GPT类模型进行训练的场景下，建议`attention-mask-type`设置为`causal`。
+ 在8k的序列长度情况下，由于计算的时间缩短，cp功能分割之后的send receive的时间反而会长于计算时间，造成性能的下降，所以建议配置 seq-length / context-parallel-size > 8k 以获取最佳效果。具体公式参考：S/(Talpha) >= 1/(Wbeta)，其中，S=seq-length / context-parallel-size， T表示芯片的理论算力，alpha表示计算效率，W表示理论通信带宽，beta表示带宽利用率。
+ 内层窗口`--cp-window-size`增大时，通信与计算并发程度更高，但是计算、通信并发时可能由于片上内存带宽抢占，整体效率下降，需要结合实际场景进行调试，例如LLaMA2裁剪模型32k序列长度，cp为16且无其他并行切分时，实测内层窗口大小为2时性能最优。

### Ascend Double Ring Attention长序列并行

#### 算法思路

原有的Ring Attention借鉴了分块Softmax原理，在不需要获取整个序列的完整矩阵情况下进行分块attention计算。 以分块方式执行自注意力和前馈网络计算，跨多个设备分布序列维度。具体地，该方法在进程之间构建注意力计算块的环状通信结构（Ring），每个进程具有一个切分后的本地QKV块。在计算完本地的attention后，通过向后发送和向前获取KV块，遍历进程设备环，以逐块的方式进行注意力和前馈网络计算。同时，本地的attention计算和KV块的通信理想情况下可以互相掩盖，从而消除了额外引入的通信开销。另外该方案在计算attention的过程中全程不需要数据拼接，支持的序列长度理论上可以无限拓展。 在此基础上Double Ring Attention算法采用分布式注意力机制，通过双环结构（Double-Ring-Attention）来优化计算和内存使用。

#### 使用场景

Ring Attention的训练场景开启后，使用方式可参考[Ring Attention长序列并行](../features/ring-attention-context-parallel.md)。

#### 使用方法

开启Ring Attention的训练场景中，将`--cp-window-size`设置为大于1的整数，即可启用Double Ring Attention算法，优化原始Ring Attention性能。`--cp-window-size [int]`默认为`1`，即使用原始的Ring Attention算法；将其设置为大于1的整数，即可启用Double Ring Attention算法，该参数为Double Ring Attention算法中双层Ring Attention的内层窗口大小。

具体使用方式参考如下示例：

1. 拷贝`MindSpeed`目录下的`tests_extend`文件夹到`Megatron`目录中，并进入`Megatron`目录。
2. 修改`tests_extend/system_tests/feature_tests/ring_attention.sh`文件中`TOKENIZER_MODEL`和`DATA_PATH`为本地路径， 并设置`--cp-window-size`为2。
3. 执行如下命令：

```shell
bash tests_extend/system_tests/feature_tests/ring_attention.sh
```

#### 使用效果

利用多个计算设备对输入序列进行并行切分，通过双环结构（Double-Ring-Attention）提升计算效率。

#### 注意事项

+ 需要确保`--context-parallel-size`能被`--cp-window-size`整除。
+ 内层窗口`--cp-window-size`增大时，通信与计算并发程度更高，但是计算、通信并发时可能由于片上内存带宽抢占，整体效率下降，需要结合实际场景进行调试，例如LLaMA2裁剪模型32k序列长度，cp为16且无其他并行切分时，实测内层窗口大小为2时性能最优。

### Ascend 混合长序列并行

目前流行的序列并行方案，Ulysses和Ring Attention存在各自的局限性。

Ulysses需要确保attention head数可以被序列并行维度整除，在GQA、MQA场景下序列并行的大小有限制，导致序列长度的扩展有限。

Ring Attention的并行维度不受attention head数限制，因此理论上序列长度可以无限拓展。但相比于Ulysses，Ring Attention不能充分利用通信和计算带宽，在序列块大小较低时性能劣于Ulysses。

#### 算法思路

对Ulysses和Ring Attention做融合，实现混合序列并行，以此解决两个方案各自缺陷。

#### 使用场景

可兼容FlashAttention，目前已默认开启FlashAttention。

序列并行维度被分为Ulysses维度和Ring Attention维度，Ulysses维度和Ring Attention维度乘积即为序列并行维度。

#### 使用方法

设置`--context-parallel-size`，默认为1，根据用户需求配置。

设置`--context-parallel-algo hybrid_cp_algo`，以启用混合序列并行。

设置`--ulysses-degree-in-cp`，需要确保`--context-parallel-size`可以被该参数整除且大于1。例如当设置`--context-parallel-size=8`时，可以设置`--ulysses-degree-in-cp=2`或`--ulysses-degree-in-cp=4`。

同时需要确保`--num-attention-heads`可以被`--ulysses-degree-in-cp`与`--tensor-model-parallel-size`的乘积整除。

混合长序列并行支持Ring Attention长序列并行相关特性，包括send receive overlap功能、Mask计算类型配置。

具体使用方式参考如下示例：

1. 拷贝`MindSpeed`目录下的`tests_extend`文件夹到`Megatron`目录中，并进入`Megatron`目录。
2. 修改`tests_extend/system_tests/feature_tests/hybrid.sh`文件中`TOKENIZER_MODEL`和`DATA_PATH`为本地路径。
3. 执行如下命令：

    ```shell
    bash tests_extend/system_tests/feature_tests/hybrid.sh
    ```

#### 使用效果

利用多个计算设备对输入序列进行并行切分，降低单设备的内存消耗。相比不开启序列并行单步耗时增加；相比重计算，计算效率提升。

## 性能调优实践

### 算法选择指南

根据不同场景选择合适的序列并行算法：

| 场景条件 | 推荐算法 | 原因 |
| --- | --- | --- |
| head数能被cp_size整除 | Ulysses | 通信效率高 |
| 序列长度8K以上 | Ring Attention | 无head数限制 |
| 需要进一步优化Ring Attention性能 | Double Ring Attention | 双环结构提升效率 |
| 需要兼顾Ulysses和Ring Attention优势 | 混合序列并行 | 融合两种算法优点 |

### 常见优化策略

| 瓶颈类型 | 优化策略 | 适用场景 |
| --- | --- | --- |
| 计算瓶颈 | 开启FlashAttention、使用FP8混合精度 | 计算密集型场景 |
| 通信瓶颈 | 调整并行策略、开启通信计算重叠 | 多卡/多节点训练 |
| 内存瓶颈 | 使用序列并行、激活值卸载 | 长序列训练、大模型训练 |
| 数据加载瓶颈 | 使用异步数据加载、预取机制 | I/O密集型场景 |

### 最佳实践建议

1. **先诊断后优化**：在进行任何优化之前，先通过profiling工具定位性能瓶颈。
2. **从简单开始**：先尝试调整并行策略，再考虑复杂的优化方案。
3. **逐步验证**：每次只调整一个参数，验证效果后再进行下一步。
4. **关注整体效率**：不要只关注单步耗时，要关注整体训练吞吐量。
5. **结合硬件特性**：根据昇腾芯片的特性选择合适的优化策略。
