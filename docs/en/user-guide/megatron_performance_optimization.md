# Performance Optimization Based on Megatron Parallelism Strategies

## Overview

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) is a distributed training acceleration library proposed by NVIDIA. It supports features such as data parallelism, model parallelism, and sequence parallelism, and is widely used in large model training. After compatibility adaptation on the MindSpeed Ascend platform, it now supports native parallelism strategies on the Ascend platform.

In long-text scenarios, model training faces high space and time complexity. Starting from the sequence dimension, MindSpeed implements multiple sequence parallelism methods to solve the sequence dimension scaling problem. This manual provides comprehensive guidance for users to perform Megatron performance optimization with MindSpeed, from performance diagnosis to optimization practice.

## Performance Diagnosis Methodology

### Performance Metric Definitions

The first step in performance optimization is to understand the performance metrics. For a batch, the time is mainly composed of the following parts:

```text
Total time in a single batch = Data loading time + Model forward and backward time + Optimizer time + Model postprocessing time + Communication time + Scheduling time
```

Each component is described as follows:

- **Data loading time**: The time for the model to load the data it needs (such as images, videos, and text), including the time to read data from hardware storage devices to CPU, the preprocessing of data on CPU (encoding/decoding and other operations), and the time to place CPU data onto the device. For models that need to be split across multiple cards, data loading also includes the time to broadcast from the data-loading card to other cards.
- **Model forward and backward time**: The time of the forward and backward processes of a deep learning model, namely the Forward and Backward processes, including the time for forward data computation and backward data differentiation.
- **Optimizer time**: The time for updating model parameters.
- **Model post-processing time**: The time after the optimizer update, including data post-processing or some necessary synchronization operations, which usually depends on model-specific operations.
- **Communication time**: The communication time between cards within a single node and between nodes in a multi-node setup. Due to the special mechanism of PyTorch, when communication and computation can run in parallel, this represents the communication time that is not masked by computation.
- **Scheduling time**: The time required for the model to go from CPU instructions to invoking the kernels on the NPU side.

### Tuning Process

Performance tuning generally follows the five-step process below:

```text
Collect profile data → Analyze operator time consumption → Analyze communication time → Analyze memory usage → Select an optimization strategy
```

1. **Collect profile data**: Run the training script and enable profiling.
2. **Analyze operator time consumption**: Identify the most time-consuming operators and locate compute bottlenecks.
3. **Analyze communication time**: Check the communication time proportion to determine whether a communication bottleneck exists.
4. **Analyze memory usage**: Check the memory usage to determine whether a memory bottleneck exists.
5. **Select an optimization strategy**: Choose an appropriate optimization solution based on the bottleneck type.

### Performance Data Collection

Collecting performance data is a key step in analyzing performance issues and identifying performance bottlenecks. MindSpeed supports profile data collection based on Ascend chips.

The commonly used parameters for enabling profiling are as follows:

```shell
# Enable profiling.
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

**Parameters**

| Parameter | Description | Default Value |
| --- | --- | --- |
| `--profile` | Whether to enable profiling | False |
| `--profile-step-start` | The step number at which profiling starts (inclusive) | 0 |
| `--profile-step-end` | The step number at which profiling ends (exclusive). Set to `-1` to collect until training ends | -1 |
| `--profile-ranks` | Specifies the ranks to collect. Set to `-1` to collect profile data from all ranks | [0] |
| `--profile-level` | Profiling level: `level0` (operator time only), `level1` (operator + communication time), `level2` (complete data) | level0 |
| `--profile-with-cpu` | Whether to collect CPU data | False |
| `--profile-record-shapes` | Whether to collect computation shapes (used to analyze memory and computation volume) | False |
| `--profile-save-path` | Path for saving collected data | ./profile_dir |

### Performance Analysis Process

After collecting profile data, you can use MindStudio Insight <!-- ](https://www.hiascend.com/document/detail/en/mindstudio/2610/GUI_baseddevelopmenttool/MindStudioInsight/docs/en/user_guide/overview.md) --> to perform visual analysis on the performance data and locate performance bottlenecks.

#### Analysis Dimension

MindStudio Insight supports multi-dimensional performance analysis:

| Analysis Dimension | Analysis Content | Positioning Target |
| --- | --- | --- |
| Operator time consumption | Identify operators with long time consumption | Compute bottleneck |
| Communication time consumption | Analyze the time proportion of communication and computation | Communication bottleneck |
| Memory  | View memory usage | Memory bottleneck |
| Pipeline | Analyze the pipeline bubble ratio of pipeline parallelism | Parallel efficiency |

#### Analysis Process

1. **Data import**: Import the collected profile data into MindStudio Insight.
2. **Visualization analysis**: View the operator time consumption distribution chart, communication time proportion, and so on.
3. **Bottleneck identification**: Locate the performance bottleneck based on the analysis results.
4. **Optimization suggestions**: Select an appropriate optimization strategy based on the bottleneck type.

### Bottleneck Type Determination

Based on the analysis results, performance bottlenecks can be classified into the following categories:

| Bottleneck Type | Determination Basis | Typical Manifestation |
| --- | --- | --- |
| **Compute bottleneck** | High proportion of operator execution time | Slow single-card training speed, low GPU/NPU utilization |
| **Communication bottleneck** | High proportion of communication time | Unsatisfactory multi-card training speedup ratio |
| **Memory bottleneck** | Memory usage approaching the upper limit | OOM errors occur during training |
| **Data loading bottleneck** | High proportion of data loading time | GPU/NPU idle waiting for data during training |

## Sequence Parallelism Optimization

### Ascend Ulysses

#### Algorithm Approach

Ulysses partitions each sample along the sequence dimension across the participating compute devices. Then, before the attention computation, it performs an all-to-all communication operation on the partitioned query (Q), key (K), and value (V) so that each compute device receives the complete sequence, but only for a non-overlapping subset of attention heads. This allows the participating compute devices to compute different attention heads in parallel. Finally, Ulysses uses another all-to-all to gather the results across attention heads while re-partitioning along the sequence dimension.

#### Use Cases

`num_head` must be divisible by `tp_size*cp_size`. This is suitable for scenarios with a large number of heads that can be evenly divided by the parallel dimensions.

#### Usage

Set `--context-parallel-size`, which defaults to `1`, according to your requirements.
Also set `--context-parallel-algo ulysses_cp_algo`.

Refer to the following example for specific usage:

1. Copy the `tests_extend` folder under the `MindSpeed` directory to the `Megatron` directory, and enter the `Megatron` directory.

2. Modify `TOKENIZER_MODEL` and `DATA_PATH` in the `tests_extend/system_tests/feature_tests/ulysses.sh` file to local paths.

3. Run the following command:

    ```shell
    bash tests_extend/system_tests/feature_tests/ulysses.sh
    ```

#### Effect

By splitting the input sequence in parallel across multiple compute devices, the memory consumption of a single device is reduced. Compared with not enabling Sequence Parallelism, the per-step time increases, while the computational efficiency improves compared with recomputation.

### Ascend Ring Attention

#### Algorithm Approach

Ring Attention draws on the principle of block-wise Softmax, performing block-wise attention computation without requiring the complete matrix of the entire sequence. The authors therefore propose executing self-attention and feed-forward network computations in a block-wise manner, distributing the sequence dimension across multiple devices. Specifically, this method constructs a ring-shaped communication structure (Ring) for attention computation blocks among processes, where each process holds a partitioned local QKV block. After computing the local attention, each process traverses the ring of process devices by sending its KV block backward and fetching the KV block forward, performing attention and feed-forward network computations block by block. Meanwhile, the local attention computation and the communication of KV blocks can ideally overlap with each other, thereby eliminating the additionally introduced communication overhead. In addition, this scheme requires no data concatenation throughout the attention computation process, and the supported sequence length can theoretically be extended indefinitely.

#### Use Cases

This approach is used when training GPT-style models and the data enters the MoE layer, with an actual sequence length of 8k or more.

Unlike the Ulysses approach, this approach does not require `head_size` to be divisible by `cp_size`.

It is compatible with FlashAttention, which is currently enabled by default.

To allow computation and communication to overlap with each other, in theory it is necessary to ensure that the sequence length assigned to each computation block satisfies $c \geq F/B$, where F is the FLOPS of each device and B is the bandwidth between devices. For the detailed derivation, refer to the original paper. In practice, it is necessary to ensure that the sequence length assigned to each computation block is sufficiently large to achieve good overlap.

#### Usage

| Key Parameter | Description | Optional | Value Range |
| --- | --- | --- | --- |
| `--context-parallel-size` [int] | The number of CP to enable, configured according to user requirements. | Yes | Default: `1` |
| `--seq-length` [int] | The length of the input sequence. | No | - |
| `--use-cp-send-recv-overlap` | Recommended to enable. When enabled, the send receive overlap feature is supported. | Yes | Default: `True` |
| `--attention-mask-type` | Mask computation type. | Yes | Default: causal (lower triangular) Mask computation. Setting it to general represents full computation |
| `--context-parallel-algo` | The Long Sequence Parallelism algorithm option. When set to `megatron_cp_algo`, Ring Attention is enabled. | Yes | Default: `ulysses_cp_algo`. Other values can be `megatron_cp_algo`, `hybrid_cp_algo`, `adaptive_cp_algo`, or `hybrid_adaptive_cp_algo` |
| `--megatron-cp-in-bnsd` | When enabled, FA uses BNSD computation. | Yes | Default: True |
| `--cp-window-size` [int] | Uses the original Ring Attention algorithm. When set to a value greater than `1`, the Double Ring Attention algorithm is used to optimize the performance of the original Ring Attention. `--cp-window-size` is the inner window size of the double-layer Ring Attention in the algorithm, and it is necessary to ensure that `cp_size` is divisible by this parameter. | Yes | Default: `1` |

Refer to the following example for specific usage:

1. Copy the `tests_extend` folder under the `MindSpeed` directory to the `Megatron` directory, and enter the `Megatron` directory
2. Modify `TOKENIZER_MODEL` and `DATA_PATH` in the `tests_extend/system_tests/feature_tests/ring_attention.sh` file to local paths, and set `--cp-window-size` to 1
3. Run the following command:

```shell
bash tests_extend/system_tests/feature_tests/ring_attention.sh
```

#### Effect

By splitting the input sequence in parallel across multiple compute devices, the memory consumption of a single device is reduced. Compared with not enabling Sequence Parallelism, the per-step time increases, while the computational efficiency improves compared with recomputation.

#### Precautions

+ When enabling Context Parallel, the FlashAttention feature must be enabled at the same time; otherwise, the feature is not supported.
+ When training GPT-type models, it is recommended to set `attention-mask-type` to `causal`.
+ At a sequence length of 8k, because the computation time is shortened, the send/receive time after CP splitting may become longer than the computation time, causing performance degradation. Therefore, it is recommended to configure seq-length / context-parallel-size > 8k for optimal results. For the specific formula, refer to: S/(Talpha) >= 1/(Wbeta), where S = seq-length / context-parallel-size, T represents the theoretical computing power of the chip, alpha represents the computation efficiency, W represents the theoretical communication bandwidth, and beta represents the bandwidth utilization.
+ When the inner window `--cp-window-size` increases, the degree of concurrency between communication and computation becomes higher. However, when computation and communication run concurrently, the overall efficiency may decrease due to contention for on-chip memory bandwidth. It needs to be tuned according to the actual scenario. For example, for the LLaMA2 pruned model with a 32k sequence length, CP of 16, and no other parallel splitting, the measured performance is optimal when the inner window size is 2.

### Ascend Double Ring Attention

#### Algorithm Approach

The original Ring Attention draws on the block-wise Softmax principle to perform block-wise attention computation without requiring the complete matrix of the entire sequence. It performs self-attention and feed-forward network computation in a block-wise manner, distributing the sequence dimension across multiple devices. Specifically, this method builds a ring-shaped communication structure (Ring) of attention computation blocks among processes, where each process holds a partitioned local QKV block. After computing the local attention, it traverses the process device ring by sending backward and fetching forward KV blocks, performing attention and feed-forward network computation block by block. Meanwhile, the local attention computation and KV block communication can ideally overlap with each other, thereby eliminating the additionally introduced communication overhead. In addition, this scheme requires no data concatenation throughout the attention computation process, and the supported sequence length can theoretically be extended indefinitely. On this basis, the Double Ring Attention algorithm adopts a distributed attention mechanism and optimizes computation and memory usage through a double-ring structure (Double-Ring-Attention).

#### Use Cases

After Ring Attention training is enabled, refer to [Ring Attention](../features/ring-attention-context-parallel.md) for usage.

#### Usage

In a training scenario where Ring Attention is enabled, set `--cp-window-size` to an integer greater than 1 to enable the Double Ring Attention algorithm and optimize the performance of the original Ring Attention. `--cp-window-size [int]` defaults to `1`, which means the original Ring Attention algorithm is used. Setting it to an integer greater than 1 enables the Double Ring Attention algorithm. This parameter specifies the inner window size of the two-level Ring Attention in the Double Ring Attention algorithm.

Refer to the following example for specific usage:

1. Copy the `tests_extend` folder under the `MindSpeed` directory to the `Megatron` directory, and enter the `Megatron` directory.
2. Modify `TOKENIZER_MODEL` and `DATA_PATH` in the `tests_extend/system_tests/feature_tests/ring_attention.sh` file to local paths, and set `--cp-window-size` to `2`.
3. Run the following command:

```shell
bash tests_extend/system_tests/feature_tests/ring_attention.sh
```

#### Effect

The input sequence is split in parallel across multiple compute devices, and computational efficiency is improved through the double-ring structure (Double-Ring-Attention).

#### Precautions

+ Ensure that `--context-parallel-size` is divisible by `--cp-window-size`.
+ As the inner window `--cp-window-size` increases, the degree of concurrency between communication and computation becomes higher. However, when computation and communication run concurrently, on-chip memory bandwidth contention may reduce overall efficiency. Therefore, tuning is required based on the actual scenario. For example, for the LLaMA2 pruned model with a 32k sequence length, cp of 16, and no other parallel partitioning, the measured performance is optimal when the inner window size is 2.

### Ascend Hybrid Long Sequence Parallelism

The currently popular sequence parallelism schemes, Ulysses and Ring Attention, each have their own limitations.

Ulysses requires that the number of attention heads be divisible by the sequence parallelism dimension. In GQA and MQA scenarios, the size of sequence parallelism is restricted, which limits the extension of sequence length.

The parallelism dimension of Ring Attention is not restricted by the number of attention heads, so in theory the sequence length can be extended indefinitely. However, compared with Ulysses, Ring Attention cannot fully utilize communication and computation bandwidth, and its performance is inferior to Ulysses when the sequence block size is small.

#### Algorithm Approach

Fuse Ulysses and Ring Attention to implement hybrid sequence parallelism, thereby addressing the respective shortcomings of the two approaches.

#### Use Cases

It is compatible with FlashAttention, which is now enabled by default.

The sequence parallelism is divided into the Ulysses dimension and the Ring Attention dimension. The product of the Ulysses dimension and the Ring Attention dimension is the sequence parallelism dimension.

#### Usage

Set `--context-parallel-size`, which defaults to `1`, according to your requirements.

Set `--context-parallel-algo hybrid_cp_algo` to enable hybrid sequence parallelism.

Set `--ulysses-degree-in-cp`, and ensure that `--context-parallel-size` is divisible by this parameter and greater than 1. For example, when `--context-parallel-size=8` is set, `--ulysses-degree-in-cp=2` or `--ulysses-degree-in-cp=4` can be set.

At the same time, ensure that `--num-attention-heads` is divisible by the product of `--ulysses-degree-in-cp` and `--tensor-model-parallel-size`.

Hybrid Long Sequence Parallelism supports the Ring Attention Long Sequence Parallelism related features, including the send receive overlap function and Mask computation type configuration.

For specific usage, refer to the following example:

1. Copy the `tests_extend` folder under the `MindSpeed` directory to the `Megatron` directory, and enter the `Megatron` directory.
2. Modify `TOKENIZER_MODEL` and `DATA_PATH` in the `tests_extend/system_tests/feature_tests/hybrid.sh` file to local paths.
3. Execute the following command:

    ```shell
    bash tests_extend/system_tests/feature_tests/hybrid.sh
    ```

#### Effect

The input sequence is split in parallel across multiple compute devices, reducing the memory consumption of a single device. Compared with not enabling Sequence Parallelism, the per-step time increases; compared with recomputation, the computational efficiency improves.

## Performance Tuning Practice

### Algorithm Selection

Select the appropriate sequence parallelism algorithm based on different scenarios:

| Scenario Condition | Recommended Algorithm | Reason |
| --- | --- | --- |
| The number of heads is divisible by cp_size | Ulysses | High communication efficiency |
| Sequence length of 8K or above | Ring Attention | No restriction on the number of heads |
| Further optimization of Ring Attention performance is required | Double Ring Attention | The double-ring structure improves efficiency |
| The advantages of both Ulysses and Ring Attention are needed | Hybrid Sequence Parallelism | Combines the strengths of both algorithms |

### Common Optimization Strategies

| Bottleneck Type | Optimization Strategy | Applicable Scenario |
| --- | --- | --- |
| Compute Bottleneck | Enable FlashAttention and use FP8 mixed precision | Compute-intensive scenarios |
| Communication Bottleneck | Adjust the parallelism strategy and enable communication-computation overlap | Multi-card/multi-node training |
| Memory Bottleneck | Use Sequence Parallelism and activation offloading | Long-sequence training and large-model training |
| Data Loading Bottleneck | Use asynchronous data loading and prefetching mechanisms | I/O-intensive scenarios |

### Best Practice Recommendations

1. **Diagnose before optimizing**: Before making any optimization, first identify the performance bottleneck using profiling tools.
2. **Start simple**: Try adjusting the parallelism strategy first before considering complex optimization approaches.
3. **Verify incrementally**: Adjust only one parameter at a time, and verify the effect before proceeding to the next step.
4. **Focus on overall efficiency**: Do not focus solely on per-step time; pay attention to the overall training throughput.
5. **Combine hardware characteristics**: Select appropriate optimization strategies based on the characteristics of Ascend chips.
