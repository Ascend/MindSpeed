# Performance Optimization Based on Megatron Parallelism Strategies

## Overview

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) is a distributed training acceleration library proposed by NVIDIA, supporting features such as data parallelism, model parallelism, and sequence parallelism, and is widely used in large model training. After compatibility adaptation for the MindSpeed Ascend platform, it now supports native parallelism strategies on the Ascend platform.
Although many parallelism strategies have been adapted, models still face high spatial and temporal complexity in long-text scenarios. Starting from the sequence dimension, MindSpeed has implemented numerous sequence parallelism methods to solve the problem of sequence dimension scaling.
This document introduces from the perspective of sequence parallelism, guiding users through using MindSpeed for Megatron performance optimization, with focuses on the following four sequence parallelism algorithms and their usage methods:

- [Ulysses Long Sequence Parallelism](#ulysses-long-sequence-parallelism)
- [Ring Attention Long Sequence Parallelism](#ring-attention-long-sequence-parallelism)
- [Double Ring Attention Long Sequence Parallelism](#double-ring-attention-long-sequence-parallelism)
- [Hybrid Long Sequence Parallelism](#hybrid-long-sequence-parallelism)

## Ulysses Long Sequence Parallelism

### Algorithm Overview

Ulysses partitions individual samples along the sequence dimension across the participating compute devices. Then, before the attention computation, it performs an all-to-all communication operation on the partitioned queries (Q), keys (K), and values (V) so that each compute device receives the complete sequence, but only for a non-overlapping subset of the attention heads. This allows the participating compute devices to compute different attention heads in parallel. Finally, Ulysses uses another all-to-all to gather the results across the attention heads while re-partitioning along the sequence dimension.

### Application Scenario

`num_head` must be divisible by `tp_size`*`cp_size`.

### Usage

Set `--context-parallel-size`, which defaults to 1, and configure it according to your requirements.
Also set `--context-parallel-algo ulysses_cp_algo`.

#### Example Script

1. Copy the `tests_extend` folder from the `MindSpeed` directory to the `Megatron` directory, and enter the `Megatron` directory.
2. Modify the `TOKENIZER_MODEL` and `DATA_PATH` in the `tests_extend/system_tests/feature_tests/ulysses.sh` file to local paths.
3. Execute the following command:

```shell
 bash tests_extend/system_tests/feature_tests/ulysses.sh
```

### Application Effect

By using multiple compute devices to split the input sequence in parallel, the memory consumption of a single device is reduced. Compared to not enabling Sequence Parallelism, the single-step latency increases, but the computational efficiency is improved compared to recomputation.

## Ring Attention Long Sequence Parallelism

### Algorithm Overview

Ring Attention draws on the principle of blockwise Softmax, performing blockwise attention computation without requiring the complete matrix of the entire sequence. The authors therefore propose performing self-attention and feedforward network computations in a blockwise manner, distributing the sequence dimension across multiple devices. Specifically, this method constructs a ring communication structure (Ring) for attention computation blocks among processes, where each process holds a partitioned local QKV block. After computing the local attention, it traverses the process device ring by sending KV blocks backward and fetching KV blocks forward, performing attention and feedforward network computations block by block. Meanwhile, local attention computation and KV block communication can ideally overlap with each other, thereby eliminating the additionally introduced communication overhead. Furthermore, this approach requires no data concatenation throughout the attention computation process, and the supported sequence length can theoretically be extended indefinitely.

### Application Scenario

When training GPT-type models and the actual sequence length exceeds 8K when data enters the MoE layer.

Unlike the Ulysses approach, this approach does not require `head_size` to be divisible by `cp_size`.

It is compatible with `FlashAttention`, which is now enabled by default.

To achieve ideal overlap between computation and communication, it is theoretically necessary to ensure that the sequence length assigned to each computation block satisfies $c \geq F/B$, where F is the FLOPS of each device and B is the bandwidth between devices. For the detailed derivation, refer to the original paper. In practice, the sequence length assigned to each computation block must be sufficiently large to achieve effective overlap.

### Usage

| Important Parameter | Description |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --context-parallel-size [int] | The number of CP processes to enable. The default value is 1. Configure this parameter based on your requirements. |
| --seq-length [int] | The length of the input sequence. |
| --use-cp-send-recv-overlap | It is recommended to enable this parameter. When enabled, the send/receive overlap function is activated. |
| --attention-mask-type [general/causal] | Optional. Sets the mask computation type. The default value is `causal` (lower triangular mask). Setting it to `general` enables full computation. |
| --context-parallel-algo megatron_cp_algo | The long sequence parallelism algorithm option. The default value is `ulysses_cp_algo`. Setting it to `megatron_cp_algo` enables Ring Attention. |
| --megatron-cp-in-bnsd | When enabled, FA uses BNSD computation. |
| --cp-window-size [int] | Optional. The default value is `1`, which uses the original Ring Attention algorithm. Setting it to a value greater than `1` enables the Double Ring Attention algorithm, which optimizes the performance of the original Ring Attention. `--cp-window-size` specifies the inner window size of the two-layer Ring Attention in the algorithm. Ensure that cp_size is divisible by this parameter. |

#### Example Script

1. Copy the `tests_extend` folder from the `MindSpeed` directory to the `Megatron` directory, and enter the `Megatron` directory.

2. Modify `TOKENIZER_MODEL` and `DATA_PATH` in the `tests_extend/system_tests/feature_tests/ring_attention.sh` file to local paths, and set `cp-window-size` to 1.
3. Execute the following command:

```shell
bash tests_extend/system_tests/feature_tests/ring_attention.sh
```

### Application Effect

Multiple compute devices are used to partition the input sequence in parallel, reducing memory consumption on a single device. Compared to not enabling sequence parallelism, the single-step latency increases, but the computational efficiency is improved compared to recomputation.

### Precautions

- When enabling Context Parallel, Flash Attention must also be enabled; otherwise, the feature is not supported.
- When training GPT-like models, it is recommended to set `attention-mask-type` to `causal`.
- For a sequence length of 8K, the reduced computation time means that the send/receive time after CP partitioning may actually exceed the computation time, leading to performance degradation. Therefore, it is recommended to configure seq-length / context-parallel-size > 8K for optimal results. Refer to the formula: S/(Talpha) >= 1/(Wbeta), where S = seq-length / context-parallel-size, T represents the theoretical compute power of the chip, alpha represents the computational efficiency, W represents the theoretical communication bandwidth, and beta represents the bandwidth utilization.
- When the inner window `--cp-window-size` increases, the degree of concurrency between communication and computation becomes higher. However, during concurrent computation and communication, overall efficiency may degrade due to on-chip memory bandwidth contention. Debugging should be performed based on the actual scenario. For example, with a pruned llama2 model at a 32k sequence length, cp set to 16, and no other parallel partitioning, measured performance is optimal when the inner window size is 2.

## Double Ring Attention Long Sequence Parallelism

### Algorithm Overview

The original Ring Attention leverages the principle of blockwise Softmax to perform blockwise attention computation without requiring the complete matrix of the entire sequence. It performs self-attention and feedforward network computations in a blockwise manner, distributing the sequence dimension across multiple devices. Specifically, this method constructs a ring communication structure (Ring) for attention computation blocks among processes, where each process holds a partitioned local QKV block. After computing the local attention, it traverses the process device ring by sending KV blocks backward and fetching KV blocks forward, performing attention and feedforward network computations in a blockwise manner. Ideally, the local attention computation and KV block communication can overlap and hide each other, thereby eliminating the additionally introduced communication overhead. Furthermore, this approach requires no data concatenation throughout the entire attention computation process, and the supported sequence length can theoretically be extended infinitely. Building on this, the Double Ring Attention algorithm adopts a distributed attention mechanism, optimizing computation and memory usage through a double-ring structure (Double-Ring-Attention).

### Application Scenario

Training scenarios with Ring Attention enabled

For the method to enable Ring Attention, see [Ring Attention Long Sequence Parallelism](../features/ring-attention-context-parallel.md)

### Usage

In training scenarios with Ring Attention enabled, set `--cp-window-size` to an integer greater than 1 to enable the Double Ring Attention algorithm and optimize the performance of the original Ring Attention.

| Important Parameter                   | Description                                                                                                                                        |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| --cp-window-size [int] | The default value is `1`, which uses the original Ring Attention algorithm. Setting `--cp-window-size` to an integer greater than 1 enables the Double Ring Attention algorithm. This parameter specifies the inner window size of the double-ring attention in the Double Ring Attention algorithm. |

#### Example Script

1. Copy the `tests_extend` folder from the `MindSpeed` directory to the `Megatron` directory, and enter the `Megatron` directory.
2. Modify `TOKENIZER_MODEL` and `DATA_PATH` in the `tests_extend/system_tests/feature_tests/ring_attention.sh` file to local paths, and set `cp-window-size` to 2.
3. Execute the following command:

```shell
bash tests_extend/system_tests/feature_tests/ring_attention.sh
```

### Application Effect

Utilizes multiple compute devices to partition the input sequence in parallel, improving computational efficiency through a double-ring structure (Double-Ring-Attention).

### Precautions

+ Ensure that `--context-parallel-size` is divisible by `--cp-window-size`.
+ When the inner window `--cp-window-size` increases, the degree of concurrency between communication and computation becomes higher. However, during concurrent computation and communication, overall efficiency may decline due to on-chip memory bandwidth contention. Debugging should be performed based on the actual scenario. For example, for a pruned llama2 model with a 32K sequence length, when cp is 16 and there is no other parallel partitioning, actual measurements show that the performance is optimal when the inner window size is 2.

## Hybrid Long Sequence Parallelism

Currently, the popular sequence parallelism schemes, Ulysses and Ring Attention, each have their own limitations.

Ulysses requires that the number of attention heads be divisible by the sequence parallelism dimension. In GQA and MQA scenarios, the size of sequence parallelism is restricted, which limits the scalability of the sequence length.

The parallelism dimension of Ring Attention is not constrained by the number of attention heads, so theoretically, the sequence length can be extended infinitely. However, compared to Ulysses, Ring Attention cannot fully utilize communication and computation bandwidth, and its performance is inferior to Ulysses when the sequence chunk size is small.

### Algorithm Overview

Integrate Ulysses and Ring Attention to implement hybrid sequence parallelism, thereby addressing the shortcomings of each individual scheme.

### Application Scenario

It is compatible with FlashAttention, which is enabled by default.

The Sequence Parallelism dimension is divided into a Ulysses dimension and a ring attention dimension. The product of the Ulysses dimension and the ring attention dimension is the Sequence Parallelism dimension.

### Usage

Set `--context-parallel-size`. The default value is 1. Configure it based on your requirements.

Set `--context-parallel-algo hybrid_cp_algo` to enable hybrid sequence parallelism.

Set `--ulysses-degree-in-cp`. Ensure that `--context-parallel-size` is divisible by this parameter and greater than 1. For example, when `--context-parallel-size=8` is set, `--ulysses-degree-in-cp` can be set to 2 or 4.

Also ensure that `--ulysses-degree-in-cp` is divisible by the number of attention heads.

Hybrid Long Sequence parallelism supports features related to Ring Attention long sequence parallelism, including the send-receive overlap function and mask computation type configuration.

#### Example Script

1. Copy the `tests_extend` folder from the `MindSpeed` directory to the `Megatron` directory, and enter the `Megatron` directory.
2. Modify the `TOKENIZER_MODEL` and `DATA_PATH` in the `tests_extend/system_tests/feature_tests/hybrid.sh` file to local paths.
3. Execute the following command:

```shell
bash tests_extend/system_tests/feature_tests/hybrid.sh
```

### Application Effect

By partitioning the input sequence across multiple compute devices in parallel, the memory consumption of a single device is reduced. Compared to not enabling Sequence Parallelism, the single-step latency increases, but the computational efficiency is improved compared to recomputation.
