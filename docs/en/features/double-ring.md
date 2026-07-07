# Double Ring Attention Long-Sequence Parallelism

## Background and Challenges

Training large language models on long sequences requires substantial memory and computational resources. The industry has proposed sequence parallelism methods based on head parallelism and context parallelism. In the attention block, the head parallelism approach retains the entire sequence and computes attention for different heads in parallel, while the context parallelism approach splits the QKV tensors into chunks along the sequence dimension. However, both approaches face limitations when applied to large-scale LLMs with extremely long sequences. First, head parallelism encounters scalability issues. In head parallelism, the degree of sequence parallelism inherently cannot exceed the number of attention heads. Therefore, there is an upper limit to the extent to which head parallelism can scale. Second, context parallelism encounters communication efficiency issues. Context parallelism employs point-to-point (P2P) communication primitives. However, P2P suffers from low intra-node bandwidth utilization and low inter-node network resource utilization. This bottleneck makes it difficult to overlap communication with computation when scaling the context parallelism dimension.

## Solution

Supports the Double Ring Attention algorithm, further accelerating the original Ring Attention implementation. For algorithm details, refer to the original paper:
> LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism (<https://arxiv.org/pdf/2406.18485>)

### Approach

The original Ring Attention leverages the blockwise Softmax principle to perform blockwise attention computation without requiring the full matrix of the entire sequence. It performs self-attention and feedforward network computations in a blockwise manner, distributing the sequence dimension across multiple devices. Specifically, this method constructs a ring communication structure (Ring) for attention computation blocks among processes, where each process holds a sharded local QKV block. After computing the local attention, it traverses the process device ring by sending KV blocks backward and fetching KV blocks forward, performing attention and feedforward network computations block by block. Meanwhile, local attention computation and KV block communication can ideally overlap with each other, thereby eliminating the additionally introduced communication overhead. Furthermore, this approach requires no data concatenation throughout the entire attention computation process, and the supported sequence length can theoretically be extended infinitely. Building on this, the Double Ring Attention algorithm adopts a distributed attention mechanism, optimizing computation and memory usage through a dual-ring structure (Double Ring Attention).

## Application Scenario

Training scenarios where Ring Attention is already enabled

For how to enable Ring Attention, refer to [the document](ring-attention-context-parallel.md)

## Usage

In training scenarios where Ring Attention is enabled, set `--cp-window-size` to an integer greater than 1 to enable the Double Ring Attention algorithm and optimize the performance of the original Ring Attention.

| Key Parameters                   | Description                                                                                                                                        |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| --cp-window-size [int] | Defaults to `1`, which uses the original Ring Attention algorithm. Set `--cp-window-size` to an integer greater than 1 to enable the Double Ring Attention algorithm. This parameter specifies the inner ring window size of the two-level Ring Attention in the Double Ring Attention algorithm. |

## Application Effects

The input sequence is partitioned in parallel across multiple compute devices, and computational efficiency is improved through the dual-ring structure (Double Ring Attention).

## Notes

1. Ensure that `--context-parallel-size` is divisible by `--cp-window-size`.
2. Ensure that `--cp-window-size` is less than `--context-parallel-size`.
3. When the inner window `--cp-window-size` increases, the degree of concurrency between communication and computation becomes higher. However, due to potential on-chip memory bandwidth contention during concurrent computation and communication, overall efficiency may decrease. Debugging should be performed based on the actual scenario. For example, for a pruned Llama2 model with a 32k sequence length, when cp is 16 and no other parallel partitioning is used, actual measurements show optimal performance with an inner window size of 2.
