# Ring Attention Long-Sequence Parallelism

## Background and Challenges

From generative AI to scientific research models, long-sequence training is becoming increasingly important. In the generative AI domain, tasks such as conversational AI, long-document summarization, and video generation all require reasoning over long contexts across spatial and temporal dimensions. Similarly, chapter-level and book-level summarization (tens or even hundreds of thousands of tokens) is also critical in conversational AI and summarization tasks. Existing parallelism methods such as data, tensor, and pipeline parallelism cannot partition along the sequence dimension. As the sequence length (S) increases, the training memory overhead grows at a rate of $O$($S^2$). Therefore, specific optimizations are needed for long-sequence scenarios to address the training requirements of long-context training.

## Solution

The Ring Attention long-sequence parallelism scheme is supported to address the sequence dimension scaling problem. For details, see the original paper:
> Ring Attention with Blockwise Transformers for Near-Infinite Context (<https://arxiv.org/pdf/2310.01889>)

Supports the Double Ring Attention algorithm, further accelerating the original Ring Attention implementation. For algorithm details, see the original paper:
> LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism (<https://arxiv.org/pdf/2406.18485>)

### Approach

Ring Attention draws on the principle of blockwise Softmax, performing blockwise attention computation without requiring the complete matrix of the entire sequence. Therefore, the authors propose performing self-attention and feedforward network computations in a blockwise manner, distributing the sequence dimension across multiple devices. Specifically, this method constructs a ring communication structure for attention computation blocks among processes, where each process holds a sharded local QKV block. After computing the local attention, it traverses the process device ring by sending KV blocks backward and fetching KV blocks forward, performing attention and feedforward network computations in a block-by-block manner. Meanwhile, the local attention computation and KV block communication can ideally overlap with each other, thereby eliminating the additionally introduced communication overhead. In addition, this approach requires no data concatenation throughout the attention computation process, and the supported sequence length can theoretically be extended infinitely.

## Application Scenario

When training GPT-like models and the actual sequence length exceeds 8K when data enters the MoE layer.

Unlike the Ulysses scheme, this scheme does not require `head_size` to be divisible by `cp_size`.

It is compatible with FlashAttention, which is now enabled by default.

To achieve overlap between computation and communication, it is theoretically necessary to ensure that the sequence length assigned to each computation block satisfies $c \geq F/B$, where F is the FLOPS of each device and B is the bandwidth between devices. For the detailed derivation, refer to the original paper. In practice, the sequence length assigned to each computation block must be sufficiently large to achieve effective overlap.

## Usage

| Name                                     | Description                                                                                                                                                         |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --context-parallel-size [int]            | The number of CP instances to enable. The default value is 1. Configure this parameter based on user requirements.                                                                                                                                     |
| --seq-length [int]                       | The length of the input sequence.                                                                                                                                                     |
| --use-cp-send-recv-overlap               | It is recommended to enable this option. When enabled, it activates the send/receive overlap feature.                                                                                                                            |
| --attention-mask-type [general/causal]   | Optional. Sets the mask computation type. The default value is `causal` (lower triangular) mask computation. Setting it to `general` indicates full computation.                                                                                                          |
| --context-parallel-algo <b>megatron_cp_algo</b> | Long sequence parallelism algorithm option. The default value is `megatron_cp_algo`, which enables Ring Attention.                                                                                     |
| --megatron-cp-in-bnsd                    | When enabled, FA uses BNSD computation.                                                          |
| --cp-window-size [int]                   | Optional. The default value is `1`, which uses the original Ring Attention algorithm. When set to a value greater than `1`, the Double Ring Attention algorithm is used to optimize the performance of the original Ring Attention. `--cp-window-size` specifies the inner window size of the two-layer Ring Attention in the algorithm. It must be ensured that cp_size is divisible by this parameter.|

## Application Effects

By partitioning the input sequence across multiple compute devices, memory consumption per device is reduced. Compared to not enabling sequence parallelism, the per-step time increases, but computational efficiency is improved relative to recomputation.

## Notes

1. When enabling Context Parallel, the Flash Attention feature must also be enabled; otherwise, the feature is not supported.
2. When training GPT-like models, it is recommended to set `attention-mask-type` to `causal`.
3. When the sequence length is 8k, the computation time is reduced, and the send/receive time after cp function partitioning may become longer than the computation time, causing performance degradation. Therefore, it is recommended to configure seq-length / context-parallel-size > 8k for optimal results. For the specific formula, refer to: S/(Talpha) >= 1/(Wbeta), where S = seq-length / context-parallel-size, T represents the theoretical computing power of the chip, alpha represents the computational efficiency, W represents the theoretical communication bandwidth, and beta represents the bandwidth utilization.
4. When the inner window `--cp-window-size` is increased, the degree of communication and computation concurrency becomes higher. However, during concurrent computation and communication, the overall efficiency may decrease due to on-chip memory bandwidth contention. This needs to be tuned based on the actual scenario. For example, with a Llama2 pruned model at 32k sequence length, cp set to 16, and no other parallel partitioning, actual measurements show that the optimal performance is achieved when the inner window size is 2.
