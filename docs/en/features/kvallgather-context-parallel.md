# KVAllGather Long Sequence Parallelism

## Background and Challenges

From generative AI to scientific research models, long sequence training is becoming increasingly important. In the generative AI domain, tasks such as conversational AI, long document summarization, and video generation all require reasoning over long contexts in both spatial and temporal dimensions. Similarly, chapter- and book-level summarization (tens or even hundreds of thousands of words) is also critical in conversational AI and summarization tasks. Existing parallelism methods such as data, tensor, and pipeline parallelism cannot partition along the sequence dimension. As the sequence dimension (S) grows, the training memory overhead increases at a rate of $O$($S^2$). Therefore, specific optimizations for long sequence scenarios are needed to address the training requirements of long-sequence training.

## Solution

In the KVAllGather long sequence parallelism scheme, each compute device first holds different segments of the data sample along the sequence dimension. Then, before the attention computation begins, an all-gather communication operation is performed on the sharded keys and values, enabling each device to obtain the complete key and value sequences. Finally, each device uses its local query to perform attention computation with the complete keys and values, producing the corresponding output.

For more details on the KVAllGather long-sequence parallelism scheme, see the original paper:

> Section 3.3.2 of [The Llama 3 Herd of Models](<https://arxiv.org/abs/2407.21783>)

## Application Scenario

**Backend limitation:** When CP>1, `--transformer-impl transformer_engine` must be set.

KVAllGather computes the attention corresponding to the local query by gathering the complete keys and values. For the supported mask scope, see the notes below.

In scenarios such as GQA (Grouped-Query Attention) and MQA (Multi-Query Attention), the time required for all-gather communication of keys and values is far less than the overall computation time, making the benefits of this scheme even more significant.

This approach is fully compatible with Flash Attention, which is enabled by default for accelerated computation.

## Usage

| Training Type | Supported Mask Types |
| --- | --- |
| Standard training | causal |
| EOD Reset | causal, and the [EOD Reset configuration constraints](eod-reset.md) must be satisfied |

`--cp-comm-type allgather` is equivalent to `all_gather`.

Use `--cp-comm-type` to select the communication mode.

| Important Parameter | Parameter Description |
|----------------------------------------------------|---------------------------------------------------------|
| --context-parallel-size [int]                      | Sets the number of processes for CP. The default is 1; configure according to user requirements. |
| --cp-comm-type <b>all_gather</b> | Long-sequence parallel algorithm option. Set to `all_gather` to enable KVAllGather long-sequence parallelism. |
| --seq-length [int]                                 | The length of the input sequence. |

## Effects

By partitioning the input sequence across multiple compute devices, this approach reduces memory consumption on individual devices. Compared to not using sequence parallelism, per-step latency increases, but computational efficiency improves over recomputation.

## Notes

1. When enabling KVAllGather long sequence parallelism, you must also set `--transformer-impl transformer_engine`; otherwise, the feature is not supported.
2. Both standard training and EOD Reset only support `--attention-mask-type causal`.
3. For fixed-length padding training scenarios, a load-balanced sequence partitioning method is used, and `--seq-length` must be divisible by `2 * context-parallel-size`.
4. For EOD Reset, `--variable-seq-lengths --micro-batch-size 1` must be set, and all layers uniformly use `all_gather`; the Ring-based per-document online padding is not used. The sequence length still needs to satisfy the native CP partitioning requirements.
