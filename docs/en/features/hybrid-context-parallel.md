# Hybrid Long-Sequence Parallelism

## Background and Challenges

From generative AI to scientific research models, long sequence training is becoming increasingly important. In the generative AI domain, tasks such as conversational AI, long document summarization, and video generation all require reasoning over long contexts in both spatial and temporal dimensions. Similarly, chapter-level and book-level summarization (involving tens or even hundreds of thousands of words) is also valued in conversational AI and summarization tasks. Existing parallelism methods such as data, tensor, and pipeline parallelism cannot perform partitioning along the sequence dimension. As the sequence dimension (S) grows, the training memory overhead increases at a rate of $O$($S^2$). Therefore, specific optimizations are needed for long sequence scenarios to meet the training requirements of long training scenarios.

Popular sequence parallelism schemes, Ulysses and Ring Attention, each have their own limitations.

Ulysses requires that the number of attention heads be divisible by the sequence parallelism dimension. In GQA and MQA scenarios, the size of sequence parallelism is limited, resulting in limited expansion of the sequence length.

The parallelism dimension of Ring Attention is not limited by the number of attention heads, so the sequence length can theoretically be extended indefinitely. However, compared to Ulysses, Ring Attention cannot fully utilize communication and computation bandwidth, and its performance is inferior to Ulysses when the sequence chunk size is small.

## Solution

Ulysses and Ring Attention are fused to implement hybrid sequence parallelism, thereby addressing the respective shortcomings of both approaches.
For details, see the paper [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719).

## Application Scenario

It is compatible with FlashAttention, which is now enabled by default.

The sequence parallelism dimension is divided into the Ulysses dimension and the Ring Attention dimension, and the product of the Ulysses dimension and the Ring Attention dimension is the sequence parallelism dimension.

## Usage

<table><thead>
  <tr>
    <th width='200'>Important Parameters</th>
    <th>Parameter Description</th>

  </tr></thead>
<tbody>

  <tr>
    <td rowspan="7"> --context-parallel-size [int]</td>
    <td>Required. Sets the long sequence parallelism size. Defaults to 1. Configure based on user requirements.</td>

</tr>
</tbody>

  <tr>
    <td rowspan="7"> --ulysses-degree-in-cp [int]</td>
    <td>Ensure that --context-parallel-size is divisible by this parameter and greater than 1.
<br>For example, when --context-parallel-size is set to 8, you can set --ulysses-degree-in-cp to 2 or --ulysses-degree-in-cp to 4.
<br>Also ensure that --ulysses-degree-in-cp is divisible by the number of attention heads.</td>
</tr>
<tbody>

  <tr>
    <td rowspan="7"> --context-parallel-algo<b>    hybrid_cp_algo</b></td>
    <td>Optional. Sets the long sequence parallelism algorithm.
<br>
ulysses_cp_algo: Enables Ulysses long sequence parallelism.
<br>
<b>hybrid_cp_algo</b>: Enables Hybrid long sequence parallelism.
<br>
megatron_cp_algo: Enables Ring Attention long sequence parallelism.</td>

  </tr>
  </tbody></table>

Hybrid long sequence parallelism supports Ring Attention long sequence parallelism-related features, including send-receive overlap functionality and mask computation type configuration.

## Application Effects

By leveraging multiple compute devices to partition the input sequence in parallel, the memory consumption of a single device is reduced. Compared to not enabling sequence parallelism, the per-step time increases, but computational efficiency is improved compared to recomputation.

## Acknowledgments

1. GitHub project address:
<https://github.com/feifeibear/long-context-attention>

2. Paper preprint address:
USP: A Unified Sequence Parallelism Approach for Long Context Generative AI
<https://arxiv.org/abs/2405.07719>
