# Hybrid Long-Sequence Parallelism

## Background and Challenges

From generative AI to scientific research models, long sequence training is becoming increasingly important. In the generative AI domain, tasks such as conversational AI, long document summarization, and video generation all require reasoning over long contexts in both spatial and temporal dimensions. Similarly, chapter-level and book-level summarization (involving tens or even hundreds of thousands of words) is also valued in conversational AI and summarization tasks. Existing parallelism methods such as data, tensor, and pipeline parallelism cannot perform partitioning along the sequence dimension. As the sequence dimension (S) grows, the training memory overhead increases at a rate of $O$($S^2$). Therefore, specific optimizations are needed for long sequence scenarios to meet the training requirements of long training scenarios.

Popular sequence parallelism schemes, Ulysses and Ring Attention, each have their own limitations.

Ulysses requires that the number of attention heads be divisible by the sequence parallelism dimension. In GQA and MQA scenarios, the size of sequence parallelism is limited, resulting in limited expansion of the sequence length.

The parallelism dimension of Ring Attention is not limited by the number of attention heads, so the sequence length can theoretically be extended indefinitely. However, compared to Ulysses, Ring Attention cannot fully utilize communication and computation bandwidth, and its performance is inferior to Ulysses when the sequence chunk size is small.

## Solution

Ulysses and Ring Attention are fused to implement hybrid sequence parallelism, thereby addressing the respective shortcomings of both approaches. For details, see the paper [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719).

## Application Scenario

The hybrid long-sequence parallel scheme is applicable to the following typical scenarios:

- It is compatible with FlashAttention, which is currently enabled by default.

- The SP dimension is divided into the Ulysses dimension and the Ring Attention dimension. The product of the Ulysses dimension and the Ring Attention dimension equals the overall sequence parallel dimension.

## Usage

<table>
 <thead>
    <tr>
      <th width='200'>Important Parameter</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default Value</th>
    </tr>
  </thead>
 <tbody>
  <tr>
    <td> --context-parallel-size [int]</td>
    <td>Sets the long-sequence parallel size. The default is 1; configure according to user requirements.</td>
    <td>No</td>
    <td>1</td>
  </tr>
  <tr>
    <td> --ulysses-degree-in-cp [int]</td>
    <td>This parameter must be greater than 1, and --context-parallel-size must be divisible by this parameter. For example, when --context-parallel-size is set to 8, you can set --ulysses-degree-in-cp to 2 or --ulysses-degree-in-cp to 4.
    <br>Additionally, ensure that --num-attention-heads is divisible by the product of --ulysses-degree-in-cp * --tensor-model-parallel-size.
    </td>
    <td>/</td>
    <td>/</td>
  </tr>
  <tr>
    <td> --context-parallel-algo</td>
      <td>
        Long-sequence parallel algorithm options:
        <ul>
          <li><code>ulysses_cp_algo</code>: Enables Ulysses long-sequence parallelism</li>
          <li><b>hybrid_cp_algo</b>: Enables Hybrid long-sequence parallelism</li>
          <li><code>megatron_cp_algo</code>: Enables Ring Attention long-sequence parallelism</li>
        </ul>
      </td>
    <td>No</td>
    <td>megatron_cp_algo</td>
  </tr>
  </tbody>
</table>

Hybrid long-sequence parallelism supports Ring Attention long sequence parallelism-related features, including send-receive overlap functionality and mask computation type configuration.

## Effects

By leveraging multiple compute devices to partition the input sequence in parallel, the memory consumption of a single device is reduced. Compared to not enabling sequence parallelism, the per-step time increases, but computational efficiency is improved compared to recomputation.
