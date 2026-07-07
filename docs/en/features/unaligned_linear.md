# Unaligned Linear

## Background and Challenges

Megatron-LM-like frameworks have become one of the mainstream solutions for large model training. TP (tensor parallelism) is a fundamental parallelism paradigm for large model training, but this paradigm still has shortcomings in certain scenarios. For example, it requires that the number of attention heads and the sequence length of the large model be divisible by TP; failure to meet this condition will cause an exception during parameter validation. This feature provides a solution for cases where the number of attention heads or the sequence length is not divisible by TP.

## Solution

- **Sequence length not divisible by TP**: Unlike the padding approach (which pads the sequence length to an integer multiple of TP), this solution addresses the issue through a sequence allocation strategy. TP ranks with an index less than **(seq_len%tp_size)** are allocated **(seq_len//tp_size+1)** sequence length, while the others are allocated **(seq_len//tp_size)** sequence length. For example, with seq_len=1026 and tp_size=4, tp0 and tp1 are allocated a sequence length of 257, while tp2 and tp3 are allocated a sequence length of 256.
- **Number of attention heads not divisible by TP**: For MHA-structured models, TP cards with index less than **(num_attention_heads%tp_size)** are allocated **(num_attention_heads//tp_size+1)** attention heads, while the remaining cards are allocated **(num_attention_heads//tp_size)** attention heads. For example, with num_attention_heads=25 and tp_size=4, tp0 is allocated 7 attention heads, while tp1, tp2, and tp3 are each allocated 6 attention heads. Notably, the TP sharding of the model's attention-related weights correlates with the number of heads. Assuming hidden_size=3200, the qkv_weight size is (9600,3200) and the dense_weight size is (3200,3200). The qkv weight size for tp0 is (2688,3200) and the dense weight size is (3200,896), while for tp1, tp2, and tp3, the qkv weight size is (2304,3200) and the dense weight size is (3200,768). Note that for GQA-structured models, weight sharding and attention head sharding are distributed according to the num_query_groups ratio.

## Application Scenario

- This feature can be used in scenarios where the sequence length or the number of attention heads is not divisible by TP.

## Usage

Add the --unaligned-linear parameter to the model parameters.

### Precautions

- Unaligned linear layers will cause load imbalance across TPs
- This feature does not support mc2, 2D tensor parallelism, CP feature (requires TP*CP to be divisible by the number of attention heads), etc.
- Special model architectures require special adaptation for this feature. Currently, MHA and GQA architectures are adapted, while MOE, MLA, and other architectures are not yet supported

### Setting Training Script Parameters

```shell
# Enable unaligned linear layer
--unaligned-linear \
```

## Application Effects

- **Supplementary feature scenario**: supplements scenarios where the number of attention heads or sequence length is not divisible by the TP size.
- **Potential performance impact**: the number of attention heads and sequence length processed by each TP are inconsistent, leading to load imbalance. It is recommended to consider this situation during model structure design.

In summary, this feature is designed to address the limitations in TP (tensor parallelism) scenarios. The feature itself introduces performance impacts due to load imbalance, so attention should be paid to this impact during model design and hyperparameter optimization.

## Notes

- Unaligned linear layers are not supported on the legacy branch, meaning they cannot be enabled together with `--use-legacy-models`.
