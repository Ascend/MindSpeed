# Unaligned Ring Long Sequence Parallelism

## Background and Challenges

With the development of generative AI and scientific research models, long sequence training has become increasingly important. However, traditional Ring sequence parallelism requires that the sequence length must be divisible by the Context Parallel size (CP size). This imposes limitations when processing dynamic or irregular inputs, especially in multimodal apps where the sequence length of input data may be unpredictable and frequently changing. Therefore, a mechanism is needed to support operations in these unaligned scenarios to accommodate a wider range of application scenarios.

## Solution

To address the limitations of traditional Ring sequence parallelism when processing unaligned sequence lengths, the "Unaligned Ring" mechanism establishes a shape negotiation protocol, exchanges effective length information before communication, and isolates changes through `get_unaligned_cp_shapes` to obtain the current rank length and target rank length. It transmits non-uniform sequences during chunked computation and communication, enabling RingAttention computation with non-uniform partitioning.

## Application Scenario

The "Unaligned Ring" feature is applicable to the following typical scenarios:

- **Multimodal learning**: When processing various types of data such as images, videos, and text, the sequence lengths of different data types vary significantly, making it difficult to unify them to a fixed CP size.
- **Real-time data analysis**: When processing streaming data, the arrival time of data is uncertain, resulting in potentially different sequence lengths for each processing operation.
- **Personalized recommendation systems**: The sequence lengths of user behavior data are usually different, and in such cases, support for unaligned operations is also required.

## Usage

To use the "unaligned Ring" feature, users need to pass the `shapes` parameter when calling the `ringattn_context_parallel` interface, as shown in the example code below.

```python
# Example code
 output = ringattn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, shapes=shapes)

```

`get_unaligned_cp_shapes` is an important function in unaligned Ring sequence parallelism. The main purpose of the `shapes` parameter is to obtain the sub-sequence length information after unaligned splitting of the sequence. Within the function, elements are accessed via shapes[block_id] and shapes[next_block_id], so the `shapes` parameter can be any data structure that supports index access through the [] operator, including but not limited to:

1. List: for example, [100, 100, 20]
2. Tuple: for example, (100, 100, 20)
3. Dictionary: for example, {0: 100, 1: 100, 2: 20}

The `get_unaligned_cp_shapes` function ultimately returns a list containing two elements: [shapes[block_id], shapes[next_block_id]], which correspond to the values at the block_id and next_block_id indices, respectively.

```python
def get_unaligned_cp_shapes(shapes, block_id, next_block_id):
    if shapes is None:
        return None
    unaligned_cp_shapes = [shapes[block_id], shapes[next_block_id]]
    return unaligned_cp_shapes
```

## Application Effects

By introducing "unaligned ring," the system improves its adaptability to different input lengths. This not only resolves the issues encountered by traditional ring sequence parallelism when processing dynamic or irregular input sequences, but also maintains good scalability.

## Notes

- Unaligned ring long sequence parallelism currently only supports the scenario where --attention-mask-type is general.
