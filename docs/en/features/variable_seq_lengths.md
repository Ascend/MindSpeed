# PP Support for Variable Sequence Lengths

## Background and Challenges

In deep learning model training, especially when dealing with multimodal tasks, the sequence lengths of input data are often not fixed. For models that adopt the pipeline parallelism (PP) strategy, processing sequences of different lengths typically requires adjusting all sequences to a uniform length by padding or truncation. While this approach simplifies data processing and model design, it leads to wasted computational resources and memory, particularly when processing shorter sequences that require significant padding.
**Key challenges:**

- **Memory inefficiency**: Large amounts of padding can lead to low memory utilization.
- **Computational inefficiency**: Unnecessary computations are performed on the padded portions.

## Solution

To address the above challenges, support for dynamic shapes has been introduced, allowing sequences in each micro-batch to retain their original lengths. This feature synchronizes the shape of incoming data across pipeline stages by communicating tensor shape information in advance before sending the tensor, ensuring accuracy in memory allocation and preprocessing.

## Application Scenario

- **Variable-length text processing**: Tasks such as document classification and machine translation, where text lengths vary significantly.
- **Enhanced model generalization**: Enables the model to better adapt to inputs of various lengths, thereby improving its performance in real-world apps.

## Usage

### Precautions

- When using pipeline parallelism with fixed sequence lengths, enabling this feature introduces unnecessary communication overhead and is therefore not recommended.
- Closely monitor memory consumption during training to avoid overflow issues caused by variable sequence lengths.

### Setting Training Script Parameters

```shell
# Enable pipeline parallelism, PP >= 2
--pipeline-model-parallel-size ${PP} \
# Enable dynamic shape support for PP
--variable-seq-lengths
```

Limitations:

1. `--moe-token-dispatcher-type alltoall_seq` and `--moe-token-dispatcher-type allgather` are not currently supported.

## Application Effects

- **Optimized resource utilization**: Compared to traditional methods where all sequences must be padded to a uniform length, this solution reduces unnecessary padding operations, effectively saving memory space, lowering computational load, and improving overall performance.
- **Improved flexibility**: This feature gives the model greater adaptability, enabling it to efficiently process input data of various lengths, thereby enhancing the model's generalization capability. This is particularly important for tasks that require processing variable-length inputs (such as text classification, machine translation, etc.).
- **More authentic data representation**: Preserving the true length of the original text helps the model capture text features more accurately.
- **Potential performance impact**: Despite its many advantages, in certain scenarios (such as when pipeline parallelism is enabled and the original sequences are of equal length or need to be truncated to maintain consistent length), enabling this feature may increase complexity and slow down training speed. Therefore, these factors should be comprehensively considered during design and deployment to ensure optimal overall system performance.

In summary, PP support for dynamic shapes is an effective optimization approach for specific application scenarios. It can significantly improve resource utilization and data processing flexibility while maintaining model performance. Users should weigh the pros and cons based on actual circumstances to decide whether to enable this feature.
