# Multi-Parameter Pipeline and Variable Sequence Lengths

## Background and challenges

In large-scale distributed training for deep learning, pipeline parallelism (PP) improves efficiency by partitioning a model into multiple stages and executing them concurrently on different devices. However, when processing complex multimodal data, PP faces new challenges:

- **For multi-parameter pipeline**: Traditional PP typically involves the transfer of only a single tensor. However, in multi-parameter passing scenarios, the transfer of multiple variables must be handled. This not only increases communication complexity but also requires precise management of attributes such as the shape and dtype of each variable.
- **For variable sequence lengths**: When the sequence length of input data is variable, the traditional approach is to pad all sequences to a uniform length, which leads to wasted memory and computational resources.

## Solution

To address these challenges, a series of optimizations have been developed:

- **Multi-parameter pipeline**: An efficient communication mechanism has been developed to support data transmission of various types and formats, and the backpropagation algorithm has been improved so that the system can automatically identify and process gradient information from multiple outputs.
- **Variable sequence lengths**: Support for dynamic shapes has been introduced, allowing sequences in each micro-batch to retain their original lengths. This is achieved by communicating the shape information of tensors in advance before sending them, synchronizing the shape of the data to be received across pipeline stages, and ensuring the accuracy of memory allocation and preprocessing.

## Application Scenario

- **Multi-parameter pipeline**: Suitable for tasks that need to process large amounts of multimodal data, such as large-scale neural network training involving text, images, and audio, where multiple parameters must be passed at each stage of the pipeline parallelism.
- **Variable sequence lengths**: Ideal for tasks with significant variations in text length, such as document classification and machine translation, while also enhancing the model's generalization capability.

## Usage

**NOTE**
Users need to modify the `args.pipeline_tensor_shapes` value in the `validate_args` function within the `mindspeed/features_manager/pipeline_parallel/multi_parameter.py` module to match the tensor transmission of the actual model's pipeline stages, including tensor dimensions (Shape) and data types (Dtype).

**Training Script Configuration**

- Support for PP scenarios

```shell
# Configuration example
# PP >= 2
--pipeline-model-parallel-size ${PP} \
--use-multiparameter-pipeline-model-parallel \
--variable-seq-lengths \
```

- Support for VPP scenarios

```shell
# Configuration example
# PP >= 2, num-layers-per-virtual-pipeline-stage is not None
--pipeline-model-parallel-size ${PP} \
--num-layers-per-virtual-pipeline-stage 1 \
--use-multiparameter-pipeline-model-parallel \
--variable-seq-lengths \
```

## Usage Constraints

1. Currently incompatible with the `--moe-fb-overlap` and `dualpipev` features.

## Application Effects

Simultaneously supports passing multiple parameters between pipeline stages and processing variable-length input data.
