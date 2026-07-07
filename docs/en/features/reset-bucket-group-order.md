# Bucket Reordering

## Background and Challenges

During the training of large models, inconsistency between the model definition order and execution order is a very common issue, especially when redefining common transformer components or using multimodal large models. This directly leads to precision issues and computation-communication serialization problems when using the overlap-param-gather parameter. The current megatron 0.12.1 solution resolves the precision issue, but inevitably introduces the problem of serialized computation and communication.

## Solution

To address the above problem, a strategy for parameter bucket reordering is introduced. By recording the bucket_group order from the first iteration, it enables pipeline overlap of computation and communication in subsequent iterations, effectively improving resource utilization.

### When `overlap-param-gather` is enabled, enable `--reset-bucket-group-order`

During the first iteration, we record the order in which bucket groups are triggered. After the first forward pass is complete, the bucket order is fully recorded. When the second iteration begins, except for the first bucket trigger which cannot overlap with computation, every subsequent prefetch of the next bucket's communication will overlap with the current computation.

## Application Scenario

This feature is suitable for training scenarios that use data parallelism strategies, and is especially effective when the model definition order is highly chaotic. In such cases, bucket communication is unordered, and there is significant serialization between computation and communication. Enabling overlap-param-gather alone does not yield significant results. Turning on the reset-bucket-group-order parameter can improve throughput by 0.85% on Pangu, and by approximately 1% on Llama2 when the model definition order is manually shuffled.

## Usage

* To enable the bucket reordering algorithm, add the following parameter to the training configuration:
    `--reset-bucket-group-order`
* Ensure the following three parameters are also enabled.
    `--use-distributed-optimizer`
    `--overlap-grad-reduce`
    `--overlap-param-gather`
