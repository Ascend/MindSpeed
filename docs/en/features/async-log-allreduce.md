# Async Log Allreduce

## Background and Challenges

In distributed training, traditional synchronous log allreduce operations block the training pipeline, waiting for all processes to complete loss value computation and communication. This synchronous approach leads to the following issues:

1. Underutilization of computing resources due to idle waiting time
2. Training throughput is limited by the slowest process
3. Logging becomes a performance bottleneck, impacting overall training efficiency.

This is especially pronounced in large-scale distributed training, where the overhead of synchronous log allreduce becomes significant when using a large number of NPUs, potentially affecting the overall training speed.

## Solution

The Async Log Allreduce feature addresses the above challenges through the following approaches:

1. **Asynchronous communication**: Uses non-blocking allreduce operations, allowing the training process to continue without waiting for log communication to complete.
2. **Overlapped computation and communication**: Overlaps log-related all-reduce operations with subsequent training steps to improve hardware utilization.
3. **Deferred processing**: Synchronizes communication results only when logs need to be actually recorded, rather than waiting immediately.

This feature implements asynchronous communication through the `async_op=True` parameter of `torch.distributed.all_reduce`, and uses a special loss value handling mechanism to ensure data consistency.

## Use Scenario

This feature is particularly suitable for the following scenarios:

1. Large-scale distributed training (hundreds to thousands of NPUs)
2. Scenarios requiring frequent logging of training metrics
3. Scenarios where computation and communication need to be highly overlapped
4. Apps sensitive to training throughput

## Usage

Add `--async-log-allreduce` to the training script; it is disabled by default. Use the loss interface of `pretrain_gpt.py`; there is no need to manually initiate the log all-reduce in `loss_func`.

The third return value of a custom loss function should be a metrics dictionary: each entry is either a two-element Tensor containing the "loss sum, token count" or a scalar Tensor that has already been reduced. The metric names and shapes must be consistent across different microbatches.

## Effects

Enabling the Async Log Allreduce feature brings the following improvements:

1. **Larger training throughput**
2. **Higher resource utilization**
