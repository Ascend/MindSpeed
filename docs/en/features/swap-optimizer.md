# Swap Optimizer

## Background and Challenges

In large model training, forward and backward computations are typically performed in BF16 format, while gradient updates use FP32 format. As a result, the optimizer needs to retain one FP32 copy of the weights and two FP32 momentums, leading to a memory footprint of `parameter_count * 12` Bytes. This portion of memory is not used during the forward and backward phases, yet it increases peak memory usage and can cause OOM during model training. Although techniques such as distributed optimizers can reduce this memory footprint, they cannot eliminate it entirely, and the reduction ratio heavily depends on the degree of data parallelism (DP).

## Solution

This feature reduces peak memory usage by offloading optimizer states to host-side memory during the forward and backward phases, retaining only a logical view on the device side, and then loading them back to the device side during the step update phase.

## Approach

1. During optimizer initialization in `shard_fp32_from_float16_groups`, weights are copied from model weights (bf16) to optimizer weights (fp32).
To avoid impacting peak memory usage, each copied weight must be swapped to the host side immediately. The same applies during weight loading—a swap operation is performed after each weight is loaded. Since this only occurs during the initialization phase, the performance impact is negligible.
2. During the step phase, to enable parallelism between h2d and d2h, approximately `numel(shard_fp32_from_float16_groups) // swap_optimizer_times`
parameters are first issued for h2d operations in one batch, followed by AdamW computation and copying to model weights (bf16), and finally d2h to release memory.
3. Since d2h and h2d are asynchronous copies, to ensure correct timing, the second round of d2h must wait for the previous round's h2d operation to complete before issuing the second round.

![img.png](../figures/swap-optimizer.png)

## Application Scenario

Model training scenarios that use the distributed optimizer `--use-distributed-optimizer` and where `--optimizer-selection` is set to `fused_adamw`.

## Usage

`--swap-optimizer`: Enables the swap optimizer feature.

`--swap-optimizer-times`: Default value is 16. Sets the number of swap operations during the step update phase. A larger value increases parallelism, which can reduce performance degradation but will increase peak memory usage.

Recommended Configuration

```bash
export CPU_AFFINITY_CONF=1,lazy_bind:0
```

This configuration enables the coarse-grained core binding mode, binding tasks to the NUMA CPU cores corresponding to the NPU. This effectively avoids cross-NUMA memory access and reduces scheduling overhead, thereby improving computational stability and performance.

## Notes

1. This feature is only applicable to model training scenarios where the distributed optimizer `--use-distributed-optimizer` is enabled and `--optimizer-selection` is set to `fused_adamw`.
2. This feature is currently incompatible with other optimizer-related features such as `--reuse-fp32-param` and the fused ema adamw optimizer.
