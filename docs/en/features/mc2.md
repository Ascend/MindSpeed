# Ascend MC2

## Prerequisites

Only applicable to system environments with version identifiers `CANN 8.0.RC2` and `Ascend HDK 24.1.RC2`, and all subsequent iterative versions.

Attempting to use this configuration on non-specified versions may trigger system-level abnormal behavior, including but not limited to runtime errors.

## Background and Challenges

In large model training scenarios with TP and SP enabled, there is a strong dependency between matmul computation and all-reduce operations (when SP is not enabled), or between matmul computation and all_gather/reduce_scatter operations (when SP is enabled). When the model has a large number of parameters, both the communication volume and computation volume are substantial. Executing them serially introduces significant idle waiting time.

## Solution

To address the strong dependency between matmul computation and communication operations, Ascend has developed the MC2 solution.

MC2 fuses matmul computation with collective communication operations through fused operators, splitting large computation and communication tasks into smaller computation subtasks and communication subtasks. By pipelining these subtasks, communication and computation can overlap and mask each other, thereby reducing waiting and idle time and improving utilization.

## Approach

On the Python script side, MindSpeed fuses the originally serial matmul and all_gather/reduce_scatter operations through the MC2 fusion operator interface.

For the specific implementation, see [mc2_linears_seq_parallel](../../../mindspeed/core/tensor_parallel/ascend_turbo/mc2_linears_seq_parallel.py).

For the MC2 operator interface, see [mc2_opraters_api](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000449.html).

## Application Scenario

When TP and SP are enabled, it is recommended to enable MC2 for further optimization. Both frozen and unfrozen model weight scenarios are supported.

### Description

You can freeze weights by setting the `requires_grad` attribute to `False`.

```python
# Example 1: freeze all parameters
for param in model.parameters():
    param.requires_grad = False
```

```python
# Example 2: freeze all ColumnParallelLinear and RowParallelLinear except output_layer
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
for name, module in model.named_modules():
    if ('output_layer' not in name
            and (isinstance(module, ColumnParallelLinear) or isinstance(module, RowParallelLinear))):
        for param in module.parameters():
            param.requires_grad = False
```

## Usage

Set `--use-ascend-mc2` to enable the MC2 operator.

**You must also enable** `--sequence-parallel`.

## Application Effects

In training scenarios with TP and SP enabled, using MC2 can reduce memory overhead and improve computational efficiency.

## Notes

1. MoE models do not currently support enabling MC2.
2. The compute-communication parallel CoC feature --use-ascend-coc is not currently compatible.
3. This feature is not supported on Atlas 900 A3 hardware.
