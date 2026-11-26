# Ascend MC2

## Prerequisites

**Backend limitation:** MC2 only applies to the native ColumnParallelLinear/RowParallelLinear and does not cover the linear layers under the TE path (`--transformer-impl transformer_engine`, implemented by TENPU on Ascend).

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

For the MC2 operator interface, see [mc2_operators_api](https://www.hiascend.com/document/detail/en/Pytorch/60RC1/apiref/apilist/ptaoplist_000449.html).

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

> [!NOTE]
>
> **Applicable scope**: Example 2 only applies to the `--transformer-impl local` path. Under the `--transformer-impl transformer_engine` path, the linear layers are Megatron's TE wrapper classes and are not subclasses of `ColumnParallelLinear` / `RowParallelLinear`; `isinstance` is always `False` - **silent freezing failure: no error, no printing, and no parameters are frozen**.

The classes of the linear layers under the two paths:

| Module | `local` | `transformer_engine` |
| --- | --- | --- |
| `self_attention.linear_qkv` | `ColumnParallelLinear` | `TELayerNormColumnParallelLinear` |
| `mlp.linear_fc1` | `ColumnParallelLinear` | `TELayerNormColumnParallelLinear` |
| `self_attention.linear_proj` | `RowParallelLinear` | `TERowParallelLinear` |
| `mlp.linear_fc2` | `RowParallelLinear` | `TERowParallelLinear` |

- `local`: The linear layers use Megatron's native implementation; after enabling `--use-ascend-mc2`, they are replaced by `MindSpeedMC2Column/RowParallelLinear`, which still inherit from the native classes, so the `isinstance` check in Example 2 still matches.
- `transformer_engine`: On Ascend, the implementation is provided by TENPU (the `transformer_engine` package in the environment); the model's linear layers are Megatron's TE wrapper classes, located in `megatron.core.extensions.transformer_engine`, inheriting from TENPU's `te.pytorch.Linear` / `LayerNormLinear`.
- `TELayerNormColumnParallelLinear` incorporates LayerNorm and does not inherit from `TELinear`, so it must be matched separately; its `layer_norm_weight` remains trainable.

```python
# Example 3: freeze all column + row linear layers, applicable to both paths
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

FREEZE_TYPES = (ColumnParallelLinear, RowParallelLinear,
                TEColumnParallelLinear, TERowParallelLinear, TELayerNormColumnParallelLinear)
for name, module in model.named_modules():
    if 'output_layer' in name or not isinstance(module, FREEZE_TYPES):
        continue
    for param_name, param in module.named_parameters(recurse=False):
        if 'norm' in param_name:  # LayerNorm fused into the linear layer, kept trainable
            continue
        param.requires_grad = False
    print(f'rank:{torch.distributed.get_rank()}, frozen: {name}')
```

Example 3 must import the TE wrapper classes from `megatron.core.extensions.transformer_engine` (the model instance is exactly an instance of this class); `transformer_engine.pytorch.*` are TENPU's base classes, and the model contains no such instances; once TENPU changes to no longer inherit from them, it will silently fail again.

The failure is silent, so it is recommended to keep the printing and assert the count in CI: `number of ranks x (num_layers / pipeline_model_parallel_size) x 4` (with 8 devices/8 layers/PP=2, this equals 128, and freezing only row gives 64).

## Usage

For the native linear layer path, set `--transformer-impl local --use-ascend-mc2` to enable the MC2 operator.

TP must be greater than 1, and `--sequence-parallel` must be enabled.

## Effects

In training scenarios with TP and SP enabled, using MC2 can reduce memory overhead and improve computational efficiency.

## Notes

1. MoE models do not currently support enabling MC2.
2. In FP8 scenarios, only the `mxfp8` recipe is supported.
3. This feature is not supported on Atlas 900 A3 hardware.
