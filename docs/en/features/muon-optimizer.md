# Muon Optimizer

## Background and Challenges

Muon is a matrix parameter optimizer designed for large model training. Unlike element-wise optimizers such as Adam, Muon primarily operates on 2D weight matrices. It first maintains a momentum update similar to SGD, and then performs Newton-Schulz orthogonalization on the momentum matrix, so that the update direction is subject to better matrix structural constraints. For the numerous linear layer parameters in Transformers, Muon can serve as the primary optimizer for matrix parameters.

In large model training, not all parameters are suitable for Muon. 2D matrix weights are appropriate for Muon, while parameters such as embeddings, output layers, biases, and norms typically still use scalar optimizers like Adam. Therefore, when using Muon, it is necessary to manage two types of optimizers simultaneously and ensure they work together in processes such as learning rate, weight decay, gradient clipping, state saving, and parameter synchronization.

After enabling parallel capabilities such as TP, MoE, and ZeRO, Muon also encounters additional challenges:

1. **TP parameter duplicate filtering**: Under tensor parallelism, some parameters are duplicate views across different TP ranks. Duplicate parameters must be filtered out during gradient norm calculation, gradient clipping, and zero count statistics; otherwise, the statistical results will be double-counted.
2. **TP sharded matrix orthogonalization**: Muon's Newton-Schulz orthogonalization operates on matrix updates. After a matrix is sharded by TP, it is necessary to know the sharding dimension and communication group of the parameter in order to calculate the scaling factor based on the correct global matrix shape and perform cross-TP communication when needed.
3. **QKV fused weight handling**: Some models fuse the Q, K, and V weights into a single linear layer. Applying Muon's orthogonalization directly to the fused large matrix may not produce the expected results; instead, it needs to be split according to the Q/K/V structure and processed separately.
4. **MoE/expert parallel communication group selection**: Expert parameters may use the expert tensor parallel group, while dense parameters use the normal TP group. Muon must distinguish between dense and expert parameters to avoid using the wrong group for gradient statistics and the TP duplicate filter.
5. **ZeRO parameter sharding and synchronization**: When `--use-distributed-optimizer` is enabled, Muon follows the layer-wise distributed optimizer path. Each DP rank is only responsible for updating a portion of the complete parameters, and the updated parameters must be synchronized back to all ranks.
6. **bf16 master param attribute inheritance**: In bf16 training, the optimizer creates fp32 master params. Custom attributes such as `expert_tp` and `is_qkv` that Muon depends on need to be synchronized from model parameters to master params; otherwise, TP group selection and QKV splitting will fail.

## Solution

MindSpeed integrates Muon into the Megatron optimizer construction process through optimizer feature patching. When `--optimizer muon` is enabled, MindSpeed takes over the optimizer construction entry point, completing parameter classification, parameter tagging, Muon construction, scalar optimizer construction, and wrapper encapsulation.

The overall approach is as follows:

1. Scan model parameters and decide whether to use Muon or a scalar optimizer based on parameter shape and attributes.
2. Mark QKV weights with `is_qkv`, and mark expert tensor parallel parameters with `expert_tp`.
3. Build `TensorParallelMuon` for 2D matrix parameters.
4. Build a scalar optimizer for non-matrix parameters, defaulting to Adam, which can also be configured via `--muon-scalar-optimizer`.
5. Split optimizer wrappers by dense/expert parameters, so that each wrapper carries the correct `grad_stats_parallel_group` and `tp_group`.
6. In bf16 scenarios, extend the tensor-parallel attribute copy logic to synchronize the custom attributes required by Muon to the fp32 master param.
7. In the layer-wise distributed optimizer path, assign owner rank by complete parameters, and synchronize all-gather parameters after step.

## Implementation Principles

Muon only processes 2D matrix parameters that meet the conditions. For each parameter entering Muon, the optimizer maintains a `momentum_buffer`, and at each step, it first updates the momentum with the current gradient, then obtains the current update matrix based on whether Nesterov is enabled.

Subsequently, Muon performs Newton-Schulz iteration on the update matrix to obtain the approximately orthogonalized update direction. The MindSpeed local implementation supports different Newton-Schulz coefficients, iteration steps, and scale modes, and supports computing the global matrix shape based on the parameter sharding dimension in TP scenarios.

For non-matrix parameters, MindSpeed does not use Muon but places them into the scalar optimizer parameter group. This avoids applying inappropriate matrix orthogonalization logic to parameters such as embedding, bias, norm, and output layers.

## TP Adaptation

When TP is enabled, MindSpeed adapts Muon primarily in the following aspects:

1. **TP group awareness**

   `TensorParallelMuon` selects either a normal TP group or an expert TP group based on parameter attributes. Dense parameters use the normal TP group, while expert parameters use the expert TP group.

2. **TP duplicate filter**

   MindSpeed patches `param_is_not_tensor_parallel_duplicate` to support explicitly passing `tp_group`. During gradient norm calculation, gradient clipping, and zero count statistics, duplicate parameters are filtered based on the corresponding wrapper's `tp_group`.

3. **TP sharded matrix processing**

   Muon reads the parameter's `partition_dim`. When a parameter is TP-sharded along a certain dimension, that dimension is multiplied by the TP group size before computing the scaling factor, restoring the semantics of the global matrix shape.

4. **QKV split**

   For parameters marked as `is_qkv`, `TensorParallelMuon` splits the fused weight according to the Q/K/V dimensions in the model configuration, performs orthogonalization on the Q, K, and V submatrices respectively, and then concatenates them back to the original shape.

5. **bf16 master param attribute copy**

   bf16 training creates fp32 master params. MindSpeed extends the tensor-parallel attribute copy process to copy `expert_tp` and `is_qkv` from the original model parameters to the master params, ensuring that expert parameters and QKV parameters can still be correctly identified during the optimizer step.

## ZeRO Adaptation

When Muon is used with `--use-distributed-optimizer`, MindSpeed uses the layer-wise distributed optimizer path to adapt to ZeRO-like parameter sharding scenarios.

This path does not operate according to the contiguous parameter buffer sharding method of Megatron's native distributed optimizer. Instead, it assigns owner ranks based on complete parameters:

1. Sort all parameters by parameter size.
2. Assign owner ranks among DP/EP DP ranks in a ping-pong manner, balancing the parameter volume across ranks as much as possible.
3. Each rank's local optimizer only retains the parameters it is responsible for updating.
4. Gradients are still synchronized by DDP, so each rank can obtain the gradients corresponding to the complete parameters.
5. After the optimizer step, all-gather the updated parameters from each rank back to all ranks.

Different ranks may have different numbers of parameters, and the flat tensor lengths also differ. MindSpeed directly uses `torch.distributed.all_gather` to collect flat tensors of unequal lengths, then unflattens them according to each rank's parameter list and copies them back to the model parameters.

In the torch checkpoint scenario, the layer-wise distributed optimizer saves additional optimizer state files per DP rank, and each rank loads its own corresponding optimizer state during recovery, avoiding the use of optimizer state from only a single DP rank in the main checkpoint.

## Application Scenario

Muon is suitable for large model training scenarios where you want to use matrix orthogonalization updates for 2D matrix parameters such as Transformer linear layers, while retaining scalar optimizers like Adam to handle non-matrix parameters.

Recommended use cases include:

1. bf16 or fp32 large model training.
2. Using TP training and expecting Muon to correctly handle TP sharded matrices.
3. Using MoE/expert parallel training and needing to distinguish between dense and expert communication groups.
4. Using `--use-distributed-optimizer` and wanting to reduce the optimizer parameter update pressure on each DP rank through the layer-wise distributed optimizer.

## Usage

Enable Muon:

```bash
--optimizer muon \
--bf16
```

Enable the layer-wise distributed optimizer path for Muon:

```bash
--optimizer muon \
--bf16 \
--use-distributed-optimizer
```

Enable overlap of parameter gather and forward in the layer-wise distributed optimizer path:

```bash
--optimizer muon \
--bf16 \
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather
```

Common optional parameters are as follows:

```bash
--muon-momentum 0.95
--muon-nesterov
--muon-scale-mode spectral
--muon-fp32-matmul-prec medium
--muon-coefficient-type quintic
--muon-num-ns-steps 5
--muon-tp-mode blockwise
--muon-extra-scale-factor 1.0
--muon-scalar-optimizer adam
--apply-wd-to-qk-layernorm
```

Parameter descriptions are as follows:

| Name | Default Value | Description |
| --- | --- | --- |
| `--optimizer` | `adam` | Specifies the optimizer type. Set to `muon` to enable the Muon optimizer. |
| `--use-distributed-optimizer` | Off | When used with Muon, enables the layer-wise distributed optimizer path. |
| `--overlap-grad-reduce` | Off | Enables overlapping gradient reduce with backward computation; must be enabled when using `--overlap-param-gather`. |
| `--overlap-param-gather` | Off | In the layer-wise distributed optimizer path, delays parameter synchronization to the DDP forward pre-hook, triggering asynchronous gather via buckets. |
| `--overlap-param-gather-with-optimizer-step` | Off | This switch is currently not supported in the Muon path. |
| `--muon-momentum` | `0.95` | Momentum coefficient used internally by Muon to update the `momentum_buffer`. |
| `--muon-nesterov` | Off | Whether to use the Nesterov form in Muon's internal momentum update. |
| `--muon-scale-mode` | `spectral` | Scaling method for the Muon update matrix. Supports `spectral`, `unit_rms_norm`, and `shape_scaling`. |
| `--muon-fp32-matmul-prec` | `medium` | Precision configuration for fp32 matmul in Newton-Schulz iterations. |
| `--muon-coefficient-type` | `quintic` | Coefficient type used in Newton-Schulz iterations. |
| `--muon-num-ns-steps` | `5` | Number of Newton-Schulz iteration steps. |
| `--muon-tp-mode` | `blockwise` | Computation method for Newton-Schulz in TP scenarios. Supports `blockwise`, `duplicated`, and `distributed`. |
| `--muon-extra-scale-factor` | `1.0` | An additional scaling factor multiplied onto the Muon update. |
| `--muon-scalar-optimizer` | `adam` | Scalar optimizer used for non-matrix parameters. Supports `adam` and `lion`. |
| `--muon-no-split-qkv` | QKV split enabled by default | Disables the QKV fused weight splitting process. |
| `--apply-wd-to-qk-layernorm` | Off | Retains weight decay for Q/K layernorm; when enabled, other 1D parameters and biases still skip weight decay by default. |

To disable QKV split, configure:

```bash
--muon-no-split-qkv
```

## Application Effects

When enabled, MindSpeed selects the appropriate optimizer path for different parameter types:

1. 2D matrix parameters are updated using Muon's Newton-Schulz orthogonalization.
2. Non-matrix parameters such as embedding, output layer, bias, and norm use a scalar optimizer, defaulting to Adam.
3. In TP scenarios, Muon processes matrix orthogonalization, gradient statistics, and duplicate parameter filtering based on parameter sharding information and the TP group.
4. In MoE/expert parallel scenarios, dense/expert parameters carry their corresponding communication groups respectively.
5. In ZeRO/layer-wise distributed optimizer scenarios, each DP rank only updates the parameters it is responsible for and synchronizes parameters after the step; torch checkpoint saves and restores the corresponding optimizer state per DP rank.
6. When `--overlap-param-gather` is enabled, parameter synchronization is attached to the DDP bucket, and asynchronous gather is triggered by the forward pre-hook to reduce the blocking of parameter synchronization on the training step.

## Notes

1. The current Muon path does not support fp16. Please use bf16 or fp32.
2. The current Muon path does not support simultaneous use with MindSpeed custom FSDP or torch FSDP2.
3. When Muon is used together with `--use-distributed-optimizer`, it will follow the MindSpeed layer-wise distributed optimizer path.
4. The Muon path supports `--overlap-param-gather`, but requires enabling both `--use-distributed-optimizer` and `--overlap-grad-reduce` simultaneously.
5. The Muon path does not currently support `--overlap-param-gather-with-optimizer-step`.
6. Layer-wise all-gather relies on `torch.distributed.all_gather` supporting tensors of unequal length. It is still recommended to verify DP>1, EP>1, MoE, and checkpoint recovery scenarios on the target cluster.
7. The default scalar optimizer is Adam; if `--muon-scalar-optimizer lion` is configured, a Lion implementation must be provided separately.
