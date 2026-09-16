# Megatron Fully Sharded Data Parallelism (FSDP)

## Background and Challenges

As large model weights increase, there is a need to further improve memory utilization efficiency. The previous ZeRO-1 operation only shards the optimizer states within the DP domain, but does not shard the model weights and gradients. This makes model weights and gradients account for the bulk of static memory usage. FSDP can also shard weights and gradients within the DP domain, thereby further reducing the size of static memory.

## Solution

Each DP rank only holds a shard of the parameters. Before the forward pass of each weight block, an All Gather is performed within the DP domain. After the forward pass, the gathered weights are released, retaining only the shard. Before the backward pass, an All Gather is first performed to obtain the complete weights. After the backward pass, a Reduce Scatter is performed to sum the gradients across all DP ranks, while retaining only the shard corresponding to that DP rank.

## Application Scenario

When DP > 1, the model weights occupy a large amount of memory, and you want to further shard the weights and gradients to save memory.

## Usage

To enable fully sharded data parallelism, add the following configuration:

```bash
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params
--no-gradient-accumulation-fusion
--use-distributed-optimizer
--ckpt-format fsdp_dtensor
```

`core_r0.18.0` uses the Megatron FSDP interface. The legacy `--use-custom-fsdp` flag remains supported and is mapped to `--use-megatron-fsdp` before Megatron argument validation. The checkpoint format must be `fsdp_dtensor`.

You need to disable `CUDA_DEVICE_MAX_CONNECTIONS`.

```bash
unset CUDA_DEVICE_MAX_CONNECTIONS
```

## Effects

Sharding parameters, gradients, and optimizer states can reduce persistent training-state memory. Parameter all-gathers add communication, while separate reduce-scatter outputs and communication dtype conversion require temporary buffers. Measure peak memory and iteration time with the target model and parallel configuration.

> [!NOTE]
>
> MindSpeed adapts the basic functionality of this feature. It is not recommended to combine it with other features in the repository. For usage, [refer to this script](../../../tests_extend/system_tests/feature_tests/custom_fsdp.sh).

The 0.18 adaptation uses the new gradient bucket reduction pipeline and preserves communication dtype conversion, double buffering, and asynchronous release. The HCCL path uses separate reduce-scatter output storage. In the SUM path, gradients are scaled once before communication without relying on NCCL premultiplied SUM. The MoE router follows the 0.18 implementation, preserving its bias and router dtype support.

The existing `custom_fsdp.sh` MoE test from master is reused. In 0.18, `--moe-grouped-gemm` requires `--transformer-impl transformer_engine`. Training accuracy and `fsdp_dtensor` checkpoint save/restore still need validation on multiple Ascend devices.

## Dependencies and Support Boundaries

Use a TransformerEngineNPU version containing the DTensor FusedAdam fix, such as commit [5c85c0d](https://gitcode.com/GuoHaifeng1999/TransformerEngineNPU/commit/5c85c0df3e9c1887a4aa5a000f96b2f01d9fb031), and verify that the training process imports that installation. Older TENPU versions reject DTensor parameters; selecting `fsdp_dtensor` alone does not resolve this optimizer error.

| Item | Current adaptation |
|------|--------------------|
| Basic FSDP | Megatron FSDP in Core 0.18; the existing script uses `optim_grads_params`, not PyTorch FSDP2 |
| Parameters and gradients | Unquantized FP32/FP16/BF16 DTensor parameters, ordinary or decoupled gradients, and empty local shards through the matching TENPU version |
| Optimizer states | Floating-point moments and the existing FP32/FP16 master-weight paths |
| Communication lifecycle | Upstream communication dtype, double buffering, and asynchronous release are preserved; target-device validation remains necessary |
| HSDP/HFSDP and AVG | Existing branches are retained with separate RS outputs; multi-device combinations remain unverified |
| Quantization and remainders | The TENPU DTensor path does not support quantized model parameters, FP8/uint8 moments, or INT16 parameter remainders |
| Checkpoints | Existing `fsdp_dtensor` save/load integration; real multi-device I/O and recovery with a different DP size remain unverified |
| Feature combinations | Arbitrary combinations are not guaranteed; the existing MoE script also requires Ascend accuracy and performance validation |

These are implementation boundaries, not claims that all configurations have passed NPU end-to-end tests. CPU local-shard checks do not replace HCCL or real checkpoint validation. The Core 0.18 checkpoint default remains `torch_dist`; explicitly selecting `fsdp_dtensor` does not convert existing checkpoints.

The default script exits at iteration 50 but uses a save interval of 2000, and it includes `--no-load-optim` and `--no-load-rng`. To validate full recovery, configure save/load directories and an appropriate save interval in the same existing test, remove those two options, and compare the next-step parameters, optimizer states, and loss. Prepare the data and tokenizer paths before running it.
