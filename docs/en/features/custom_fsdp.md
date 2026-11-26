# Megatron Fully Sharded Data Parallelism (FSDP)

## Background and Challenges

As large model weights increase, there is a need to further improve memory utilization efficiency. ZeRO-1 only shards the optimizer states within the DP domain, and model weights and gradients still occupy a large amount of persistent memory. FSDP further shards weights and gradients to reduce the persistent training-state memory of each DP rank.

This document describes the basic adaptation of Megatron FSDP in `core_r0.18.0`, not PyTorch FSDP2.

## Solution

Under the `optim_grads_params` strategy, each DP rank persistently holds a parameter shard. When the forward and backward computations require complete parameters, an All Gather is performed within the DP domain; after the computation ends, the complete parameters that are no longer needed are released according to the upstream buffer lifecycle. The backward gradients are reduced and sharded via Reduce Scatter, and each rank uses its local gradient shard to update the corresponding parameters.

The 0.18 adaptation uses the new gradient bucket reduction pipeline and preserves communication dtype conversion, double buffering, and asynchronous release. The HCCL path uses separate reduce-scatter output storage. In the SUM path, gradients are scaled once before communication without relying on NCCL premultiplied SUM. The MoE router follows the 0.18 implementation, preserving its bias and router dtype support.

## Application Scenario

When DP > 1, the model weights, gradients, and optimizer states occupy a large amount of memory, and you want to further shard these training states.

## Usage

1. Enable Megatron FSDP and add the following configuration:

```bash
    --use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params
--no-gradient-accumulation-fusion
--use-distributed-optimizer
    --ckpt-format fsdp_dtensor
```

    The legacy `--use-custom-fsdp` from master remains supported and is mapped to `--use-megatron-fsdp` before Megatron argument validation. The Core 0.18 checkpoint default is `torch_dist`, and Megatron FSDP must explicitly use `fsdp_dtensor`; changing the format argument does not automatically convert existing checkpoints.

2. After loading the environment script, unset `CUDA_DEVICE_MAX_CONNECTIONS`:

    ```bash
    unset CUDA_DEVICE_MAX_CONNECTIONS
    ```

3. When using `--moe-grouped-gemm` in the existing MoE test case, you must also specify:

    ```bash
    --transformer-impl transformer_engine
    ```

4. Install a TransformerEngineNPU version containing the DTensor FusedAdam fix, such as a version containing commit [5c85c0d](https://gitcode.com/GuoHaifeng1999/TransformerEngineNPU/commit/5c85c0df3e9c1887a4aa5a000f96b2f01d9fb031), and verify that the training environment actually imports that installation package. Older TENPU versions reject the DTensor parameters provided by FSDP; only adjusting the checkpoint format cannot resolve the optimizer type error.

## Support Scope and Validation Status

| Item | Current adaptation |
|------|--------------|
| Basic FSDP | 0.18 Megatron FSDP is integrated; the existing script uses `optim_grads_params` |
| Parameters and gradients | With the matching TENPU, supports unquantized FP32/FP16/BF16 DTensor parameters, as well as ordinary gradients, decoupled gradients, and empty shard handling |
| Optimizer states | With the matching TENPU, supports floating-point moments, as well as FP32/FP16 master weights under the existing conditions |
| Communication lifecycle | Upstream communication dtype, double buffering, and asynchronous release are preserved; validation on the target NPU environment is required |
| HSDP/HFSDP and AVG | Upstream branches are retained and adapted for separate RS outputs; multi-device combination validation has not yet been completed |
| Quantization and remainder | The current TENPU DTensor path does not support quantized model parameters, FP8/uint8 moments, or INT16 parameter remainder |
| Checkpoints | The existing `fsdp_dtensor` save/load path is integrated; real multi-device save/restore and recovery with a changed DP size still need validation |
| Feature combinations | Arbitrary combinations with other features in the repository are not guaranteed; the training accuracy and performance of the existing MoE script also need validation in a multi-device Ascend environment |

The above "integrated" means that the basic adaptation of the source code is complete; it does not mean that all combinations have passed NPU end-to-end tests. TENPU's CPU local-shard validation cannot replace HCCL training and real checkpoint recovery validation.

## Effects

Sharding parameters, gradients, and optimizer states can reduce persistent training-state memory. Parameter all-gathers add communication, while separate reduce-scatter outputs and communication dtype conversion require temporary buffers. Measure peak memory and iteration time with the target model and parallel configuration.

> [!NOTE]
>
> The existing [custom_fsdp.sh script](../../../tests_extend/system_tests/feature_tests/custom_fsdp.sh) from master is reused, and no new test cases are added. Modify the data and tokenizer paths before running it.
>
> The default script exits at iteration 50 but uses a save interval of 2000, and it includes `--no-load-optim` and `--no-load-rng`, so a single default run cannot demonstrate full checkpoint recovery. To validate recovery, configure save/load directories and an appropriate save interval in the same existing test, remove these two no-load options, and compare the next-step parameters, optimizer states, and loss after recovery.
