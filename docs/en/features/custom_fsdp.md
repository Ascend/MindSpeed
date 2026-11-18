# Megatron Fully Sharded Data Parallelism (FSDP)

## Background and Challenges

As model weights grow, training requires more efficient use of device memory. ZeRO-1 shards optimizer states across the DP group, while model weights and gradients still occupy substantial persistent memory. FSDP also shards weights and gradients to reduce persistent training-state memory per DP rank.

This document describes the basic Megatron FSDP adaptation in `core_r0.18.0`, rather than PyTorch FSDP2.

## Solution

With `optim_grads_params`, each DP rank keeps a parameter shard. When forward or backward computation requires full parameters, an All Gather collects them across the DP group. Full parameters are released according to the upstream buffer lifecycle when no longer needed. Reduce Scatter reduces and shards backward gradients, and each rank updates its parameters using local gradient shards.

The 0.18 adaptation uses the new gradient bucket reduction pipeline and preserves communication dtype conversion, double buffering, and asynchronous release. HCCL reduce-scatter uses separate output storage. In the SUM path, gradients are scaled once before communication without relying on NCCL premultiplied SUM. The MoE router follows the 0.18 implementation, preserving bias and router dtype support.

## Application Scenario

Use FSDP when DP > 1 and weights, gradients, and optimizer states occupy substantial device memory that can be reduced by further sharding.

## Usage

1. Enable Megatron FSDP with the following configuration:

    ```bash
    --use-megatron-fsdp
    --data-parallel-sharding-strategy optim_grads_params
    --no-gradient-accumulation-fusion
    --use-distributed-optimizer
    --ckpt-format fsdp_dtensor
    ```

    The legacy `--use-custom-fsdp` flag from master remains supported and is mapped to `--use-megatron-fsdp` before Megatron argument validation. The default checkpoint format in 0.18 is `torch_dist`; Megatron FSDP requires explicitly selecting `fsdp_dtensor`. Changing this option does not convert existing checkpoints.

2. Unset `CUDA_DEVICE_MAX_CONNECTIONS` after loading the environment setup script:

    ```bash
    unset CUDA_DEVICE_MAX_CONNECTIONS
    ```

3. When using `--moe-grouped-gemm` from the existing MoE example, also specify:

    ```bash
    --transformer-impl transformer_engine
    ```

4. Install a TransformerEngineNPU version containing the DTensor FusedAdam fix, such as commit [5c85c0d](https://gitcode.com/GuoHaifeng1999/TransformerEngineNPU/commit/5c85c0df3e9c1887a4aa5a000f96b2f01d9fb031), and verify that the training process imports that installation. Older TENPU versions reject DTensor parameters; changing the checkpoint format alone does not resolve this optimizer error.

## Support and Validation Status

| Item | Current adaptation |
|------|--------------------|
| Basic FSDP | Megatron FSDP in Core 0.18 is integrated; the existing script uses `optim_grads_params` |
| Parameters and gradients | The matching TENPU version supports unquantized FP32/FP16/BF16 DTensor parameters, ordinary and decoupled gradients, and empty local shards |
| Optimizer states | The matching TENPU version supports floating-point moments and the existing FP32/FP16 master-weight paths |
| Communication lifecycle | Upstream communication dtype, double buffering, and asynchronous release are preserved; target NPU validation remains necessary |
| HSDP/HFSDP and AVG | Upstream branches are retained with separate RS outputs; multi-device combinations remain unverified |
| Quantization and remainders | The TENPU DTensor path does not support quantized model parameters, FP8/uint8 moments, or INT16 parameter remainders |
| Checkpoints | Existing `fsdp_dtensor` save/load paths are integrated; real multi-device save/restore and recovery with a different DP size remain unverified |
| Feature combinations | Arbitrary combinations with other repository features are not guaranteed; training accuracy and performance of the existing MoE script also require validation on multiple Ascend devices |

Integration means that the basic source adaptation is complete, rather than that every configuration has passed NPU end-to-end tests. TENPU CPU local-shard checks do not replace HCCL training or real checkpoint recovery validation.

## AG stream and start-sync prefetch optimization

To align memory behavior with the master branch, enable `--fsdp-memory-compatibility` to apply both optimizations below. The option defaults to false and requires Megatron FSDP with optimization level >= 2.

```bash
--use-megatron-fsdp \
--fsdp-memory-compatibility
```

You can also enable FSDP with `--use-custom-fsdp`. See [Usage](#usage) above for the remaining configuration.

### 1: Dynamic calling-stream AG

MCore 0.18 creates a separate stream for parameter All Gather (AG). Using different streams for computation and AG can affect memory reuse by the caching allocator and increase reserved memory.

When enabled, the option passes `ag_stream=None` to `AllGatherPipeline`, so each AG uses the NPU stream current at the time of the call. This avoids using a fixed separate AG stream or capturing a temporary initialization stream. The RS stream, HCCL internal communication streams, and HSDP outer AG stream retain their existing behavior.

### 2: No extra start-sync prefetch

`start_param_sync` can gather the first parameter bucket after an optimizer update and before the next forward pass. MCore 0.18 may also prefetch subsequent buckets at this stage, bringing full parameters into memory early and increasing peak memory.

When enabled, the option sets `prefetch=False` for forward parameter gathering by the matching instance only during normal asynchronous `start_param_sync`. First-bucket gathering and its wait behavior are preserved to align start-sync behavior with master. Ordinary forward and backward prefetch, forced synchronization, the AG prefetch threshold, and RS queue capacity retain their existing behavior.

## Effects

Further sharding weights, gradients, and optimizer states can reduce persistent training-state memory.
