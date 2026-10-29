# Communication Overlap for Megatron MoE AlltoAll Dispatcher

## Background and Challenges

In MoE, a large amount of EP communication is not overlapped with computation, accounting for a significant proportion of end-to-end time. This overhead can be reduced by interleaving communication with computation, thereby improving model training performance.

## Solution

During the forward pass, asynchronous communication is used to overlap with computation as much as possible. Meanwhile, the entire computation graph is partitioned into subgraphs, enabling concurrent communication and computation during the backward pass to accelerate model training.
This feature currently supports the alltoall dispatcher.
Additionally, the alltoall branch is compatible with Megatron's shared_expert_overlap and, through finer-grained overlapping, achieves further performance improvements over the native solution.

## Usage

**Backend limitation:** `--transformer-impl transformer_engine` must be set, and `local` is not supported.

Use the following parameters to enable communication overlap, and set `--expert-model-parallel-size` to greater than 1:

```bash
--moe-alltoall-overlap-comm \
--transformer-impl transformer_engine \
--moe-token-dispatcher-type alltoall \
--moe-grouped-gemm
```

- Only `alltoall` is supported; `allgather` and `flex` are not supported.
- `--delay-wgrad-compute`, `--overlap-dispatch-backward-with-experts-wgrad`, and `--overlap-moe-expert-parallel-comm` cannot be enabled at the same time.
- `--moe-zero-memory` only supports `disable` or `level0`.
- `--recompute-activation-function` cannot be used together with `--hybrid-layer-pattern`.
- Supports being enabled together with `--overlap-grad-reduce`.

## Application Scenario

This feature is suitable for scenarios using the megatron-moe dropless branch where training performance needs to be improved. Compared with the baseline dispatcher scenario, performance can be improved by over 10%.
After enabling `--moe-shared-expert-overlap`, performance can still be improved by more than 4%.
Enabling this feature will increase memory usage, which is normal. In this case, you can use the ZeroMemory feature to adjust memory usage. For an introduction to and configuration of ZeroMemory, see [here](megatron-moe-zero-memory.md).
