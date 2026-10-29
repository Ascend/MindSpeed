# Alltoall Dispatcher Optimization

## Background and Challenges

### 1. repeat_interleave Parallelism

In the Alltoall dispatcher branch, the repeat_interleave operator is invoked. This operator uses only a single block dimension for serial computation on a single launch stream, resulting in long execution times. Moreover, the operator's output is not used until after the alltoall, permute, and alltoallv operations.

### 2. Computation-Communication Overlapping

In the alltoall branch, the permutation function performs an allgather operation to complete the split H dimension of all tokens, and then performs expert computation on the data blocks. This operation is serial, but since there is no dependency between tokens of different experts, it can be modified to a parallel operation.

## Solution

### 1. repeat_interleave Parallelism

By creating a new dispatch stream, the repeat_interleave operator invocation is offloaded to the new stream. When block dim resources are sufficient, the two operators can be computed in parallel, saving time.

### 2. Computation-Communication Overlapping

Tokens can be split according to the requirements of each expert, and then allgather communication + expert computation are performed on the tokens one by one. Since the computation of the first expert depends only on the first communication and there is no dependency between experts, the communication for the second expert can proceed concurrently while the first expert's computation is being performed, achieving computation-communication overlapping.

## Application Scenario

**Backend restriction:** When Grouped GEMM, Alltoall communication overlapping, or FB overlap is enabled, `--transformer-impl transformer_engine` must be set.

When using mcore MoE, the `--moe-token-dispatcher-type alltoall` option is enabled.

## Usage

The supported types of `--moe-token-dispatcher-type` are as follows:

| Type | Support |
| --- | --- |
| `alltoall` | Supported; you can select Alltoall communication overlapping, FB overlap, MC2, or permute fusion according to the constraints. |
| `allgather` | Supported, and it is the default type; it uses the dedicated [Allgather Configuration](megatron-moe-allgather-dispatcher.md) and does not support dynamic sequence lengths. |
| `flex` (`deepep/hybridep`) | Not supported; NPU adaptation has not been completed yet. |

Set `--moe-token-dispatcher-type alltoall` to select the Alltoall dispatcher. For the computation-communication overlapping configuration, see [Alltoall Communication Overlapping](megatron-moe-alltoall-overlap-comm.md).

To use `--moe-alltoall-mc2`, the scenario must be dropless, with `--expert-tensor-parallel-size 1` set and `--moe-expert-capacity-factor` not set.

## Effects

Enabling this feature reduces training time and improves performance.
