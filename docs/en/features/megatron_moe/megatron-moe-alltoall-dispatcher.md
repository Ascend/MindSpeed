# Alltoall Dispatcher Optimization

## Background and Challenges

### 1. repeat_interleave Parallelism

In the Alltoall dispatcher branch, the repeat_interleave operator is invoked. This operator uses only a single block dimension for serial computation on a single launch stream, resulting in long execution times. Moreover, the operator's output is not used until after the alltoall, permute, and alltoallv operations.

### 2. Computation-Communication Overlapping

In the alltoall branch, the permutation function performs an allgather operation at the end to complete the split H dimension of all tokens, and then performs expert computation on the data blocks. This operation is serial, but since there is no dependency between tokens of different experts, it can be modified to a parallel operation.

## Solution

### 1. For repeat_interleave Parallelism

By creating a new dispatch stream, the repeat_interleave operator invocation is offloaded to the new stream. When block dim resources are sufficient, the two operators can be computed in parallel, saving time.

### 2. For Computation-Communication Overlapping

Tokens can be split according to the requirements of each expert, and then allgather communication and expert computation are performed on the tokens one by one. Since the computation of the first expert depends only on the first communication and there is no dependency between experts, the communication for the second expert can proceed concurrently while the first expert's computation is being performed, achieving computation-communication overlap.

## Application Scenario

When using mcore MoE, the `--moe-token-dispatcher-type alltoall` option is enabled.

## Usage

Enable the `--moe-permutation-async-comm` parameter.

### Notes

When `--moe-grouped-gemm` is enabled, expert computation is merged into a single operator, so the computation-communication overlap optimization becomes ineffective.

## Application Effects

Enabling this feature reduces training time and improves performance.
