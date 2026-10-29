# Allgather Dispatcher Optimization

## Background and Challenges

### 1. Gather/Scatter Replacement

In the Allgather branch of Megatron MoE, gather/scatter operations are used. The gather/scatter functionality performs element-wise value retrieval/assignment along a dimension axis based on indices. This operation involves a large number of random memory addresses, which significantly impacts performance.

In Megatron MoE, the gather/scatter calls are primarily invoked in the following manner: the index is expanded along dimensions using the expand operation, and the expanded index is then used to perform element-wise value retrieval/assignment on hidden_states.

```python
self.global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
local_hidden_states = torch.gather(global_hidden_states, 0, self.global_local_map)
```

### 2. Asynchronous Communication

In the Allgather dispatcher branch, the permute function performs allgather communication on three data items—hidden_states, max_ind, and max_prob—at the beginning. These operations are serial, but there is no serial dependency between the individual computation tasks.

## Solution

### 1. Gather/Scatter Replacement

Since index is expanded via expand, the content in each of its rows is identical. There is no need to use gather/scatter for element-wise operations. Instead, the index and indexput operators can be used for row-wise operations, achieving an equivalent substitution for gather/scatter.

### 2. Asynchronous Communication

By reordering communication tasks and issuing them asynchronously using the async=True parameter, computation and communication can be overlapped.

## Application Scenario

**Backend limitation:** When enabling the communication overlap optimization described in this document, `--moe-grouped-gemm --transformer-impl transformer_engine` must be set.

This optimization strategy applies to deep learning models deployed with the Mcore MoE (Mixture of Experts) architecture,
with the `--moe-token-dispatcher-type allgather` flag enabled.

## Usage

Set `--moe-token-dispatcher-type allgather` to select the Allgather dispatcher (the default type). For the supported scope of each dispatcher type, see [the type table](megatron-moe-alltoall-dispatcher.md#usage).

When enabling the asynchronous communication optimization, use `--moe-permutation-async-comm`, and refer to [Allgather Communication Overlap](megatron-moe-allgather-overlap-comm.md) to also configure `--moe-allgather-overlap-comm`, `--moe-grouped-gemm`, and `--transformer-impl transformer_engine`.

## Usage Constraints

- `--variable-seq-lengths` is not supported, including the CP+EOD scenarios that require this option; in such cases, use `alltoall`.
- `--moe-permute-fusion` and `--moe-shared-expert-overlap` are not supported; Allgather communication overlap uses the dedicated configuration mentioned above.

## Effects

According to actual test data, for a DeepSeek-V2-like MoE model at the billion-parameter level, after applying the above optimization measures, the end-to-end training performance improves by approximately 10%. This means it not only accelerates model convergence but also reduces the computing resources required to reach the same level of accuracy.
