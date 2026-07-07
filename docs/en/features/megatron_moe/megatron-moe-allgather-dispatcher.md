# Allgather Dispatcher Optimization

## Background and Challenges

### 1. Gather/Scatter Operation

In the Allgather branch of Megatron MoE, gather/scatter operations are used. The gather/scatter functionality performs element-wise value retrieval/assignment along a dimension axis based on indices. This operation involves a large number of random memory addresses, which significantly impacts performance.

In Megatron MoE, the gather/scatter calls are primarily invoked in the following manner: the index is expanded along dimensions using the expand operation, and the expanded index is then used to perform element-wise value retrieval/assignment on hidden_states.

```python
self.global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
local_hidden_states = torch.gather(global_hidden_states, 0, self.global_local_map)
```

### 2. Asynchronous Communication

In the Allgather dispatcher branch, the permute function performs allgather communication on three data items—hidden_states, max_ind, and max_prob—at the beginning. These operations are serial, but there is no serial dependency between the individual computation tasks.

## Solution

### 1. Gather/Scatter Operation

Since index is expanded via expand, the content in each of its rows is identical. There is no need to use gather/scatter for element-wise operations. Instead, the index and indexput operators can be used for row-wise operations, achieving an equivalent substitution for gather/scatter.

### 2. Asynchronous Communication

By reordering communication tasks and issuing them asynchronously using the async=True parameter, computation and communication can be overlapped.

## Application Scenario

This optimization strategy applies to deep learning models deployed with the Mcore MoE (Mixture of Experts) architecture, with the `--moe-token-dispatcher-type allgather` flag enabled.

## Usage

Enable the parameter `--moe-permutation-async-comm`.

## Application Effects

According to actual test data, for MoE models at the billion-parameter scale similar to DeepSeek-V2, the end-to-end training performance improved by approximately 10% after adopting the above optimization measures. This means that not only is the model convergence speed accelerated, but the computational resource consumption required to achieve the same level of accuracy is also reduced.
