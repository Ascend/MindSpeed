# Megatron MoE Alltoall Dispatcher Memory Optimization

## Background and Challenges

The dynamic memory in MoE is relatively large. After using the overlap strategy, the dynamic memory further increases, making the memory wall problem severe. At this point, using ordinary recomputation will exacerbate the performance issues caused by recomputation due to its coarse granularity.

## Solution

For this scenario, we use communication-heavy strategies, fine-grained recomputation, and targeted swapping to save memory, employing computation to hide the communication-heavy operations and swapping, and hiding recomputation with uncovered communication.
Supports `alltoall` and `alltoall_seq` dispatchers.

- level0 performs recomputation in the expert computation part, with relatively small performance loss. level1 performs recomputation to a greater extent, with relatively more performance loss than level0. The memory saved is 70%+ and 90%+ of the memory that can be saved by recomputing the MLP, respectively, and the speed is better than recomputing the MLP.
- The MLP here also includes the shared expert part.
- In the `alltoall` branch, the probs recomputation is moved forward to further improve memory savings.

## Usage

Enable this feature by setting:
`--moe-zero-memory level0` or `--moe-zero-memory level1`

The following must also be enabled:

- `--moe-alltoall-overlap-comm` or `--moe-fb-overlap`
- If used with `--moe-fb-overlap`, refer to [AlltoAll Communication Overlap Across Microbatches in MoE](megatron-moe-fb-overlap.md).

Parameters:

- For `--moe-zero-memory level0`, since the performance loss is relatively small, layer count configuration is not supported. The feature is enabled for all layers.

- For `--moe-zero-memory level1`, all layers are enabled by default. You can also configure the number of layers to apply memory optimization using `--moe-zero-memory-num-layers x`, where `x` is the number of layers. The value of x should be greater than or equal to 0 and less than or equal to the total number of model layers (`num_layers//pp`).

## Applicable Scenarios

1. Currently supports `alltoall` and `alltoall_seq` dispatcher modes, suitable for scenarios where megatron-moe requires recomputation.
2. Supports enabling `level0` when `moe-fb-overlap` is active.
3. The `moe-zero-memory-num-layers` configuration is not supported when `--moe-fb-overlap` is enabled.
