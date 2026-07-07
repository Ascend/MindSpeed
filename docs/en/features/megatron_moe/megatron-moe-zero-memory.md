# Megatron MoE Alltoall Dispatcher Memory Optimization

## Background and Challenges

The dynamic memory in MoE is relatively large. After using the overlap strategy, the dynamic memory further increases, making the memory wall problem severe. At this point, using ordinary recomputation will exacerbate the performance issues caused by recomputation due to its coarse granularity.

## Solution

For this scenario, we use communication-heavy strategies, fine-grained recomputation, and targeted swapping to save memory, employing computation to hide the communication-heavy operations and swapping, and hiding recomputation with uncovered communication.
Supports `alltoall` and `alltoallseq` dispatchers.

- level0 performs recomputation in the expert computation part, with relatively small performance loss. level1 performs recomputation to a greater extent, with relatively more performance loss than level0. The memory saved is 70%+ and 90%+ of the memory that can be saved by recomputing the MLP, respectively, and the speed is better than recomputing the MLP.
- The MLP here also includes the shared expert part.
- In the `alltoall` branch, the probs recomputation is moved forward to further improve memory savings.

## Usage

Enable this feature by turning it on.
`--moe-zero-memory level0` or `--moe-zero-memory level1`

The following must also be enabled:

- `--moe-alltoall-overlap-comm` or `--moe-fb-overlap` to use this feature.
- If used with `--moe-fb-overlap`, refer to that feature's documentation for specific considerations.

Under level1, the number of layers for memory optimization can be configured. By default, it is enabled for all layers:

- `--moe-zero-memory-num-layers x`

Where x is the number of layers to set, x should be greater than or equal to 0 and less than or equal to the total number of model layers (num_layers//pp). Since level0 has minimal performance loss, it does not support configuring the number of layers and is enabled for all layers by default.

## Applicable Scenarios

1. Currently supports `alltoall` and `alltoall_seq` dispatcher modes, suitable for scenarios where megatron-moe requires recomputation.
2. Supports enabling level0 when `moe-fb-overlap` is active.
3. The `moe-zero-memory-num-layers` configuration is not supported when `--moe-fb-overlap` is enabled.
