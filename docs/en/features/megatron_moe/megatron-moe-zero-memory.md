# Megatron MoE Alltoall Dispatcher Memory Optimization

## Background and Challenges

The dynamic memory in MoE is relatively large. After using the overlap strategy, the dynamic memory further increases, making the memory wall problem severe. At this point, using ordinary recomputation will exacerbate the performance issues caused by recomputation due to its coarse granularity.

## Solution

For this scenario, we use communication-heavy strategies, fine-grained recomputation, and targeted swapping to save memory, employing computation to hide the communication-heavy operations and swapping, and hiding recomputation with uncovered communication.
Currently supports the `alltoall` dispatcher.

- level0 performs recomputation in the expert computation part.
- The MLP here also includes the shared expert part.
- In the `alltoall` branch, the probs recomputation is moved forward to further improve memory savings.

## Usage

**Backend limitation:** `--transformer-impl transformer_engine` must be set, and `local` is not supported.

Set `--moe-zero-memory level0` and enable `--moe-alltoall-overlap-comm` or `--moe-fb-overlap`. This optimization applies to all MoE layers.

- When using `--moe-fb-overlap`, the [FB overlap usage constraints](megatron-moe-fb-overlap.md#usage-constraints) must be satisfied.
- `level0` cannot be used together with `--moe-apply-probs-on-input` or `--moe-latent-size`.

## Applicable Scenarios

Applicable to MoE `alltoall` dispatcher scenarios that require recomputation.
