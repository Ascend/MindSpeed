# Communication Overlap for Megatron MoE Allgather Dispatcher

## Background and Challenges

In MoE, a large amount of EP communication is not hidden, accounting for a significant proportion of the end-to-end time. This overhead can be overlapped with computation to improve model training performance.

## Solution

During the forward pass, asynchronous communication is used to overlap with computation as much as possible. At the same time, the entire computation flow is split into subgraphs, enabling communication-computation overlap during the backward pass as well, thereby accelerating model training. This feature is specifically optimized for the `allgather` dispatcher.

## Usage

Enable this feature by turning on `--moe-allgather-overlap-comm`.

The following must also be enabled:

- `--moe-permutation-async-comm`
- `--moe-token-dispatcher-type allgather`
- `--moe-grouped-gemm`, currently only supports Grouped MLP.

## Application Scenario

Applicable to megatron-moe, specifically the dropless scheme branch, when EP communication becomes a bottleneck and communication overlap for EP communication is needed.
Enabling this feature will increase memory usage, which is expected behavior.
