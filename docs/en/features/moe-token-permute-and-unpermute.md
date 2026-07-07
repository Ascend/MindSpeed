# MoE Token Permute and Unpermute Fusion

## Background and Challenges

In the MoE architecture, the MoEAlltoAllTokenDispatcher scheduler is responsible for assigning tokens to various experts for processing and then reassembling the processed results back into the original token order. This process typically involves the following steps:
Token routing: determining which expert should process each token. This is accomplished through an expert gating mechanism, which selects the most suitable expert for each token.
Data rearrangement (Permute): grouping tokens by their selected experts so that each expert can process its assigned tokens in parallel. This typically involves permuting the tokens.
Expert processing: each expert processes its assigned tokens in parallel.
Result reassembly (Unpermute): after processing is complete, the results from different experts need to be reassembled back into the original token order.
In the above workflow, the data rearrangement and result reassembly steps are among the performance bottlenecks. This is because these two steps involve significant data movement, especially when using distributed training.

## Solution

To optimize this process, MindSpeed fuses the MoE Token Permute and Unpermute operations into a single operator, thereby improving model training performance.

## Usage

1. Add `--moe-permute-fusion` or `--use-fused-moe-token-permute-and-unpermute` to the launch script. Both are equivalent, but `--moe-permute-fusion` is recommended as the preferred option.
2. The following configuration is recommended for optimal performance; otherwise, enabling this fused operator may degrade performance in certain scenarios.
(1) When `--moe-token-dispatcher-type alltoall` is set, configure `--expert-tensor-parallel-size 1`. (2) When `--moe-token-dispatcher-type alltoall_seq` is set, enable `--moe-tp-extend-ep`.

## Usage Constraints

1. Supported dispatcher types: Currently, only `--moe-token-dispatcher-type alltoall` and `--moe-token-dispatcher-type alltoall_seq` are supported. `--moe-token-dispatcher-type allgather` is not yet supported.
2. Compatibility between the fusion operator and expert capacity parameters: To enable `--moe-expert-capacity-factor`, `--moe-pad-expert-input-to-capacity` must also be enabled for compatibility with the fusion operator. If only `--moe-expert-capacity-factor` is enabled without `--moe-pad-expert-input-to-capacity`, the fusion operator is not yet compatible.
3. System environment requirements: Limited to system environments with version identifiers `CANN 8.3.RC1` / `PTA 7.2.RC1` and all subsequent iterative versions.

## Application Effects

Enabling the fusion operator not only effectively saves memory resources but also improves model training performance.
