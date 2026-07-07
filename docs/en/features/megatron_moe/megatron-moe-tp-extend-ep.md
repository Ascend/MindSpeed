# Megatron MoE TP Extended EP

## Background and Challenges

When TP+EP is enabled, the expert layer TP group partitions expert parameters. In fine-grained small expert scenarios of MoE, the GMM operator efficiency drops significantly after TP partitioning.

## Solution

To address the GMM operator efficiency degradation after TP partitioning in small expert scenarios, the expert layer TP group does not partition expert parameters but instead partitions the number of experts.

## Use Cases

Fine-grained small experts, similar to DeepSeek-V2 models, where each expert has a small number of parameters.

## Usage

Enable `--moe-tp-extend-ep` to use this feature.

The following must also be enabled:

- `--moe-permutation-async-comm`
- `--moe-grouped-gemm`, currently only supports Grouped MLP.

Also ensure that `--num-experts` is divisible by `tp * ep`.

### NOTE

Currently, this feature does not support the Moe Token drop and pad mode, meaning `--moe-expert-capacity-factor` must be None.
Currently, only the alltoall_seq dispatcher is supported.

## Performance

By avoiding TP splitting of expert parameters, the GMM operator efficiency is improved in small expert scenarios, thereby enhancing the overall model training performance. For MoE models at the trillion-parameter scale similar to DeepSeek-V2 with fine-grained small experts, the performance can be improved by over 10%.
