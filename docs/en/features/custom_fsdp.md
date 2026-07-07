# Megatron Fully Sharded Data Parallel (FSDP)

## Background and Challenges

As large model weights increase, there is a need to further improve GPU memory utilization efficiency. The previous Zero-1 operation only shards the optimizer states within the DP domain, but does not shard the model weights and gradients. This makes model weights and gradients account for the bulk of static memory usage. FSDP can also shard weights and gradients within the DP domain, thereby further reducing the size of static memory.

## Solution

Each DP rank only holds a shard of the parameters. Before the forward pass of each weight block, an All Gather is performed within the DP domain. After the forward pass, the gathered weights are released, retaining only the shard. Before the backward pass, an All Gather is first performed to obtain the complete weights. After the backward pass, a Reduce Scatter is performed to sum the gradients across all DP ranks, while retaining only the shard corresponding to that DP rank.

## Application Scenario

When DP > 1, the model weights occupy a large amount of memory, and you want to further shard the weights and gradients to save memory.

## Usage

To enable fully sharded data parallelism, add the following configuration:

```bash
--use-custom-fsdp
--data-parallel-sharding-strategy optim_grads_params
--no-gradient-accumulation-fusion
--use-distributed-optimizer
```

You need to disable `CUDA_MAX_CONNECTIONS`.

```bash
unset CUDA_MAX_CONNECTIONS
```

## Application Effects

The model and weights are further sharded, reducing memory usage. However, because additional communication is introduced in each forward and backward pass, performance will decrease.

## Notes

MindSpeed adapts the basic functionality of this feature. It is not recommended to combine it with other features in the repository. For usage, [refer to the script](../../../tests_extend/system_tests/feature_tests/custom_fsdp.sh)
