# Megatron Distributed Checkpoints

## Background and Challenges

In the traditional weight saving and loading process, specifically in scenarios where the `--ckpt-format torch` parameter is specified in the training script, each card holds a complete copy of the optimizer state. During the saving phase, each card saves the full optimizer state to disk, resulting in data redundancy.
To reduce the data redundancy of optimizer states, although enabling the distributed optimizer allows each card to hold only the sharded optimizer state, thereby reducing runtime memory overhead and the disk space occupied by saved weight files, this strategy introduces All-Gather operations during the saving phase and Scatter operations during the loading phase, thus increasing communication overhead.

## Solution

To address the above issues, a fully sharded strategy is introduced, which fully shards both model parameters and optimizer states. During the weight saving and loading process, each card only saves and loads its respective shard of data, eliminating data redundancy. Compared with the distributed optimizer, this approach eliminates the All-Gather operation during the saving phase and the Scatter operation during the loading phase.

## Application Scenario

Supports TP, PP, CP, EP, and VPP parallel configuration scenarios as well as scenarios where Megatron native features are enabled. MindSpeed feature-enabled scenarios are not yet adapted.

## Usage

- Specify `--ckpt-format torch_dist` and `--save <weight save path>` in the script to enable the distributed weight saving feature. The model will be saved as distributed weights during the saving phase.
- `--auto-detect-ckpt-format` is an optional parameter used to automatically detect the weight format for loading. When this parameter is used in a training script that specifies `--ckpt-format torch_dist` and `--load <weight load path>`, the loading phase will automatically detect whether the weights are in `--ckpt-format torch_dist` format or `--ckpt-format torch` format and load them accordingly.

## Notes

1. When loading weights generated in the `--ckpt-format torch_dist` format, specify the same `--ckpt-format` parameter, i.e., use `--ckpt-format torch_dist`.
2. `--ckpt-format torch_dist` has not yet been adapted for scenarios where MindSpeed features are enabled.
3. In CP scenarios, `--ckpt-format torch_dist` currently only supports `--context-parallel-algo` set to `megatron_cp_algo`.

## Application Effects

After enabling distributed weights, the model and optimizer states are stored in shards during saving. Compared to the traditional weight format, this saves disk space for weight files. Compared to the distributed optimizer, it eliminates the All-Gather operation during saving and the Scatter operation during loading.
