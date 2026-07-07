# Norm Recomputation

## Background and Challenges

Large model training often encounters insufficient memory issues.

## Solution

Similar to activation recomputation, this feature supports recomputation of Norm layers. It utilizes the checkpoint mechanism from the activation recomputation feature to perform recomputation on Norm layers.
For details, refer to the paper [Accelerating the Training of Large Language Models using Efficient Activation Rematerialization and Optimal Hybrid Parallelism](https://www.usenix.org/conference/atc24/presentation/yuan).

## Application Scenario

Primarily used in training scenarios when users encounter insufficient memory or need to further save memory.

## Usage

The following parameter configurations need to be added to the training script.

`--recompute-norm  # Enable Norm recomputation`
`--recompute-norm-num-layers ${num}   # num indicates the number of layers for Norm recomputation`

### Notes

* The norm recomputation feature is only supported on the mcore branch, not the legacy branch. That is, it can only be enabled via `--recompute-norm` when `--use-mcore-models` is turned on.
* Norm recomputation is compatible with simultaneous activation recomputation and full recomputation:
  * When enabled simultaneously, only `--recompute-method` set to `block` is supported.
  * When enabled simultaneously, each type of recomputation will be applied to the specified number of layers for full recomputation and norm recomputation respectively. That is, no single layer will undergo both full recomputation and norm recomputation.

* Execution priority: full recomputation layers are calculated first, followed by norm recomputation layers.

## Application Effects

When enabled, it saves the output activation memory of RMSNorm/LayerNorm layers. Since norm computation is relatively fast, the impact on overall performance after recomputation is minimal. For scenarios where TP and SP are enabled, the effect is less noticeable because the activation memory is already sharded within the TP domain. For models that do not use TP and SP, this feature can be considered.
