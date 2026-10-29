# MindSpeed core_r0.18.0 Version Upgrade Compatibility Changes

## Repeated Config Construction in get_batch Increases Host Overhead

### Symptom

After upgrading to Megatron-LM core_r0.18.0, if training profiling shows that `get_batch` takes a high amount of time on the host side and that `core_transformer_config_from_args` is called repeatedly, consider enabling `cache-get-batch-config`. Such overhead accumulates as microbatches are repeatedly executed and may affect training throughput.

### Root Cause

The `get_batch` of the training entry calls `core_transformer_config_from_args(args)` to construct a `TransformerConfig`, which is used to determine information such as the pipeline layout and the stage where the MTP layers reside. Even if the model configuration does not change, each call still reconstructs and validates the configuration, increasing host-side overhead; some intermediate pipeline stages perform this step first even when they do not need to read data.

### Solution

Add the following to the training launch arguments of a model already integrated with MindSpeed:

```bash
--cache-get-batch-config
```

`--cache-get-batch-config` is disabled by default and is an optimization level 2 feature. The current default value of `--optimization-level` is `2`; if a script sets it to `0` or `1`, it must also be changed to `2`. To disable the configuration cache, simply remove `--cache-get-batch-config`.

The core purpose of this feature is to reuse the configuration required by `get_batch`, reducing the overhead of repeated construction. The mechanism works as follows:

1. In the `pretrain` entry, MindSpeed locates the training entry based on the passed-in `forward_step_func` and wraps the `core_transformer_config_from_args` within it. This entry must provide both a callable `get_batch` and `core_transformer_config_from_args`.
2. When the configuration is constructed directly by this `get_batch` for the first time, it is created and cached as usual; subsequent calls with the same cache key reuse it directly. The cache key includes the args object identity, the configuration class, the MLA switch, and the heterogeneous layer configuration path.
3. When other functions call the configuration construction interface, the original logic is still executed. The cached object is the `TransformerConfig`, while data batches are still read through the original flow.

A custom training entry that does not meet the above conditions, or that constructs the configuration indirectly through other helper functions, will not hit this cache. Keep the related configuration stable during use: modifying fields of the same args object in place that are not part of the cache key does not automatically refresh the cached configuration.

To validate, keep the remaining training parameters unchanged and compare the `get_batch` time on the host, steady-state step time, and throughput before and after enabling the feature, and check whether the loss is normal. If the main cost comes from data reading or data transmission, continue to identify the corresponding bottleneck. For more details, see get_batch Config Cache.

## RoPE Fusion Is Enabled by Default; Old Scripts Must Explicitly Use no-rope-fusion to Disable It

### Symptom

After upgrading to the core_r0.18.0 adaptation version, RoPE may still use the fused computation path even if the training script does not add the old parameter `--use-fused-rotary-pos-emb`. Simply removing the old enabling parameter no longer guarantees a return to the unfused implementation, and performance or numerical comparisons before and after migration may therefore use different computation paths.

For the current MLA YaRN adaptation path in MindSpeed, the following error may also occur when fusion is not disabled:

```text
AssertionError: MLA Yarn RoPE does not support RoPE fusion
```

### Root Cause

Control of RoPE fusion is unified into Megatron's `apply_rope_fusion` field, and the command line adopts an "enabled by default, disabled explicitly" approach. `--no-rope-fusion` sets this field to `False`; the old parameter `--use-fused-rotary-pos-emb` is kept as a compatibility alias that sets the same field to `True`, and fusion is no longer controlled by an independent legacy field.

Therefore, the fact that a script does not pass the old enabling parameter does not mean fusion is disabled. Whether fusion applies also depends on the model path: for example, the current MLA YaRN adaptation implementation in MindSpeed requires `apply_rope_fusion=False`.

### Solution

**Keep the default fusion behavior:** In a training script already integrated with MindSpeed via `import megatron_adaptor`, simply use RoPE position encoding; no additional fusion-enabling parameter is required:

```bash
--position-embedding-type rope
```

**Disable fusion:** If you need to keep the unfused behavior of an old script, compare the fusion switch, or use the MLA YaRN path that requires fusion to be disabled as described above, add the following to the existing training arguments:

```bash
--no-rope-fusion
```

This parameter disables the RoPE fusion computation, while the model still uses the original position encoding. To restore the default fusion behavior, simply remove this parameter.

When migrating an old script, adjust it according to the following table:

| Original Intent | Current Recommended Configuration |
| --- | --- |
| Enable RoPE fusion | Keep the RoPE position encoding configuration; no need to add the old enabling parameter |
| Do not use RoPE fusion | Explicitly add `--no-rope-fusion` |
| Keep the old parameter `--use-fused-rotary-pos-emb` | Still compatible, but new scripts are advised to use the enabled-by-default approach |

Do not pass both `--use-fused-rotary-pos-emb` and `--no-rope-fusion` at the same time: they write to the same field, and the parsing result depends on the order of the command-line arguments.

The core purpose of this change is to unify the fusion switch and provide an explicit way to disable it. The mechanism works as follows:

1. The RoPE adaptation in MindSpeed is a basic patch at optimization level 0 and is registered under all supported levels 0/1/2. It does not need to be enabled via `--optimization-level 2`, nor can fusion be disabled by lowering the optimization level.
2. Argument parsing writes the switch to `args.apply_rope_fusion`, which is then passed to the model configuration. Megatron's RoPE dispatch logic selects the fused or unfused path accordingly; the regular RoPE implementation in MindSpeed also reads this field. When disabled, that implementation uses a combination of rotation, multiplication, and addition operations.
3. Being enabled by default is still subject to applicability conditions: non-RoPE position encoding disables fusion during Megatron argument validation; the EOD Reset scenario with `--reset-attention-mask` enabled also disables fusion through MindSpeed validation.

To validate, check `apply_rope_fusion` in the final arguments and confirm that the model configuration is consistent with it. With the model, precision, and inputs fixed, compare the loss, gradients, and steady-state step time between the default configuration and the one with `--no-rope-fusion` added, to confirm the actual computation path and performance changes. For more details, see [Rotary Position Embedding Fusion](features/rotary-embedding.md).
