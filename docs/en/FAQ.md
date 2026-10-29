# MindSpeed FAQs

## Data helpers overflow bug

### Symptom

After increasing parameters such as `gbs` and `iteration` that theoretically do not affect model memory, an OOM occurs, or the following error is reported during the model's dataset preprocessing stage:

```shell
Traceback (most recent call last):
  File "pretrain_gpt.py", line 121, in <module>
    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
  File "/home/ma-user/modelarts/user-job-dir/GPT-3-kernel_ID2728_for_PyTorch_zgcl/megatron/training.py", line 150, in pretrain
    process_non_loss_data_func)
  File "/home/ma-user/modelarts/user-job-dir/GPT-3-kernel_ID2728_for_PyTorch_zgcl/megatron/training.py", line 689, in train
    opt_param_scheduler)
  File "/home/ma-user/modelarts/user-job-dir/GPT-3-kernel_ID2728_for_PyTorch_zgcl/megatron/training.py", line 417, in train_step
    optimizer, fwd_bwd_timers, forward_only=False)
  File "/home/ma-user/modelarts/user-job-dir/GPT-3-kernel_ID2728_for_PyTorch_zgcl/megatron/schedules.py", line 654, in forward_backward_pipelining_without_interleaving
    timers, collect_non_loss_data)
  File "/home/ma-user/modelarts/user-job-dir/GPT-3-kernel_ID2728_for_PyTorch_zgcl/megatron/schedules.py", line 118, in forward_step
    output_tensor, loss_func = forward_step_func(data_iterator, model)
  File "pretrain_gpt.py", line 84, in forward_step
    data_iterator)
  File "pretrain_gpt.py", line 45, in get_batch
    data = next(data_iterator)
  File "/home/ma-user/anaconda/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 530, in __next__
    data = self._next_data()
  File "/home/ma-user/anaconda/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 570, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/home/ma-user/anaconda/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    return self.collate_fn(data)
  File "/home/ma-user/anaconda/lib/python3.7/site-packages/torch/utils/data/_utils/collate.py", line 157, in default_collate
    return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
  File "/home/ma-user/anaconda/lib/python3.7/site-packages/torch/utils/data/_utils/collate.py", line 157, in <dictcomp>
    return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
  File "/home/ma-user/anaconda/lib/python3.7/site-packages/torch/utils/data/_utils/collate.py", line 146, in default_collate
    return default_collate([torch.as_tensor(b) for b in batch])
  File "/home/ma-user/anaconda/lib/python3.7/site-packages/torch/utils/data/_utils/collate.py", line 138, in default_collate
    return torch.stack(batch, 0, out=out)
RuntimeError: stack expects each tensor to be equal size, but got [8193] at entry 0 and [8246] at entry 1
```

### Root Cause

In the `build_sample_idx()` function in the `megatron/core/datasets/helpers.cpp` file, an int32 array named `sample_idx` is created to record the index of each sample, while the index of each sample is computed using the int64 variable `doc_idx_index`.
The assignment `sample_idx[2 * sample_index] = doc_idx_index;` carries a potential overflow risk.
When the sentences in the dataset are short and the required `training steps * Global Batch Size * Sequence Length` is large, `doc_idx_index` may exceed the representable range of int32, causing the final index to overflow.

### Solution

- Workaround:

  Reduce the number of model training steps.

- Recommended solution:

  - Change the related variables to the int64 data type. For details, see [fix data helpers overflow bug](https://github.com/NVIDIA/Megatron-LM/pull/598).
    You can run the `mindspeed -P` command in the Megatron-LM directory to apply the modification automatically.

  - Delete the `helpers.cpython-xx-xxx-linux-gnu.so` file in the `megatron/core/datasets/` directory.

  - Delete the generated dataset cache folder, for example `enwiki/my-t5_text_sentence/cache/GPTDataset_indices`.

## Torch Extensions Stuck

### Symptom

During model execution, the process gets stuck in the following scenario and remains unresponsive for more than ten minutes.

```bash
Using ~/.cache/torch_extensions/py38_cpu as PyTorch extensions root...
Using ~/.cache/torch_extensions/py38_cpu as PyTorch extensions root...
Using ~/.cache/torch_extensions/py38_cpu as PyTorch extensions root...
Using ~/.cache/torch_extensions/py38_cpu as PyTorch extensions root...
Using ~/.cache/torch_extensions/py38_cpu as PyTorch extensions root...
Using ~/.cache/torch_extensions/py38_cpu as PyTorch extensions root...
Using ~/.cache/torch_extensions/py38_cpu as PyTorch extensions root...
Using ~/.cache/torch_extensions/py38_cpu as PyTorch extensions root...
```

> **Note**: In some scenarios, log redirection may prevent the above `Using ~/.cache/torch_extensions/...` compilation logs from being printed. In this case, if the process gets stuck during the compilation phase with no other output, you can manually enter the `~/.cache/torch_extensions/py3xx_cpu` directory to check whether any residual `.lock` files exist, so as to confirm the issue.

### Root Cause

This issue is a PyTorch extension compilation problem. Before compilation begins, one of the threads generates a `.lock` file to lock the compilation folder, while the other threads wait.
If the compiling thread is forcibly terminated midway for other reasons, the `.lock` file is not cleaned up. As a result, when compilation starts a second time, all threads see that the `.lock` file exists and begin waiting.

### Solution

Delete the `~/.cache/torch_extensions/py3xx_cpu` folder corresponding to the Python version used to run the model, and then restart the program.

## GradNorm Becomes NaN in Long-Duration Stability Test of Megatron-LM 0.7.0

### Symptom

In Megatron-LM 0.7.0, when using the MindSpeed custom `--tokenizer-type PretrainedFromHF`, after a certain number of steps in a long-duration stability test, the loss fluctuates abnormally and eventually `grad norm` becomes NaN. An example of the error is as follows:

```bash
2024-09-18 11:14:247 iteration 427/ 5000  consumed samples: 6832 elapsed time per iteration (
ms): 209.8 | Learning rate: 1.229919E-06 | global batch size:   16 | Lm loss: 8.567080E+00 | loss scale: 1.0 | grad norm: 35.518 | number of skipped iterations:   О | number of nan iterations: 0
[2024-09-18 11:14:25] iteration 428/   5000] consumed samples: 6848 elapsed time per iteration (
ms): 210.5 | Learning rate: 1.229826E-06 | global batch size: _ 16 | lm loss: 7.180392E+00 | loss scale: 1.0 | grad norm: 36.838 ] number of skipped iterations:   О | number of nan iterations:
Traceback (most recent call last):
File "pretrain_gpt.py", line 247, in <module>
pretrain(
File "/home/Megatron-LM/megatron/training/training.py", Line 274, in pretrain
iteration, num floating point operations so far = train(
File "/home/Megatron-LM/megatron/training/training.py", Line 1027, in train
train step(forward step func,
File "/home/Megatron-LM/megatron/training/training.py", Line 550, in train_step
losses reduced = forward backward func(
File "/home/Megatron-LM/megatron/core/pipeline parallel/schedules.py", line 1400, in forward backward
pipelining without interleaving
config.finalize model grads func(
File "/home/Megatron-LM/megatron/core/distributed/finalize model_grads.py", Line 113, in finalize mode
l grads
model chunk.finish grad sync()
File "/home/Megatron-LM/megatron/core/distributed/distributed data parallel.py", Line 248, in finish_g
rad sync
buffer.finish grad sync()
File "/home/Megatron-LM/megatron/core/distributed/param and_grad buffer.py", Line 513, in finish_grad
sync
bucket.finish grad sync()
File "/home/Megatron-LM/megatron/core/distributed/param and_grad buffer.py", Line 151, in finish_grad
sync
self.start grad sync()
File "/home/Megatron-LM/megatron/core/distributed/param and grad buffer.py", Line 114, in start_grad_s
ync
assert not norm.isnan( ), (
AssertionError: Rank 13: found NaN in local grad norm in backward pass before data-parallel communication collectieve. Device: 5, node: node-15-11
```

### Root Cause

- When generating the dataset used in the problem scenario, the `--append-eod` parameter was added, which appends an `eos` flag to the end of each data sample.
- Megatron 0.7.0 added a `pad` function to the dataset extraction process (in the `class GPTDataset` class). In `PretrainedFromHF` mode, the `pad` flag and the `eos` flag are configured to the same value (`pad_token_id == eos_token_id`). The `pad` flag is removed from `loss_mask`, but what is actually removed is the `eos` flag.
- The combination of the above two causes leads to `grad norm` being NaN. This is a native Megatron issue, and the same error also occurs on GPUs under the same configuration.

### Solution

In `--tokenizer-type PretrainedFromHF` mode, do not use `--append-eod` to generate the dataset.

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

**Keep the default fusion behavior:** In a training script already integrated with MindSpeed via `import mindspeed.megatron_adaptor`, simply use RoPE position encoding; no additional fusion-enabling parameter is required:

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
