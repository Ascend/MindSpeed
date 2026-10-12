# MindSpeed FAQ

## Data helpers overflow bug

### 问题现象

在增大gbs、iteration等理论上不影响模型内存的参数后，出现OOM现象，或者在模型预处理数据集的阶段报如下错误：

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

### 问题根因

在`megatron/core/datasets/helpers.cpp`文件里的`build_sample_idx()`函数中创建了`sample_idx`的int32数组去记录每个sample的index，
而每个sample的index又是以`doc_idx_index`这个int64的变量去计算，在`sample_idx[2 * sample_index] = doc_idx_index;`这个赋值操作中存在溢出的可能。
在数据集中的句子较短，而要求`训练的步数 * Global Batch Size * Sequence Length`较大的情况下就会出现`doc_idx_index`超过int32的表达范围而导致最终的index溢出。

### 解决方案

- 规避方案：

  减小模型训练步数。

- 推荐方案：

  - 将相关变量修改为int64数据类型，具体可查看[fix data helpers overflow bug](https://github.com/NVIDIA/Megatron-LM/pull/598)。
    可以在Megatron-LM目录下，运行`mindspeed -P`命令，自动完成修改。

  - 删除`megatron/core/datasets/`目录下`helpers.cpython-xx-xxx-linux-gnu.so`文件。

  - 删除已生成的数据集缓存文件夹，例如`enwiki/my-t5_text_sentence/cache/GPTDataset_indices`。

## Torch extensions卡住

### 问题现象

在模型运行时，卡在如下场景，且等待十几分钟无反应。

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

> **注意**：部分场景下日志重定向可能导致上述 `Using ~/.cache/torch_extensions/...` 编译日志未被打印，此时若编译阶段进程卡住且无其他输出，可手动进入 `~/.cache/torch_extensions/py3xx_cpu` 目录查看是否有 `.lock` 文件残留，以确认问题。

### 问题根因

此问题为PyTorch extension编译问题，编译开始前其中一个线程会生成`.lock`文件对编译文件夹进行锁定，其他线程会进行等待。
如果因为其他原因导致编译的线程中途被强制结束，`.lock`文件不会被清除，导致第二次编译开始时，所有的线程看到存在`.lock`文件，就都会开始进行等待。

### 解决方案

删除模型运行对应python版本的`~/.cache/torch_extensions/py3xx_cpu`文件夹后，重新启动程序。

## Megatron-LM 0.7.0版本长稳测试出现GradNorm为NaN

### 问题现象

在Megatron-LM 0.7.0版本中，采用mindspeed自定义`--tokenizer-type PretrainedFromHF`，长稳测试一定步数后发现loss抖动异常最终出现grad norm为nan的问题，报错示例如下：

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
File “/home/Megatron-LM/megatron/core/distributed/param and grad buffer.py", Line 114, in start_grad_s
ync
assert not norm.isnan( ), (
AssertionError: Rank 13: found NaN in local grad norm in backward pass before data-parallel communication collectie
ve. Device: 5, node: node-15-11
```

### 问题根因

- 问题场景使用的数据集生成时，增加了`--append-eod`参数，这会让每个数据sample末尾增加一个eos结束标志位。
- megatron0.7.0对数据集提取过程增加了pad功能（在`class GPTDataset`类中），`PretrainedFromHF`模式下，会将pad标志位与eos标志位配成相同值（`pad_token_id == eos_token_id`）。loss_mask中会去掉pad标志位，但实际去掉的都是eos标志位。
- 以上两个原因综合导致了grad norm为nan的问题，这个问题是megatron原生问题，相同配置下实测GPU中也会报错。

### 解决方案

在`--tokenizer-type PretrainedFromHF`模式下，不使用`--append-eod`生成数据集。

## 训练报MindSpeed-Ops无法导入

### 问题现象

默认参数训练在首次反向传播时报错：

```text
RuntimeError: Fused wgrad GEMM accumulation on NPU is provided by the MindSpeed-Ops
ATB MatmulAdd operator, but the mindspeed_ops package or its atb api cannot be imported.
Install MindSpeed-Ops, or disable the fused path with the Megatron-LM argument
--no-gradient-accumulation-fusion.
```

### 问题根因

Megatron-LM默认开启`gradient_accumulation_fusion`。MA将其权重梯度融合累加接口补丁到MindSpeed-Ops的MatmulAdd实现，但MindSpeed-Ops是独立安装包，不会随MindSpeed自动安装。安装和导入MindSpeed本身不会触发该惰性依赖；模型在首次反向传播真实调用融合接口时，环境中没有`mindspeed_ops`或其`api.atb`子模块，才会抛出错误。

### 解决方案

二选一：

- **安装MindSpeed-Ops（推荐）**：按照[软件安装](user-guide/install_guide.md)完成安装并执行精确API导入验证，以保留权重梯度融合路径。
- **关闭融合路径**：训练参数增加`--no-gradient-accumulation-fusion`。该方式无需MindSpeed-Ops，但会引入中间梯度张量和额外累加操作。

<!-- npu="A3,910b" id1 -->
<term>Atlas A2训练系列产品</term>、<term>Atlas A3训练系列产品</term>且主梯度为fp32时，还需确认CANN-NNAL环境已加载、`ATB_HOME_PATH`有效，并具备C++/Ninja编译工具链。
<!-- end id1 -->

## get_batch 重复构造配置导致 Host 开销增加

### 问题现象

升级到 Megatron-LM core_r0.18.0 后，如果训练性能分析显示 `get_batch` 在 Host 侧耗时较高，且 `core_transformer_config_from_args` 被反复调用，可以考虑开启 `cache-get-batch-config`。这类开销随 microbatch 的重复执行累积，可能影响训练吞吐。

### 问题根因

训练入口的 `get_batch` 会调用 `core_transformer_config_from_args(args)`，构造 `TransformerConfig`，用于判断流水线布局、MTP 层所在阶段等信息。即使模型配置没有变化，每次调用仍会重复构造和校验配置，增加 Host 侧开销；部分流水线中间阶段即使无需读取数据，也会先执行这一步。

### 解决方案

在已接入 MindSpeed 的训练启动参数中添加：

```bash
--cache-get-batch-config
```

`--cache-get-batch-config` 默认关闭，属于 optimization level 2 特性。当前 `--optimization-level` 默认值为 `2`；若脚本将其设置为 `0` 或 `1`，需要同时调整为 `2`。关闭配置缓存时，移除 `--cache-get-batch-config` 即可。

该特性的核心作用是复用 `get_batch` 所需的配置，减少重复构造开销。生效机制如下：

- MindSpeed 在 `pretrain` 入口根据传入的 `forward_step_func` 定位训练入口，并包装其中的 `core_transformer_config_from_args`。该入口需同时提供可调用的 `get_batch` 和 `core_transformer_config_from_args`。
- 首次由该 `get_batch` 直接发起配置构造时，正常创建并缓存配置；后续相同缓存键的调用直接复用。缓存键包含 args 对象标识、配置类、MLA 开关和异构层配置路径。
- 其他函数调用配置构造接口时仍执行原逻辑。缓存对象是 `TransformerConfig`，数据 batch 仍按原流程读取。

自定义训练入口若不满足上述条件，或通过其他辅助函数间接构造配置，则不会命中该缓存。使用时应保持相关配置稳定：原地修改同一个 args 对象中未纳入缓存键的字段，不会自动刷新已缓存的配置。

验证时保持其余训练参数一致，对比开启前后的 `get_batch` Host 耗时、稳态单步耗时和吞吐，并检查 loss 是否正常。若主要耗时来自数据读取或数据传输，应继续定位对应瓶颈。更多说明参见 [get_batch 配置缓存](features/cache-get-batch-config.md)。

## RoPE 融合默认开启，旧脚本需要显式使用 no-rope-fusion 关闭

### 问题现象

升级到 core_r0.18.0 适配版本后，即使训练脚本未添加旧参数 `--use-fused-rotary-pos-emb`，RoPE 仍可能走融合计算路径。仅删除旧的开启参数，已不能保证恢复未融合实现，迁移前后的性能或数值对比也可能因此使用不同的计算路径。

对于当前 MindSpeed 的 MLA YaRN 适配路径，未关闭融合时还可能遇到以下报错：

```text
AssertionError: MLA Yarn RoPE does not support RoPE fusion
```

### 问题根因

RoPE 融合的控制统一到 Megatron 的 `apply_rope_fusion` 字段，命令行采用“默认开启、显式关闭”的方式。`--no-rope-fusion` 将该字段设置为 `False`；旧参数 `--use-fused-rotary-pos-emb` 保留为兼容别名，将同一字段设置为 `True`，不再使用独立的旧字段控制融合。

因此，脚本没有传入旧的开启参数，不代表融合已关闭。融合是否适用还取决于模型路径：例如，当前 MindSpeed 的 MLA YaRN 适配实现要求 `apply_rope_fusion=False`。

### 解决方案

**保持默认融合行为：** 在已通过 `import mindspeed.megatron_adaptor` 接入 MindSpeed 的训练脚本中，使用 RoPE 位置编码即可，无需额外添加融合开启参数：

```bash
--position-embedding-type rope
```

**关闭融合：** 若需保持旧脚本的未融合行为、进行融合开关对比，或使用上述要求关闭融合的 MLA YaRN 路径，在已有训练参数中添加：

```bash
--no-rope-fusion
```

该参数关闭的是 RoPE 融合计算，模型仍使用原有的位置编码。恢复默认融合行为时，移除该参数即可。

旧脚本迁移时可按下表调整：

| 原有意图 | 当前推荐配置 |
| --- | --- |
| 开启 RoPE 融合 | 保留 RoPE 位置编码配置，无需添加旧的开启参数 |
| 不使用 RoPE 融合 | 显式添加 `--no-rope-fusion` |
| 保留旧参数 `--use-fused-rotary-pos-emb` | 仍可兼容，但新脚本建议使用默认开启方式 |

不要同时传入 `--use-fused-rotary-pos-emb` 和 `--no-rope-fusion`：两者写入同一个字段，解析结果受命令行先后顺序影响。

该变更的核心作用是统一融合开关，并提供明确的关闭方式。生效机制如下：

- MindSpeed 的 RoPE 适配属于 optimization level 0 基础补丁，在支持的 level 0/1/2 下均会注册。无需通过 `--optimization-level 2` 开启，也不能通过降低优化等级关闭融合。
- 参数解析将开关写入 `args.apply_rope_fusion`，随后传入模型配置。Megatron 的 RoPE 分发逻辑据此选择融合或未融合路径；MindSpeed 的普通 RoPE 实现也读取该字段。关闭后，该实现使用旋转、乘法和加法组合计算。
- 默认开启仍受适用条件约束：非 RoPE 位置编码会在 Megatron 参数校验时关闭融合；开启 `--reset-attention-mask` 的 EOD Reset 场景也会由 MindSpeed 校验关闭融合。

验证时查看最终参数中的 `apply_rope_fusion`，并确认模型配置与之保持一致。固定模型、精度和输入，对比默认配置与增加 `--no-rope-fusion` 后的 loss、梯度及稳态单步耗时，确认实际计算路径和性能变化。更多说明参见 [Rotary Position Embedding 融合优化](features/rotary-embedding.md)。
