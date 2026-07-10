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

### 问题根因

此问题为PyTorch extension编译问题，编译开始前其中一个线程会生成`.lock`文件对编译文件夹进行锁定，其他线程会进行等待。
如果因为其他原因导致编译的线程中途被强制结束，`.lock`文件不会被清除，导致第二次编译开始时，所有的线程看到存在`.lock`文件，就都会开始进行等待。

### 解决方案

删除`~/.cache/torch_extensions/py38_cpu`文件夹后，重新启动程序。

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
