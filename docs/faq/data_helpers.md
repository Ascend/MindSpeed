# Data helpers overflow bug
## 问题现象
在增大 gbs、iteration 等理论上不影响模型内存的参数后，出现OOM现象，或者在模型预处理数据集的阶段报如下错误：
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

## 问题根因
在 `megatron/core/datasets/helpers.cpp` 文件里的 `build_sample_idx()` 函数中创建了 `sample_idx` 的 int32 数组去记录每个 sample 的 index，
而每个 sample 的 index 又是以 `doc_idx_index` 这个 int64 的变量去计算，在 `sample_idx[2 * sample_index] = doc_idx_index;` 这个赋值操作中存在溢出的可能。
在数据集中的句子较短，而要求训练的步数 * Global Batch Size * Sequence Length 较大的情况下就会出现 `doc_idx_index` 超过 int32 的表达范围而导致最终的 index 溢出。

## 解决方案

#### 规避方案
1. 减小模型训练步数


#### 推荐方案
1. 将相关变量修改为 int64 数据类型，具体可查看：[PR](https://github.com/NVIDIA/Megatron-LM/pull/598)

   > 可以在 Megatron-LM 目录下，运行`mindspeed -P`命令，自动完成修改。
   >```shell
   > mindspeed -P
   >```
2. 删除 `megatron/core/datasets/` 下面的 `helpers.cpython-xx-xxx-linux-gnu.so` 文件。
3. 删除已生成的数据集缓存文件夹，例如 `enwiki/my-t5_text_sentence/cache/GPTDataset_indices`。


## 备注
此问题为 Megatron-LM 原生问题，CPP 代码难以通过 monkey patch 的方式进行修改。已多次提交修复 PR，但似乎 Megatron-LM 较为封闭，无人管理且不接受来自社区的代码提交。