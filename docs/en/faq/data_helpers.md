# Data Helpers Overflow Bug

## Symptom

After increasing parameters such as `gbs` and `iteration` that theoretically do not affect model memory, an OOM occurs, or the following error is reported during the model dataset preprocessing stage:

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

## Root Cause

In the `build_sample_idx()` function within the `megatron/core/datasets/helpers.cpp` file, an int32 array `sample_idx` is created to record the index of each sample. The index of each sample is calculated using the int64 variable `doc_idx_index`. An overflow may occur during the assignment operation `sample_idx[2 * sample_index] = doc_idx_index;`. When the sentences in the dataset are short and the required training steps *Global Batch Size* Sequence Length is large, `doc_idx_index` may exceed the representable range of int32, causing the final index to overflow.

## Solution

### Workaround

Reduce the number of model training steps.

### Recommended Solution

1. Change the relevant variables to the int64 data type. For details, see [fix data helps overflow bug](https://github.com/NVIDIA/Megatron-LM/pull/598).
  You can run the `mindspeed -P` command in the Megatron-LM directory to apply the modification automatically.

    ```shell
      mindspeed -P
    ```

2. Delete the `helpers.cpython-xx-xxx-linux-gnu.so` file under `megatron/core/datasets/`.

3. Delete the generated dataset cache folder, for example `enwiki/my-t5_text_sentence/cache/GPTDataset_indices`.
