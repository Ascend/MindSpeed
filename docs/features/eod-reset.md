# 支持EOD Reset训练场景

## EOD Reset训练场景
通常一个批次中输入进模型的文本序列是由多个文档（doc）拼接得到。在默认情况下，多个文档被视为同一序列，互相间的self attention没有掩盖。在特定情况下，多个文档间要求独立，文档间不能互相做self attention，在这种情况下attention mask和position ids需要在每个文档结束的位置（EOD）被重新设置。

## 解决方案
通过调用底层flash-attention算子的可变长模式，支持EOD Reset训练场景。同时在EOD Reset训练场景下，支持Ring Attention长序列并行，对超长序列场景进行加速。

## 使用方式
### 1. Megatron代码修改
1. 在 Megatron-LM 目录下修改`pretrain_gpt.py`文件中的`get_batch`函数。
    ```diff
    def get_batch(data_iterator):
        """Generate a batch."""

    -   # TODO: this is pretty hacky, find a better way
    -   if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
    -       return None, None, None, None, None

        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank(data_iterator)

        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)

    +   # TODO: this is pretty hacky, find a better way
    +   if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
    +       return None, None, None, None, None

        return batch.values()
    ```

2. 在 Megatron-LM 目录下修改`pretrain_gpt.py`文件中的`get_batch`函数。

    ```diff
    def is_dataset_built_on_rank():
    -   return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0
    +   return mpu.get_tensor_model_parallel_rank() == 0
    ```

### 2. 数据准备
首先确保每一个文档的末尾都添加了EOD Token。


### 3. 参数设置
#### 不启用长序列并行（CP）
打开 `--reset-attention-mask`和`--reset-position-ids`选项
#### 启用长序列并行
首先确保`--context-parallel-size`大于`1`。

设置`--context-parallel-algo=megatron_cp_algo`

打开`--reset-attention-mask`和`--reset-position-ids`选项，`--cp-attention-mask-type`为`general`。