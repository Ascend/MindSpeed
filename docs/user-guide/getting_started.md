# 快速上手

按照如下步骤操作，即可实现Megatron-LM在昇腾设备上的高效运行，且无缝集成并充分发挥MindSpeed所提供的丰富加速与优化技术。

### 1. 在Megatron-LM中导入MindSpeed适配器

在”Megatron-LM“目录下修改**pretrain_gpt.py**文件，在“import torch”下新增一行“import mindspeed.megatron_adaptor”代码，即如下修改：

  ```Python
    import torch
    import mindspeed.megatron_adaptor # 新增代码行
    from functools import partial
    from contextlib import nullcontext
    import inspect
  ```


### 2. 数据准备，参考[Megatron-LM官方文档](https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#datasets)准备训练数据

a. 下载[Tokenizer](https://huggingface.co/Xenova/gpt-3.5-turbo/tree/main)。

新建“Megatron-LM/gpt-tokenizer”目录，并将vocab.json和merges.txt文件下载至该目录。

b. 下载数据集。

以[Alpaca数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)为例。

> 注：用户需要自行设置代理，以便访问或下载数据集。

### 3. 配置环境变量

当前以root用户安装后的默认路径为例，请用户根据set_env.sh的实际路径执行如下命令。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 4. 数据处理

a. 语料格式转换。

数据处理依赖于多个第三方库，请确保已正确安装如下依赖：

```shell
pip3 install nltk pyarrow pandas
```

以下代码段展示了如何读取Parquet格式的原始语料，并将其转换为JSON格式，以便后续处理。

```python
import json
import pandas as pd

data_df = pd.read_parquet("train-00000-of-00001-a09b74b3ef9c3b56.parquet")
data_df['text'] = data_df['text'].apply(lambda v: json.dumps({"text": v}))
with open("alpaca_json.json", encoding='utf-8', mode='w') as f:
    for i, row in data_df.iterrows():
        f.write(row['text'])
        f.write('\n')
```

b. 预训练数据集生成。

若在昇腾设备上使用preprocess_data.py脚本处理数据，须在“Megatron-LM”目录下修改“tools/preprocess_data.py”脚本，在“import torch”下新增一行“import mindspeed.megatron_adaptor”代码。

```python
import torch
import mindspeed.megatron_adaptor
import numpy as np
```

新建“Megatron-LM/gpt_pretrain_data”目录，通过运行preprocess_data.py脚本，可以将转换后的JSON文件进一步处理为适合Megatron-LM预训练的二进制格式。

```python
python tools/preprocess_data.py \
   --input alpaca_json.json \
   --output-prefix ./gpt_pretrain_data/alpaca \
   --tokenizer-type GPT2BPETokenizer \
   --vocab-file ./gpt-tokenizer/vocab.json \
   --merge-file ./gpt-tokenizer/merges.txt \
   --append-eod \
   --log-interval 1000 \
   --workers 8
```

执行成功后，将在gpt_pretrain_data目录下生成两个文件：alpaca_text_document.bin和alpaca_text_document.idx，代表预处理完成的预训练数据集。

### 5. 准备预训练脚本

在“Megatron-LM”目录下准备预训练脚本train_distributed.sh，脚本示例如下：
```shell
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
CKPT_DIR=./ckpt
VOCAB_FILE=<Specify path to file>/vocab.json
MERGE_FILE=<Specify path to file>/merges.txt
DATA_PATH=<Specify path and file prefix>_text_document
TP=2
PP=2
CP=1
EP=1
DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
GPT_ARGS="
    --transformer-impl local \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 1 \
    --num-layers 8 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 1000 \
    --init-method-std 0.01 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --initial-loss-scale 4096.0 \
    --disable-bias-linear \
    --lr-warmup-fraction 0.01 \
    --fp16
"
DATA_ARGS="
    --split 990,5,5
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
"
OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
set +x

```

### 6. 配置路径

请编辑示例脚本train_distributed.sh，并设置如下环境变量以指定必要的路径：
```shell
CKPT_DIR=./ckpt
VOCAB_FILE=./gpt-tokenizer/vocab.json
MERGE_FILE=./gpt-tokenizer/merges.txt
DATA_PATH=./gpt_pretrain_data/alpaca_text_document
```
> 注：上述路径需根据您的实际情况进行适当调整。

### 7. 运行脚本启动预训练

```shell
bash ./train_distributed.sh
```

> 注：示例脚本train_distributed.sh中的部分参数（如--hidden-size、--num-layers）需根据实际场景进行适配，避免OOM等现象。
