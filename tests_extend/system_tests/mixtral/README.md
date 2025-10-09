# 模型训练系统测试

## Mixtral-8x7B训练测试

### 硬件要求

训练的最低硬件配置:

| 硬件 |       配置       |
| :--: | :--------------: |
| NPU | 16 x Ascend NPUs |

### 准备工作

1. 按照MindSpeed根目录下README完成MindSpeed、Megatron-LM源码准备和上手准备

2. 下载 Mixtral-8x7B 的 [词表和tokenizer](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main)

在Megatron-LM目录下执行如下操作：

```shell
#!/bin/bash
mkdir logs
mkdir model_from_hf
mkdir dataset
mkdir ckpt
cd ./model_from_hf/
git lfs install
git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
mv Mixtral-8x7B-v0.1 Mixtral-8x7B
cd ..
```

### 模型训练

1. 准备数据集

下载 Mixtral-8x7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

```shell
# 下载数据
cd ./dataset
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
# 处理数据   
mkdir ./dataset/Mixtral-8x7B/
python ./tests_extend/tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/Mixtral-8x7B/ \
    --output-prefix ./dataset/Mixtral-8x7B/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF
```

2. 配置 Mixtral-8x7B 预训练脚本：***pretrain_mixtral.sh***

```
# 拷贝mixtral训练所用脚本到Megatron-LM目录下
cp ../MindSpeed/tests_extend/system_tests/mixtral/pretrain_mixtral.sh .

```

```shell
# 按照如下内容修改pretrain_mixtral.sh测试脚本文件
# 设置 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 根据实际情况配置词表、数据集、模型参数保存路径
DATA_PATH="./dataset/Mixtral-8x7B/alpaca_text_document"
TOKENIZER_MODEL="./model_from_hf/Mixtral-8x7B/"
CKPT_SAVE_DIR="./ckpt/Mixtral-8x7B/"

# 根据分布式集群实际情况配置分布式参数
NPUS_PER_NODE=8
MASTER_ADDR="your master node IP"
MASTER_PORT=6000
NNODES=2
NODE_RANK="current node id"
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

# 根据实际需要设置训练并行策略
TP=2
PP=4
EP=2
```

3. 开启确定性计算

在pretrain_mixtral.sh脚本添加 `export HCCL_DETERMINISTIC=TRUE`

另外，在pretrain_gpt.py中添加代码
```
# ptdbg_ascend 参见 https://gitcode.com/Ascend/tools/blob/master/ptdbg_ascend/README.md
from ptdbg_ascend import seed_all
seed_all(mode=True)
```


4. 启动 Mixtral-8x7B 预训练脚本: ***pretrain_mixtral.sh***

```shell
bash pretrain_mixtral.sh
```

**注意**：如果使用多机训练，需要设置多机数据共享，非主节点通过数据共享读取主节点数据。或者，直接将主节点生成的数据复制到非主节点。多机训练时，在每台机器按照如上步骤准备环境，同时启动训练任务。
