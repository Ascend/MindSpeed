# Model Training System Test

## Mixtral-8x7B Training Test

### Hardware Requirements

Minimum hardware configuration for training:

| Hardware |       Configuration       |
| :------: | :-----------------------: |
|   NPU    | 16 x Ascend NPUs |

### Preparation

1. Complete the source code preparation and setup for MindSpeed and Megatron-LM as described in the README file in the MindSpeed root directory.

2. Download the [vocabulary and tokenizer](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main) for Mixtral-8x7B.

Execute the following operations in the Megatron-LM directory:

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

### Model Training

#### 1. Prepare the dataset

Download the Mixtral-8x7B [dataset](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet).

```shell
# Download data
cd ./dataset
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
# Process data
mkdir ./dataset/Mixtral-8x7B/
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/Mixtral-8x7B/ \
    --output-prefix ./dataset/Mixtral-8x7B/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF
```

#### 2. Configure the Mixtral-8x7B pre-training script: ***pretrain_mixtral.sh***

```bash
# Copy the script used for mixtral training to the Megatron-LM directory
cp ../MindSpeed/tests_extend/system_tests/mixtral/pretrain_mixtral.sh .

```

```shell
# Modify the pretrain_mixtral.sh test script file according to the following content
# Set the ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Configure the vocabulary, dataset, and model parameter save paths according to the actual situation
DATA_PATH="./dataset/Mixtral-8x7B/alpaca_text_document"
TOKENIZER_MODEL="./model_from_hf/Mixtral-8x7B/"
CKPT_SAVE_DIR="./ckpt/Mixtral-8x7B/"

# Configure the distributed parameters according to the actual situation of the distributed cluster
NPUS_PER_NODE=8
MASTER_ADDR="your master node IP"
MASTER_PORT=6000
NNODES=2
NODE_RANK="current node id"
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

# Set the training parallelism strategy based on actual needs
TP=2
PP=4
EP=2
```

#### 3. Enable deterministic computation

Add `export HCCL_DETERMINISTIC=TRUE` to the pretrain_mixtral.sh script

Additionally, add the following code to pretrain_gpt.py

```python
# For ptdbg_ascend, see https://gitcode.com/Ascend/tools/blob/master/ptdbg_ascend/README.md
from ptdbg_ascend import seed_all
seed_all(mode=True)
```

#### 4. Start the Mixtral-8x7B pre-training script: ***pretrain_mixtral.sh***

```shell
bash pretrain_mixtral.sh
```

**NOTE** If using multi-node training, you need to set up multi-node data sharing so that non-master nodes can read data from the master node through data sharing. Alternatively, directly copy the data generated on the master node to the non-master nodes. For multi-node training, prepare the environment on each machine following the steps above, and start the training tasks simultaneously.
