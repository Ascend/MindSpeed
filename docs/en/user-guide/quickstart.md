# Quick Start

This guide walks you through installation and then helps you run Megatron-LM efficiently on Ascend devices while seamlessly integrating and fully leveraging the rich acceleration and optimization technologies provided by MindSpeed.

## Environment Preparation

1. First, refer to the [MindSpeed Installation Guide](install_guide.md) to prepare the environment.

2. Import the MindSpeed adapter into Megatron-LM.

    In the `Megatron-LM` directory, modify the `pretrain_gpt.py` file and add `import mindspeed.megatron_adaptor` below `import torch`, as shown below:

    ```Python
    import torch
    import mindspeed.megatron_adaptor # Add this line
    from functools import partial
    from contextlib import nullcontext
    import inspect
    ```

## Data Preparation

Refer to the [official Megatron-LM documentation](https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#datasets) to prepare the training data.

1. Download the [Tokenizer](https://huggingface.co/Xenova/gpt-3.5-turbo/tree/main).

    Create the `Megatron-LM/gpt-tokenizer` directory and download the `vocab.json` and `merges.txt` files into this directory.

2. Download the dataset. The [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) is used as an example.

    >[!NOTE]
    >You need to configure a proxy yourself so that you can access or download the dataset.

3. Convert the corpus format.

    Data processing depends on multiple third-party libraries. Ensure that the following dependencies are installed:

    ```shell
    pip3 install nltk pyarrow pandas
    ```

    The following code snippet shows how to read the source corpus in the Parquet format and convert it to the JSON format for subsequent processing.

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

4. Generate the pretraining dataset.

    If you use the `preprocess_data.py` script to process data on an Ascend device, modify the `tools/preprocess_data.py` script in the `Megatron-LM` directory and add `import mindspeed.megatron_adaptor` below `import torch`.

    ```python
    import torch
    import mindspeed.megatron_adaptor
    import numpy as np
    ```

    Create the `Megatron-LM/gpt_pretrain_data` directory. By running the `preprocess_data.py` script, you can further process the converted JSON file into a binary format suitable for Megatron-LM pretraining.

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

    **Table 1** preprocess_data.py parameters

    |Parameter|Description|
    |-|-|
    |--input|Dataset|
    |--output-prefix|Processed dataset|
    |--tokenizer-type|Tokenizer type|
    |--vocab-file|Tokenizer file|
    |--merge-file|Tokenizer file|
    |--append-eod|Adding the `<eod>` end token|
    |--log-interval|Logging interval|
    |--workers|Number of parallel workers|

    After the command completes successfully, two files are generated in the `gpt_pretrain_data` directory: `alpaca_text_document.bin` and `alpaca_text_document.idx`. They represent the preprocessed pretraining dataset.

## Starting Pretraining

1. Configure environment variables.

    The following example uses the default path after installation as root. Run the following command according to the actual path to `set_env.sh`.

    ```shell
    source /usr/local/Ascend/cann/set_env.sh
    ```

2. Prepare the pretraining script.

    Prepare the pretraining script `train_distributed.sh` in the `Megatron-LM` directory. A sample script is shown below:

    ```bash
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

3. Configure paths.

    Edit the sample `train_distributed.sh` script and set it as follows:

    ```bash
    CKPT_DIR=./ckpt
    VOCAB_FILE=./gpt-tokenizer/vocab.json
    MERGE_FILE=./gpt-tokenizer/merges.txt
    DATA_PATH=./gpt_pretrain_data/alpaca_text_document
    ```

    **Table 2** train_distributed.sh parameters

    |Parameter|Description|
    |-|-|
    |CKPT_DIR|Path to the weights file|
    |VOCAB_FILE|Tokenizer file|
    |MERGE_FILE|Tokenizer file|
    |DATA_PATH|Dataset file|

    Adjust the paths as needed for your actual environment.

4. Run the script to start pretraining.

    ```bash
    bash ./train_distributed.sh
    ```

    > [!NOTE]
    > Some parameters in the sample `train_distributed.sh` script, such as `--hidden-size` and `--num-layers`, need to be adapted to the actual scenario to avoid out of memory (OOM) and similar issues.
