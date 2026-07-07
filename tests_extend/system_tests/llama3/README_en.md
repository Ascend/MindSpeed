# Llama3-8B

## Training

Hardware configuration for Llama3-8B training:

| Hardware | Configuration |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### Scripts

1. Install MindSpeed and Megatron-LM according to the README file.

   ```shell
   git clone https://gitcode.com/Ascend/MindSpeed.git
   pip install -e MindSpeed
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   # Git checkout to the Megatron-LM branch to use
   git checkout core_r0.8.0
   mindspeed -P
   mkdir model_from_hf
   mkdir dataset
   mkdir ckpt
   mv ../MindSpeed/tools/preprocess_data.py .
   mv ../MindSpeed/tools/data_handler.py .
   mv ../MindSpeed/tests_extend/system_tests/llama3/pretrain_llama3_8b_ptd.sh ./examples/
   ```

2. Set up the environment.

   ```bash
   # python3.8
   conda create -n test python=3.8
   conda activate test

   # Install torch and torch_npu
   pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
   pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
   pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

   # Modify the ascend-toolkit path
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

3. Download the [pre-trained weights and tokenizer](https://hf-mirror.com/unsloth/llama-3-8b/tree/main) for Llama3-8B.

   ```shell
     #!/bin/bash
     mkdir ./model_from_hf/llama-3-8b-hf/
     cd ./model_from_hf/llama-3-8b-hf/
     wget https://hf-mirror.com/unsloth/llama-3-8b/raw/main/config.json
     wget https://hf-mirror.com/unsloth/llama-3-8b/raw/main/generation_config.json
     wget https://hf-mirror.com/unsloth/llama-3-8b/raw/main/model-00001-of-00004.safetensors
     wget https://hf-mirror.com/unsloth/llama-3-8b/raw/main/model-00002-of-00004.safetensors
     wget https://hf-mirror.com/unsloth/llama-3-8b/raw/main/model-00003-of-00004.safetensors
     wget https://hf-mirror.com/unsloth/llama-3-8b/raw/main/model-00004-of-00004.safetensors
     wget https://hf-mirror.com/unsloth/llama-3-8b/raw/main/model.safetensors.index.json
     wget https://hf-mirror.com/unsloth/llama-3-8b/raw/main/special_tokens_map.json
     wget https://hf-mirror.com/unsloth/llama-3-8b/raw/main/tokenizer.json
     wget https://hf-mirror.com/unsloth/llama-3-8b/raw/main/tokenizer_config.json
     cd ../../
   ```

4. Perform pre-training.

   4.1 Prepare the dataset.

   Download the LLaMA3-8B [dataset](https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet).

   ```shell
     # Download the data
     cd ./dataset
     wget https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
     cd ..
     # Process data
     mkdir ./dataset/llama-3-8b-hf/
     # Modify ascend-toolkit path
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     python ./preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/llama-3-8b-hf/ \
       --output-prefix ./dataset/llama-3-8b-hf/alpaca \
       --workers 4 \
       --log-interval 1000 \
       --tokenizer-type PretrainedFromHF
   ```

   4.2 Start pre-training.
   Configure the llama3-8B pretraining script: `examples/pretrain_llama3_8b_ptd.sh`.

   ```shell
    # Set ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # Configure the vocabulary, dataset, and environment variable save paths based on actual conditions
    source "../MindSpeed/tests_extend/system_tests/env_npu.sh"
    CKPT_SAVE_DIR="./ckpt/"
    DATA_PATH="./dataset/llama-3-8b-hf/alpaca_text_document"  #dataset path
    TOKENIZER_MODEL="./model_from_hf/llama-3-8b-hf/"  #tokenizer path
   ```

   Start the LLaMA3-8B pretraining script: `examples/pretrain_llama3_8b_ptd.sh`.

   ```shell
    bash examples/pretrain_llama3_8b_ptd.sh
   ```

### Performance

#### Throughput

Performance comparison of LLaMA3-8B on **Ascend chips** and **reference chips**:

| Device | Model | Iteration Count | Token Throughput (tokens/s/p) |
| :--: | :-------: | :----: | :---------------------: |
| NPUs | LLaMA3-8B | 1000 | 2474 |
| Reference | LLaMA3-8B | 1000 | 2665 |
