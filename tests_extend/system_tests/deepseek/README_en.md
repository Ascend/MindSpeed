# DeepSeek-V3

## Training

Hardware configuration for DeepSeek-V3 training:

| Hardware | Configuration |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### Script

1. Install MindSpeed-LLM, MindSpeed, and Megatron-LM according to the README file.

   ```shell
    # Install the MindSpeed acceleration library
    git clone https://gitcode.com/Ascend/MindSpeed.git
    # Prepare the source code for MindSpeed-LLM and Megatron-LM
    git clone -b master https://gitcode.com/Ascend/MindSpeed-LLM.git
    git clone -b core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git  # megatron is downloaded from github, ensure network access is available github, ensure network access is available
    mkdir model_from_hf
    mkdir dataset
    mkdir ckpt
    cd MindSpeed-LLM
    cp -r ../Megatron-LM/megatron ./
   ```

2. Set up the environment.

   ```bash
   # python3.10
   conda create -n test python=3.10
   conda activate test

   # Install torch and torch_npu
   pip install torch-2.7.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
   pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

   # Modify the ascend-toolkit path
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

3. Download the [pretrained weights and tokenizer](https://hf-mirror.com/deepseek-ai/DeepSeek-V3/tree/main) for DeepSeek-V3.

   ```shell
     #!/bin/bash
     mkdir ./model_from_hf/deepseek3-hf/
     cd ./model_from_hf/deepseek3-hf/
     wget https://hf-mirror.com/deepseek-ai/DeepSeek-V3/resolve/main/config.json
     wget https://hf-mirror.com/deepseek-ai/DeepSeek-V3/resolve/main/configuration_deepseek.py
     wget https://hf-mirror.com/deepseek-ai/DeepSeek-V3/resolve/main/model-00001-of-000163.safetensors
     wget https://hf-mirror.com/deepseek-ai/DeepSeek-V3/resolve/main/model-00002-of-000163.safetensors
     ...
     wget https://hf-mirror.com/deepseek-ai/DeepSeek-V3/resolve/main/model-00162-of-000163.safetensors
     wget https://hf-mirror.com/deepseek-ai/DeepSeek-V3/resolve/main/model-00163-of-000163.safetensors
     wget https://hf-mirror.com/deepseek-ai/DeepSeek-V3/resolve/main/model.safetensors.index.json
     wget https://hf-mirror.com/deepseek-ai/DeepSeek-V3/resolve/main/modeling_deepseek.py
     wget https://hf-mirror.com/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json
     wget https://hf-mirror.com/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer_config.json
     cd ../../
   ```

4. Perform pre-training.

   4.1 Prepare the dataset.

   Download the DeepSeek-V3 [dataset](https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
     # Download data
     cd ./dataset
     wget https://hf-mirror.com/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
     cd ..
     # Process data
     mkdir ./dataset/deepseek3-hf/
     # Modify the ascend-toolkit path
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     # Execute in the MindSpeed-LLM directory
     python ./preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/deepseek3-hf/ \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralPretrainHandler \
        --output-prefix ./dataset/alpaca \
        --json-keys text \
        --workers 4 \
        --log-interval 1000 \
        --append-eod
   ```

   4.2 Start pre-training.
   Configure the DeepSeek-V3 pre-training script: `examples/pretrain_deepseek_v3_ptd_dualpipev.sh`

   ```shell
    # Set the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # Configure the tokenizer, dataset, and environment variable save paths according to the actual situation
    source "../MindSpeed/tests_extend/system_tests/env_npu.sh"
    CKPT_SAVE_DIR="./ckpt/"
    DATA_PATH="./dataset/deepseek3-hf/alpaca_text_document"  # Dataset path
    TOKENIZER_PATH="./model_from_hf/deepseek3-hf/"  # Tokenizer path
   ```

   Start the DeepSeek-V3 pre-training script: `examples/pretrain_deepseek_v3_ptd_dualpipev.sh`

   ```shell
    bash examples/pretrain_deepseek_v3_ptd_dualpipev.sh
   ```
