# DeepSeek-V3

## 训练

DeepSeek-V3 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### 脚本

1. 按照readme安装MindSpeed-LLM、MindSpeed和Megatron-LM

   ```shell
    # 安装MindSpeed加速库
    git clone https://gitcode.com/Ascend/MindSpeed.git
    # 准备MindSpeed-LLM及Megatron-LM源码
    git clone -b master https://gitcode.com/Ascend/MindSpeed-LLM.git 
    git clone -b core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git  # megatron从github下载，请确保网络能访问 
    mkdir model_from_hf
    mkdir dataset
    mkdir ckpt
    cd MindSpeed-LLM
    cp -r ../Megatron-LM/megatron ./
   ```
2. 搭建环境

   ```bash
   # python3.10
   conda create -n test python=3.10
   conda activate test

   # 安装 torch 和 torch_npu
   pip install torch-2.7.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl 
   pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

   # 修改 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
3. 下载 DeepSeek-V3 的 [预训练权重和词表](https://hf-mirror.com/deepseek-ai/DeepSeek-V3/tree/main)

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

4. 预训练

   4.1 准备数据集

   下载 DeepSeek-V3 [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
     # 下载数据
     cd ./dataset
     wget https://hf-mirror.com/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
     cd ..
     # 处理数据   
     mkdir ./dataset/deepseek3-hf/
     # 修改 ascend-toolkit 路径
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     # MindSpeed-LLM目录下执行
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

   4.2 预训练
   配置DeepSeek-V3 预训练脚本: examples/pretrain_deepseek_v3_ptd_dualpipev.sh

   ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、环境变量保存路径
    source "../MindSpeed/tests_extend/system_tests/env_npu.sh"
    CKPT_SAVE_DIR="./ckpt/"
    DATA_PATH="./dataset/deepseek3-hf/alpaca_text_document"  #数据集路径
    TOKENIZER_PATH="./model_from_hf/deepseek3-hf/"  #词表路径
   ```
   
   启动 DeepSeek-V3 预训练脚本: examples/pretrain_deepseek_v3_ptd_dualpipev.sh

   ```shell
    bash examples/pretrain_deepseek_v3_ptd_dualpipev.sh
   ```