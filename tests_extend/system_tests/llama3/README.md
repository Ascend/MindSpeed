# Llama3-8B

## 训练

Llama3-8B 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### 脚本

1. 按照readme安装MindSpeed和Megatron-LM

   ```shell
   git clone https://gitee.com/ascend/MindSpeed.git
   pip install -e MindSpeed
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   # git checkout 到使用的Megatron-LM分支
   git checkout core_r0.8.0
   mindspeed -P
   mkdir model_from_hf
   mkdir dataset
   mkdir ckpt
   mv ../MindSpeed/tests_extend/tools/preprocess_data.py .
   mv ../MindSpeed/tests_extend/tools/data_handler.py .
   mv ../MindSpeed/tests_extend/system_tests/llama3/pretrain_llama3_8b_ptd.sh ./examples/
   ```
2. 搭建环境

   ```bash
   # python3.8
   conda create -n test python=3.8
   conda activate test

   # 安装 torch 和 torch_npu
   pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
   pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
   pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

   # 修改 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
3. 下载 Llama3-8B 的 [预训练权重和词表](https://hf-mirror.com/unsloth/llama-3-8b/tree/main)

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

4. 预训练

   4.1 准备数据集

   下载 LLaMA3-8B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
     # 下载数据
     cd ./dataset
     wget https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
     cd ..
     # 处理数据   
     mkdir ./dataset/llama-3-8b-hf/
     # 修改 ascend-toolkit 路径
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     python ./preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/llama-3-8b-hf/ \
       --output-prefix ./dataset/llama-3-8b-hf/alpaca \
       --workers 4 \
       --log-interval 1000 \
       --tokenizer-type PretrainedFromHF
   ```

   4.2 预训练
   配置llama3-8B 预训练脚本: examples/pretrain_llama3_8b_ptd.sh

   ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、环境变量保存路径
    source "../MindSpeed/tests_extend/system_tests/env_npu.sh"
    CKPT_SAVE_DIR="./ckpt/"
    DATA_PATH="./dataset/llama-3-8b-hf/alpaca_text_document"  #数据集路径
    TOKENIZER_MODEL="./model_from_hf/llama-3-8b-hf/"  #词表路径
   ```
   
   启动 LLaMA3-8B 预训练脚本: examples/pretrain_llama3_8b_ptd.sh

   ```shell
    bash examples/pretrain_llama3_8b_ptd.sh
   ```

### 性能

#### 吞吐

LLaMA3-8B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备 |   模型   | 迭代数 | tokens吞吐 (tokens/s/p) |
| :--: | :-------: | :----: | :---------------------: |
| NPUs | LLaMA3-8B |  1000  |        2474         |
| 参考 | LLaMA3-8B |  1000  |        2665          |
