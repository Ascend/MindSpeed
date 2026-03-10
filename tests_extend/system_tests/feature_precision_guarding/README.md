# 特性精度看护(Feature Precision Guarding)系统用例执行和添加说明

本文档介绍MindSpeed+Megatron特性的精度看护用例执行和构造

## 特性的精度看护用例执行

以llama模型的精度看护为例

### 0.环境搭建

参考MindSpeed环境安装相关内容

### 1.准备tokenizer和数据集

下载llama2模型的tokenizer

```shell
mkdir ./model_from_hf/llama-2-7b-hf/
cd ./model_from_hf/llama-2-7b-hf/
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.model
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer_config.json
cd ../../
```

下载llama2的数据集并且进行处理

```shell
# 下载数据
cd ./dataset
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
# 处理数据   
mkdir ./dataset/llama-2-7b-hf/
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./dataset/llama-2-7b-hf/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF
```

### 2.准备初始参数

按TP1PP1准备一份参数，并保存在./ckpt_llama目录下（可以随机初始化，然后保存参数）

### 3.配置指定参数

结合训练脚本pretrain_fpg_llama.sh的配置，在fpg_llama_usecase.yaml里配置指定参数，如下所示：

```yaml
spec:
  data_path: /home/dataset/llama2/alpaca_text_document
  tokenizer_model: /home/dataset/model/llama-2-7b-hf/tokenizer.model
  mbs: 2 # MBS   micro-batch-size
  gbs: 16 # GBS  global-batch-size
  train_iters: 5000
```

### 4.配置基线参数和特性参数

参考fgp_llama_usecase.yaml配置，每个系统用例都包含pre_process和run和run步骤, 每个步骤可以配置多个处理实例，每个处理实例包含script_file和param两个配置；
**例如baseline的两个步骤**

- pre_process : 准备基线的初始化参数
- run : 执行脚本，获得loss基线

各特性的配置和baseline的配置规则一致，执行逻辑参见pretrain_gpt_fpg.py
**执行说明**

- 入口文件pretrain_gpt_fpg.py， python MindSpeed/tests_extend/system_tests/feature_precision_guarding/pretrain_gpt_fpg.py执行
- 日志存放在 ./{%YU_%m_%d}logs，日志文件命名规则：{feature_name}-{stage_name}-{state_index}-{timestamp}.log
- 执行完run实例后会自动和基线对比loss，并将结果写入report.csv文件中

## 特性的精度看护用例添加

### 方式一： 用例配置文件中直接添加

举例说明：验证MC2+分布式优化器的精度
**分析**

- MC2需要设置参数--use-ascend-mc2，MC2依赖TP和序列并行 (SP基础脚本已配置)
- 分布式优化器需要添加额外参数 --use-distributed-optimizer
在fpg_llama_usecase.yaml的features下添加一条即可，如下所示：

```yaml
  - mc2_distributed-optim: 
      pre_process:
        - { script_file: llama_param_cvt.sh,
            param: {
              py_script_path: MindSpeed/tests_extend/system_tests/feature_precision_guarding/llama_param_cvt.py,
              input_model_dir: ./ckpt_llama,
              output_model_dir: ./ckpt_llama_tp2pp2,
              tp_size: 2,
              pp_size: 2,
              num_layers: 12,
              num_heads: 8,
              hidden_size: 512
            }
        }
      run:
        - { script_file: pretrain_fpg_llama.sh,
            param: { npu_per_node: 8,
                     tp_size: 2,
                     pp_size: 2,
                     load_ckpt_dir: ./ckpt_llama_tp2pp2,
                     extra_args: '"--use-distributed-optimizer"',
                     ascend_mc2: 1
            }
        }
```

其中num_per_node是每个节点用的卡数，必填
tp_size、pp_size必填，已验证TP1PP1的5kstep的loss和TP2PP2平均绝对/相对误差均<0.5%,可以以TP1PP1为基线，也可以修改baseline中的TP和PP设置；

### 方式二： 新实例化用例文件

按fpg_llama_usecase.yaml的格式新创建一个用例文件，例如fpg_qwen_usecase.yaml
然后在pretrain_gpt_fpg.py的main方法里添加一行即可，例如：

```python
    xtest_pretrain_fpg("fpg_qwen_usecase.yaml")
```
