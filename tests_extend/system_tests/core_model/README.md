# core_model 系统用例代码覆盖运行说明
本文档介绍pretrain_gpt.py相关代码的覆盖用例执行和构造

## 环境搭建
需要安装MindSpeed+Megatron, 见MindSpeed首页[安装](https://gitee.com/ascend/MindSpeed)

**注意**：
- 将MindSpeed放在Megatron-LM目录下（方便代码覆盖统计）
- 安装CANN-NNAL（计算通信并行COC用例需要）

## 代码覆盖统计执行

### 1. 准备数据集并配置
下载tokenizer，[下载链接](https://huggingface.co/Xenova/gpt-3.5-turbo/tree/main) ，下载vocab.json和merges.txt即可
下载dataset, [下载链接]（https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet）

Megatron语料格式处理：
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
Megatron预训练数据集生成
```bash
python tools/preprocess_data.py --input alpaca_json.json \
                                --output-prefix /gpt_pretrain_data/alpaca \
                                --tokenizer-type GPT2BPETokenizer \
                                --vocab-file /gpt-tokenizer/vocab.json \
                                --merge-file /gpt-tokenizer/merges.txt \
                                --append-eod \
                                --log-interval 1000 \
                                --workers 8
```
执行成功在/gpt_pretrain_data目录可见两个文件: alpaca_text_document.bin和alpaca_text_document.idx

### 2.配置测试数据集和tokenizer

进入MindSpeed/tests_extend/system_tests/core_model目录
在gpt_usecase.yaml配置文件中的spec(指定参数模块)配置data_path、vocab_file、merge_file, 例如：
```yaml
spec:
  data_path: /home/dataset/gpt-3.5/alpaca_text_document
  vocab_file: /home/dataset/model/gpt-3.5/vocab.json
  merge_file: /home/dataset/model/gpt-3.5/merges.txt
```

### 3. 安装代码覆盖统计工具及使用
安装代码覆盖统计工具： 
```bash
pip install coverage 
```
在Megatron-LM/pretrain_gpt.py文件添加统计代码
**注意**
- 经测试当前只支持在代码文件中添加代码的方式，不支持命令行方式（怀疑受分布式影响）
- 一定要设置data_suffix，否则统计文件会覆盖，推荐按下文写法
在pretrain_gpt.py的文件头添加，如下所示
``` python
import random
import time

import coverage

cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")
cov.start()
```
在pretrain(...)方法后添加cov.stop()和cov.save()，如下所示
```python 
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    cov.stop()
    cov.save()
```

### 4. 执行用例，生成代码覆盖统计文件
进入Megatron-LM目录，配置.coveragerc文件，例如
```python
[run]
branch = True
source = ./
omit = setup.py
       pretrain_bert.py
       pretrain_ict.py
       tools/*
       tests/*
       <其他忽略、不参与统计的文件>
[report]
show_missing = True
```
批量执行用例
``` python
python MindSpeed/tests_extend/system_tests/core_model/pretrain_gpt_usecase.py
``` 
执行完成后在目录下会生成各用例的覆盖文件 .coverage.usecase-*

### 5. 生成代码覆盖统计报告

先合并所有用例的覆盖统计文件，执行以下命令：
```shell 
coverage combine .coverage.usecase-*
```
所有的统计文件会合并成一个文件: .coverage，基于该文件生成报告：
```shell 
coverage html --data-file=.coverage -d <输出目录>
```
浏览器打开报告中的index.html文件，可看到总代码覆盖率及各文件的代码覆盖率，点看具体文件可看到覆盖的代码行（绿底标示），及未覆盖的代码行（红底标示）

## 用例补充
### 方式一：用例配置文件中直接添加
举例说明：添加 内存碎片优化 + FA的用例
**分析**
- 内存碎片优化只需要设置环境变量：MEMORY_FRAGMENTATION=1
- FA只需要添加参数：--use-flash-attn
添加样例只需要在在gpt-usecase.yaml文件的products下添加一条：
\- { use_mcore: [ True, False ], tp_size: [ 2 ], pp_size: [ 2 ], memory_fragmentation: [ 1 ], extra_args: [ '"--use-flash-attn"'] }
**注意**
- 训练的其他参数见pretrain_gpt_usecase.sh，所有用例执行的入口文件是pretrain_gpt_usecase.py，详细信息看这两个文件
- use-mcore、tp_size、pp_size是必填项，其他可选，所有的value是 list格式
- use-mcore=True走的megatron/core代码分支，use-mcore=False走的megatron/legacy分支
- 参数key大小写不敏感，脚本都会转成大写，设置的环境变量都定义为key，额外参数放在extra_args里
- 实际添加的系统用例个数=各key取值个数的乘积

### 方式二：新实例化用例模板
看完方式一，如果其中固化的参数不满足要求，比如用train-samples替代train-iters，可以新创建pretrain_gpt_usecese_template2.sh和gpt_usecase_template2.yaml
**建议**
- 新实例化用例模板遵从方式一的模式
- 在pretrain_gpt_usecase.py里添加一个用例即可，不要新添入口