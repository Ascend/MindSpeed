# 简介

MindSpeed 是针对华为昇腾设备的大模型加速库。

大模型训练是一种非常复杂的过程，涉及到许多技术和挑战，其中大模型训练需要大量的显存资源是一个难题，对计算卡提出了不小的挑战。
为了在单个计算卡显存资源不足时，可以通过多张计算卡进行计算，业界出现了类似 Megatron、DeepSpeed 等第三方大模型加速库，对模型、输入数据等进行切分并分配到不同的计算卡上，最后再通过集合通信对结果进行汇总。

昇腾提供 MindSpeed 加速库，使能客户大模型业务快速迁移至昇腾设备，并且支持昇腾专有算法，确保开箱可用。

# 安装

### 1. 安装依赖

请安装最新昇腾软件栈：[https://www.hiascend.com/zh/](https://www.hiascend.com/zh/)

| 依赖软件      |
|-----------|
| Driver    | 
| Firmware  | 
| CANN      | 
| Kernel    | 
| PyTorch   | 
| torch_npu | 
| apex      | 

### 2. 安装 MindSpeed

下载源码安装:

 ```shell
 git clone -b 1.0.RC1 https://gitee.com/ascend/MindSpeed.git
 cd MindSpeed
 pip install -e .
 ```

### 3. 获取 Megatron-LM 并指定 commit id

 ```shell
 git clone https://github.com/NVIDIA/Megatron-LM.git
 cd Megatron-LM
 git checkout bcce6f54e075e3c3374ea67adefe54f3f2da2b07
 ```

# 快速上手

1. 在 Megatron-LM 目录下修改`pretrain_gpt.py`文件，在`import torch`下新增一行`import mindspeed.megatron_adaptor`

    ```diff
     import os
     import torch
    +import mindspeed.megatron_adaptor
     from torch import Tensor
     from functools import partial
     from typing import Union
    ```

2. 在 Megatron-LM 目录下修改`pretrain_gpt.py`
   文件，在model_provider函数中删除`assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"`
   。
    ```diff
    else:
    -   assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"

        model = megatron.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )
    ```

3. 在 Megatron-LM 目录下，准备好训练数据，并在示例脚本中填写对应路径，然后执行。
    ```shell
    bash examples/pretrain_gpt_distributed.sh
    ```

# 特性介绍

| 特性 | 介绍 |
| ----- | ----- |
| Megatron 数据并行 | [link](https://github.com/NVIDIA/Megatron-LM) |
| Megatron 张量并行 | [link](https://github.com/NVIDIA/Megatron-LM) |
| Megatron 流水并行  | [link](https://github.com/NVIDIA/Megatron-LM) |
| Megatron 虚拟流水并行  | [link](https://github.com/NVIDIA/Megatron-LM) |
| Megatron 序列并行  | [link](https://github.com/NVIDIA/Megatron-LM) |
| Megatron 重计算  | [link](https://github.com/NVIDIA/Megatron-LM) |
| Megatron 分布式优化器  | [link](https://github.com/NVIDIA/Megatron-LM) |
| Megatron 异步DDP  | [link](https://github.com/NVIDIA/Megatron-LM) |
| Ascend TP 重计算通信优化 | [link](docs/features/recomputation-communication.md) |
| Ascend 内存碎片优化 | [link](docs/features/memory-fragmentation.md) |
| Ascend 自适应选择重计算 | [link](docs/features/adaptive-recompute.md) |
| Ascend 计算通信并行优化 | [link](docs/features/communication-over-computation.md) |
| 【Prototype】Ulysses 长序列并行 | [link](docs/features/ulysses-context-parallel.md) |
| 【Prototype】Ascend MC2 | 暂无 |
| 【Prototype】alibi | 暂无 |
| 【Prototype】其他昇腾亲和优化 | 暂无 |

# 自定义算子

| 算子                         | 介绍                                             |
|----------------------------|------------------------------------------------|
| npu_dropout_add_layer_norm | [link](docs/ops/npu_dropout_add_layer_norm.md) |
| 【Prototype】rms_norm                   | [link](docs/ops/rms_norm.md)                   |
| 【Prototype】swiglu                     | [link](docs/ops/swiglu.md)                     |
| 【Prototype】lcal_coc                   | [link](docs/ops/lcal_coc.md)                   |

# 版本配套表

| MindSpeed版本     | Megatron版本    | PyTorch版本   | torch_npu版本    |CANN版本| Python版本                               |
| ----------------- | --- |------------- | ------------- | --------------------------------------- | ------------- |
|       1.0（商用）      | commitid bcce6f  |  2.1.0     |   6.0.RC1 |  8.0.RC1|Python3.8.x, Python3.9.x, Python3.10.x  |

# 分支维护策略

MindSpeed版本分支的维护阶段如下：

| **状态**            | **时间** | **说明**                                                               |
| ------------------- | -------- |----------------------------------------------------------------------|
| 计划                | 1—3 个月 | 计划特性                                                                 |
| 开发                | 3 个月   | 开发特性                                                                 |
| 维护                | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的MindSpeed版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布                                             |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                                                           |

# MindSpeed版本维护策略

| **MindSpeed版本** | **维护策略** | **当前状态** | **发布时间**   | **后续状态**         | **EOL日期** |
|-----------------|-----------|--------|------------|------------------|-----------|
| 1.0             |  常规版本  | 维护   | 2024/03/30 |  |           |


# 安全声明

[MindSpeed 安全声明](SECURITYNOTE.md)