# MindSpeed 安全声明

## 系统安全加固

- 用户可在运行系统配置时开启 ASLR（级别2）以提高系统安全性，保护系统随机化开启。  
可参考以下方式进行配置：

  ```
  echo 2 > /proc/sys/kernel/randomize_va_space
  ```

## 运行用户建议

- 基于安全性考虑，建议您在执行任何命令时，不建议使用root等管理员类型账户执行，遵循权限最小化原则。

## 文件权限控制

- 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
- 建议用户对训练所需文件、训练过程中保存的文件、用户个人的隐私数据、商业资产等敏感文件做好权限控制等安全措施，例如多用户共享数据集场景下的数据集文件写权限控制等，设定的权限建议参考表1进行设置。
- MindSpeed 中各类融合算子通过调用 PyTorch 中的 cpp_extension 特性进行编译，编译结果会默认缓存到 `~/.cache/torch_extensions` 目录下，建议用户根据自身需要，参考表1对生成文件做好权限控制。
- 原生 Megatron-LM 以及 PyTorch 框架运行中所生成的文件权限依赖系统设定，如 Megatron-LM 生成的数据集索引文件、torch.save 接口保存的文件等。建议当前执行脚本的用户根据自身需要，对生成文件做好权限控制，设定的权限可参考表1进行设置。
- 用户安装和使用过程需要做好权限控制，建议参考表1进行设置。如需要保存安装/卸载日志，可在安装/卸载命令后面加上参数 `--log <FILE>`， 注意对`<FILE>`文件及目录做好权限管控。

### 表1 文件（夹）各场景权限管控推荐最大值

| 类型           | Linux权限参考最大值 |
| -------------- | ---------------  |
| 用户主目录                        |   750（rwxr-x---）            |
| 程序文件(含脚本文件、库文件等)       |   550（r-xr-x---）             |
| 程序文件目录                      |   550（r-xr-x---）            |
| 配置文件                          |  640（rw-r-----）             |
| 配置文件目录                      |   750（rwxr-x---）            |
| 日志文件(记录完毕或者已经归档)        |  440（r--r-----）             | 
| 日志文件(正在记录)                |    640（rw-r-----）           |
| 日志文件目录                      |   750（rwxr-x---）            |
| Debug文件                         |  640（rw-r-----）         |
| Debug文件目录                     |   750（rwxr-x---）  |
| 临时文件目录                      |   750（rwxr-x---）   |
| 维护升级文件目录                  |   770（rwxrwx---）    |
| 业务数据文件                      |   640（rw-r-----）    |
| 业务数据文件目录                  |   750（rwxr-x---）      |
| 密钥组件、私钥、证书、密文文件目录    |  700（rwx—----）      |
| 密钥组件、私钥、证书、加密密文        | 600（rw-------）      |
| 加解密接口、加解密脚本            |   500（r-x------）        |

## 数据安全声明

- MindSpeed 依赖 CANN 的基础能力实现 AOE 性能调优、算子 dump、日志记录等功能，用户需要关注上述功能生成文件的权限控制。

## 运行安全声明

- 建议用户结合运行环境资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
- MindSpeed 在运行异常时会退出进程并打印报错信息，建议根据报错提示定位具体错误原因，包括设定算子同步执行、查看 CANN 日志、解析生成的 Core Dump 文件等方式。

## 公网地址声明
- MindSpeed代码中包含公网地址声明如下表所示：

|      类型      |                                           开源代码地址                                           |                            文件名                             |             公网IP地址/公网URL地址/域名/邮箱地址             |                   用途说明                    |
| :------------: |:------------------------------------------------------------------------------------------:|:----------------------------------------------------------:| :----------------------------------------------------------: |:-----------------------------------------:|
|  开源引入  |      https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py       |                   mindspeed/moe/gate.py                    |          https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py       |             deepspeed moe源码地址             |
|  开源引入  |      https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py       |                   mindspeed/moe/gate.py                    |          https://arxiv.org/pdf/2006.16668.pdf       |              开源引入TopKGate类实现              |
|  开源引入  |     https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py      |                   mindspeed/moe/gate.py                    |          https://arxiv.org/pdf/2202.08906.pdf       |            开源引入apply_z_loss实现             |
|  开源引入  |      https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py       |                 mindspeed/moe/moe_layer.py                 |          https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py       |             deepspeed moe源码地址             |
|  开源引入  |      https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py       |                 mindspeed/moe/moe_layer.py                 |          https://arxiv.org/pdf/2006.16668.pdf       |              开源引入MOELayer类实现              |
|  开源引入  |         https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/layer.py          |          mindspeed/moe/mixtral_parallel_mlpbm.py           |    https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/layer.py    |             deepspeed moe源码地址             |
|  开源引入  |         https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/layer.py          |                    mindspeed/moe/moe.py                    |    https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/layer.py    |             deepspeed moe源码地址             |
|  开源引入  |      https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py       |                   mindspeed/moe/utils.py                   |    https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py    |             deepspeed moe源码地址             |
|  开源引入  | https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py |                   mindspeed/moe/utils.py                   |   https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py     |             megatron moe源码地址              |
|  开源引入  |                       https://github.com/pytorch/pytorch/pull/40762                        |                   mindspeed/moe/utils.py                   |    https://github.com/pytorch/pytorch/pull/40762    |               alltoall实现源码                |
|  开源引入  |      https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py       |                   mindspeed/moe/utils.py                   |    https://arxiv.org/pdf/2006.16668.pdf    |                einsum论文地址                 |
|  开源引入  |        https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/experts.py         |                  mindspeed/moe/experts.py                  |     https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/experts.py   |             deepspeed moe源码地址             |
|  开源引入  |                               https://github.com/HazyResearch/flash-attention                               |              docs/features/flash-attention.md              |                               https://arxiv.org/pdf/2205.14135                               |            flash-attention说明文档            |
|  开源引入  |                                    https://github.com/nvidia/megatron-lm                                    |         docs/features/virtual-pipeline-parallel.md         |               https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf       |       virtual-pipeline-parallel说明文档       |
|  开源引入  |                            https://github.com/feifeibear/long-context-attention                             |          docs/features/hybrid-context-parallel.md          |                               https://arxiv.org/abs/2405.07719                               |        hybrid-context-parallel说明文档        |
|  开源引入  |                            https://github.com/feifeibear/long-context-attention                             |      docs/features/ring-attention-context-parallel.md      |                           https://arxiv.org/pdf/2310.01889                                  |    ring-attention-context-parallel说明文档    |
|  开源引入  |                          https://github.com/ofirpress/attention_with_linear_biases                          |                   docs/features/alibi.md                   |           https://arxiv.org/pdf/2108.12409                                           |                 alibi说明文档                 |
|  开源引入  |                                    https://github.com/NVIDIA/Megatron-LM                                    |             docs/features/sequence-parallel.md             |           https://arxiv.org/pdf/2205.05198                                                |           sequence-parallel说明文档           |
|  开源引入  |                                    https://github.com/NVIDIA/Megatron-LM                              |             docs/features/pipeline-parallel.md             |            https://arxiv.org/pdf/1806.03377             |           pipeline-parallel说明文档           |
|  开源引入  |                               https://github.com/NVIDIA/Megatron-LM/pull/598                                |                  docs/faq/data_helpers.md                  |            https://github.com/NVIDIA/Megatron-LM/pull/598                 |             data_helpers说明文档              |
|  开源引入  |                              https://pytorch.org/docs/stable/distributed.html                               |              mindspeed/core/parallel_state.py              |                       https://pytorch.org/docs/stable/distributed.html                       |         torch.distributed相关接口注意事项         |
|  开源引入  |                              https://github.com/pytorch/pytorch/pull/40762                               |                   mindspeed/moe/utils.py                   |                    https://github.com/pytorch/pytorch/pull/40762                      |              _AllToAll自动反向参考              |
|  开源引入  |                           https://github.com/NVIDIA/Megatron-LM                            |          mindspeed/optimizer/distrib_optimizer.py          |      https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md      | distributed_optimizer_zero3_init文档字符串参数说明 |
|  开源引入  |                           https://github.com/InternLM/InternEvo                            | mindspeed/docs/features/ring-attention-context-parallel.md |                           https://arxiv.org/pdf/2406.18485                   |    ring-attention-context-parallel说明文档    |
|  开源引入  |                           https://github.com/sail-sg/zero-bubble-pipeline-parallelism                            |   mindspeed/docs/features/nanopipe-pipeline-parallel.md    |                           https://arxiv.org/abs/2401.10241                   |      nanopipe-pipeline-parallel说明文档       |
|  开源引入  |                           https://github.com/iclr24-3434/AMPipe.git                            |             mindspeed/docs/features/ampipe.md              |                           https://openreview.net/pdf?id=yLgr02IsXY                   |                ampipe说明文档                 |
|  开源引入  |                           https://gitee.com/ascend/pytorch                            |       mindspeed/docs/features/adaptive-recompute.md        |                           https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha001/apiref/envref/envref_07_0053.html                   |     环境变量`PYTORCH_NPU_ALLOC_CONF`说明文档      |
|  开源引入  |                           https://github.com/deepseek-ai/DeepSeek-MoE                            |         mindspeed/docs/features/shared-experts.md          |                           https://arxiv.org/pdf/2401.06066                   |                 共享专家说明文档                  |
|  开源引入  |                           https://gitee.com/ascend/MindSpeed                            |                     mindspeed/setup.py                     |                           https://gitee.com/ascend/MindSpeed                   |               MindSpeed源码地址               |
|  开源引入  |                           https://gitee.com/ascend/MindSpeed/release                            |                     mindspeed/setup.py                     |                           https://gitee.com/ascend/MindSpeed/release                   |               MindSpeed源码地址               |
|  开源引入  |                           https://packaging.python.org/en/latest/single_source_version.html                            |                     mindspeed/setup.py                     |                           https://packaging.python.org/en/latest/single_source_version.html                   |               MindSpeed版本管理               |
|  开源引入  |                           https://github.com/NVIDIA/TransformerEngine/pull/719                            |                     mindspeed/core/data_parallel/distributed_data_parallel.py                     |                           https://github.com/NVIDIA/TransformerEngine/pull/719                   |       use_distributed_optimizer实现源码       |
|  开源引入  |                           https://github.com/NVIDIA/TransformerEngine/pull/719                            |                     mindspeed/mindspore/core/distributed/distributed_data_parallel.py                     |                           https://github.com/NVIDIA/TransformerEngine/pull/719                   |       use_distributed_optimizer实现源码       |

## 公开接口声明

-MindSpeed已更新其接口策略，现在除了对原生megatron在昇腾设备的无缝支持，还新增了针对融合算子的公开接口。用户在使用时，可以直接调用这些新增的融合算子接口，以充分利用MindSpeed在特定计算任务上的优化能力。
#### 判断函数是否为公开接口：
如果一个函数被定义在__all__中，并且在MindSpeed/tree/{分支}/docs 中进行了对外接口的文档记录，则该接口为公开接口，可以依赖其作为公共函数。该对外接口的具体使用方法以及场景请参照docs中的接口使用手册说明。如果需要依赖一个在文档中未记录的函数，请在MindSpeed主页开启Issue向我们确认该函数是否为公开接口、是否是因意外暴露、或者可能在未来被移除。

## 通信安全加固

[通信安全加固说明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA
)

## 通信矩阵
[通信矩阵说明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5%E4%BF%A1%E6%81%AF)

### 特殊场景
| 场景                                  | 使用方法                                         | 端口 | 可能的风险       |
|-------------------------------------| ------------------------------------------------ | ---------- | ---------- |
| 用户下载并使用HuggingFace的开源数据集            | 调用`load_dataset`函数，并填写目标开源数据集路径 | 随机端口     | 数据集可能包含敏感或不合法内容，导致合规问题。数据集中可能存在质量问题，如标签错误或数据偏差，影响数据预处理。|
| 使用`from_pretrained`信任特定代码，使用相关模型的实现 | 调用`from_pretrained`函数，设置`trust_remote_code=True` | 随机端口   | 如果trust_remote_code=True，下载的代码可能包含恶意逻辑或后门，威胁系统安全。但同时已设置local_files_only=True，程序仅会运行本地的文件来规避风险。   |
| 调用auto_settings进行训练任务时，新增端口           | torchrun拉起训练端口 auto_settings通过此端口指定MindSpeed拉起特定配置采集Profiling信息 | [1024, 65535]内 |业务需要，无风险     |
| 使用MindSpeed master分支进行训练任务时，新增48个端口           | MindSpeed 调用 Megatron 原生函数 `mpu.initialize_model_parallel` 来初始化模型并行组，并通过使用 PyTorch 分布式训练相关的 API 来启动任意任务。| [1024,65535]内  | 网络配置错误可能引发端口冲突或连接问题，影响训练效率。       |
| 使用MindSpeed core_0.9.0分支进行训练任务时，新增48个端口       | MindSpeed 调用 Megatron 原生函数 `mpu.initialize_model_parallel` 来初始化模型并行组，并通过使用 PyTorch 分布式训练相关的 API 来启动任意任务。| [1024,65535]内  | 网络配置错误可能引发端口冲突或连接问题，影响训练效率。       |
| 使用MindSpeed core_0.8.0分支进行训练任务时，新增32个端口       | MindSpeed 调用 Megatron 原生函数 `mpu.initialize_model_parallel` 来初始化模型并行组，并通过使用 PyTorch 分布式训练相关的 API 来启动任意任务。| [1024,65535]内  | 网络配置错误可能引发端口冲突或连接问题，影响训练效率。       |
| 使用MindSpeed 2.0.0_core_0.8.0分支进行训练任务时，新增32个端口 | MindSpeed 调用 Megatron 原生函数 `mpu.initialize_model_parallel` 来初始化模型并行组，并通过使用 PyTorch 分布式训练相关的 API 来启动任意任务。| [1024,65535]内  | 网络配置错误可能引发端口冲突或连接问题，影响训练效率。       |
| 使用MindSpeed core_0.7.0分支进行训练任务时，新增32个端口       | MindSpeed 调用 Megatron 原生函数 `mpu.initialize_model_parallel` 来初始化模型并行组，并通过使用 PyTorch 分布式训练相关的 API 来启动任意任务。| [1024,65535]内  | 网络配置错误可能引发端口冲突或连接问题，影响训练效率。       |
| 使用MindSpeed core_0.6.0分支进行训练任务时，新增32个端口       | MindSpeed 调用 Megatron 原生函数 `mpu.initialize_model_parallel` 来初始化模型并行组，并通过使用 PyTorch 分布式训练相关的 API 来启动任意任务。| [1024,65535]内  | 网络配置错误可能引发端口冲突或连接问题，影响训练效率。       |

