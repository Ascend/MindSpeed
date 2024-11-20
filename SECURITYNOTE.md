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
- 建议用户对训练所需文件、训练过程中保存的文件、用户个人的隐私数据、商业资产等敏感文件做好权限控制等安全措施，例如多用户共享数据集场景下的数据集文件写权限控制等，设定的权限建议参考[附录A 文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)进行设置。
- MindSpeed 中各类融合算子通过调用 PyTorch 中的 cpp_extension 特性进行编译，编译结果会默认缓存到 `~/.cache/torch_extensions` 目录下，建议用户根据自身需要，参考[附录A 文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)对生成文件做好权限控制。
- 原生 Megatron-LM 以及 PyTorch 框架运行中所生成的文件权限依赖系统设定，如 Megatron-LM 生成的数据集索引文件、torch.save 接口保存的文件等。建议当前执行脚本的用户根据自身需要，对生成文件做好权限控制，设定的权限可参考[附录A 文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)进行设置。
- 运行时 CANN 可能会缓存算子编译文件，存储在运行目录下的`kernel_meta_*`文件夹内，加快后续训练的运行速度，用户可根据需要自行对生成后的相关文件进行权限控制。
- 用户安装和使用过程需要做好权限控制，建议参考[附录A 文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)文件权限参考进行设置。如需要保存安装/卸载日志，可在安装/卸载命令后面加上参数 `--log <FILE>`， 注意对`<FILE>`文件及目录做好权限管控。

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

MindSpeed依赖PyTorch组件，PyTorch提供分布式训练能力，支持在单机和多机场景下进行训练，需要进行网络通信。其中PyTorch需要使用TCP进行通信，torch_npu使用CANN中HCCL在NPU设备间通信，通信端口见[通信矩阵信息](#通信矩阵信息)。用户需要注意并保障节点间通信网络安全，可以使用iptables等方式消减安全风险，可参考[通信安全加固](#通信安全加固)进行网络安全加固。

##### 通信矩阵信息

| 组件                  | MindSpeed                                                                                                                                                                                                                                                                                                                                                                                          |  HCCL                                |
| --------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ------------------------------------ |
| 源设备                | 运行torch_npu进程的服务器                                                                                                                                                                                                                                                                                                                                                                                  | 运行torch_npu进程的服务器             |
| 源IP                  | 设备地址IP                                                                                                                                                                                                                                                                                                                                                                                             | 设备地址IP                            |
| 源端口                | 操作系统自动分配，分配范围由操作系统的自身配置决定                                                                                                                                                                                                                                                                                                                                                                          | 默认值为60000，取值范围[1024,65520]。用户可以通过HCCL_IF_BASE_PORT环境变量指定Host网卡起始端口号，配置后系统默认占用以该端口起始的16个端口      |
| 目的设备              | 运行torch_npu进程的服务器                                                                                                                                                                                                                                                                                                                                                                                  | 运行torch_npu进程的服务器              |
| 目的IP                | 设备地址IP                                                                                                                                                                                                                                                                                                                                                                                             | 设备地址IP                            |
| 目的端口 （侦听）      | 默认29500/29400，用户可以设定端口号                                                                                                                                                                                                                                                                                                                                                                            | 默认值为60000，取值范围[1024,65520]。用户可以通过HCCL_IF_BASE_PORT环境变量指定Host网卡起始端口号，配置后系统默认占用以该端口起始的16个端口      |
| 协议                  | TCP                                                                                                                                                                                                                                                                                                                                                                                                | TCP                                   |
| 端口说明              | 1. 在静态分布式场景中（用torchrun/torch.distributed.launch使用的backend为static）， 目的端口（默认29500）用于接收和发送数据，源端口用于接收和发送数据 <br> 2. 在动态分布式场景中（用torchrun或者torch.distributed.launch使用的backend为c10d）， 目的端口（默认29400）用于接收和发送数据，源端口用于接收和发送数据。可以使用rdzv_endpoint和master_port指定端口号 <br> 3. 在分布式场景中，用torchrun或者torch.distributed.launch不指定任何参数时使用的端口号为29500 <br> 4. torchrun拉起训练端口，auto_tuning通过此端口指定MindSpeed拉起特定配置采集Profiling信息 | 默认值为60000，取值范围[1024,65520]。用户可以通过HCCL_IF_BASE_PORT环境变量指定Host网卡起始端口号，配置后系统默认占用以该端口起始的16个端口       |
| 侦听端口是否可更改     | 是                                                                                                                                                                                                                                                                                                                                                                                                  | 是                                    |
| 认证方式              | 无认证方式                                                                                                                                                                                                                                                                                                                                                                                              | 无认证方式                             |
| 加密方式              | 无                                                                                                                                                                                                                                                                                                                                                                                                  | 无                                    |
| 所属平面              | 不涉及                                                                                                                                                                                                                                                                                                                                                                                                | 不涉及                                 |
| 版本                  | 所有版本                                                                                                                                                                                                                                                                                                                                                                                               | 所有版本                               |
| 特殊场景              | 无                                                                                                                                                                                                                                                                                                                                                                                                  | 无                                     |
| 备注                  | 该通信过程由开源软件PyTorch控制，配置为PyTorch原生设置，可参考[PyTorch文档](https://pytorch.org/docs/stable/distributed.html#launch-utility)。源端口由操作系统自动分配，分配范围由操作系统的配置决定，例如ubuntu：采用/proc/sys/net/ipv4/ipv4_local_port_range文件指定，可通过cat /proc/sys/net/ipv4/ipv4_local_port_range或sysctl net.ipv4.ip_local_port_range查看                                                                                                       | 该通信过程由CANN中HCCL组件控制，torch_npu不进行控制，端口范围可参考[《环境变量参考》](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/apiref/envvar/envref_07_0001.html)的“执行相关 > 集合通信与分布式训练 > 集合通信相关配置>HCCL_IF_BASE_PORT”          |


## 附录

### A-文件（夹）各场景权限管控推荐最大值

| 类型           | linux权限参考最大值 |
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
