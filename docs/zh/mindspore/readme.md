# MindSpeed MindSpore后端介绍

MindSpeed已支持接入华为自研AI框架MindSpore，旨在提供华为全栈易用的端到端的大模型训练解决方案，以此获得更极致的性能体验。MindSpore后端提供了一套对标PyTorch的API，用户无需进行额外代码适配即可无缝切换。

---

# 安装

## 1. 安装依赖

<table border="0">
  <tr>
    <th>依赖软件</th>
    <th>软件安装指南</th>
  </tr>

  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">《 <a href="https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Debian&Software=cannToolKit">驱动固件安装指南</a> 》</td>
  </tr>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
    <td rowspan="3">《 <a href="https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0008.html?Mode=PmIns&OS=Debian&Software=cannToolKit">CANN 软件安装指南</a> 》</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
    <td>MindSpore</td>
    <td rowspan="1">《 <a href="https://gitee.com/mindspore/mindspore#%E5%AE%89%E8%A3%85">MindSpore AI框架安装指南</a> 》</td>
  </tr>
</table>

### 2. 获取 [MindSpore-Core-MS](https://gitcode.com/Ascend/MindSpeed-Core-MS/) 代码仓

执行以下命令拉取MindSpeed-Core-MS代码仓，并安装Python三方依赖库，如下所示：

```shell
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master
cd MindSpeed-Core-MS
pip install -r requirements.txt
```

可以参考MindSpeed-Core-MS目录下提供的[一键适配命令脚本](https://gitcode.com/Ascend/MindSpeed-Core-MS/#%E4%B8%80%E9%94%AE%E9%80%82%E9%85%8D)， 拉取并适配相应版本的MindSpeed、Megatron-LM和MSAdapter。

**若使用MindSpeed-Core-MS目录下的一键适配命令脚本（如[auto_convert.sh](https://gitcode.com/Ascend/MindSpeed-Core-MS/blob/master/auto_convert.sh)）可忽略后面步骤。**

### 3. 获取并适配相应版本的 MindSpeed、Megatron-LM 和 MSAdapter

（1）进入MindSpore-Core-MS目录后，获取指定版本仓库的源码：

```shell
# 获取指定版本的MindSpeed源码：
git clone https://gitcode.com/Ascend/MindSpeed.git -b master

# 获取指定版本的Megatron-LM源码：
git clone https://gitee.com/mirrors/Megatron-LM.git -b core_v0.12.1

# 获取指定版本的MSAdapter源码：
git clone https://openi.pcl.ac.cn/OpenI/MSAdapter.git -b master
```

具体版本对应关系参考MindSpore-Core-MS下的[一键适配命令脚本](https://gitcode.com/Ascend/MindSpeed-Core-MS/#%E4%B8%80%E9%94%AE%E9%80%82%E9%85%8D)，如[auto_convert.sh](https://gitcode.com/Ascend/MindSpeed-Core-MS/blob/master/auto_convert.sh)。

（2）设置环境变量：

```shell
# 在MindSpeed-Core-MS目录下执行
# 若在环境中PYTHONPATH等环境变量失效（例如退出容器后再进入等），需要重新设置环境变量
MindSpeed_Core_MS_PATH=$(pwd)
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MSAdapter:${MindSpeed_Core_MS_PATH}/MSAdapter/msa_thirdparty:${MindSpeed_Core_MS_PATH}/MindSpeed:$PYTHONPATH
echo $PYTHONPATH
```

（3）如需使用Ascend Transformer Boost（ATB）加速库算子，请先安装 CANN-NNAL 并初始化添加环境，例如：

```shell
# CANN-NNAL默认安装路径为：/usr/local/Ascend/nnal
# 运行CANN-NNAL默认安装路径下atb文件夹中的环境配置脚本set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

# 快速上手

1. 仅仅一行代码就可以轻松使能 MindSpeed 的各项功能。以 GPT 模型为例：在 Megatron-LM 目录下修改`pretrain_gpt.py`文件，在`import torch`下新增一行：`import mindspeed.megatron_adaptor`，即如下修改：

    ```diff
     import os
     import torch
    +import mindspeed.megatron_adaptor
     from functools import partial
     from typing import Union
    ```

2. （可选）若未准备好相应训练数据，则需进行数据集的下载及处理供后续使用。数据集准备流程可参考
<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/Mindspeedguide/mindspeed_0003.html">数据集处理</a>。

3. 在 Megatron-LM 目录下，准备好训练数据，并在示例脚本中填写对应路径，然后执行。以下示例脚本可供参考。

    ```shell
    MindSpeed/tests_extend/example/train_distributed_ms.sh
    ```

---

# 自定义优化级别

MindSpeed 提供了多层次的优化解决方案，并划分为三个层级，用户可根据实际需求灵活启用任意层级。高层级兼容低层级的能力，确保了整个系统的稳定性和扩展性。
用户可以通过设置启动脚本中的 `--optimization-level {层级}` 参数来自定义开启的优化层级。该参数支持以下配置：

<table>
  <thead>
    <tr>
      <th width='50'>层级</th>
      <th width='300'>层级名称</th>
      <th width='600'>介绍</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center; vertical-align: middle">0</td>
      <td>基础兼容层</td>
      <td>提供Megatron-LM框架对NPU的支持，确保无缝集成。该层包含基础功能集patch，保证可靠性和稳定性，为高级优化奠定基础。</td>
    </tr>
    <tr>
      <td style="text-align: center; vertical-align: middle">1</td>
      <td>亲和性增强层🔥</td>
      <td>兼容L0能力，集成高性能融合算子库，结合昇腾亲和的计算优化，充分释放昇腾算力，显著提升计算效率。</td>
    </tr>
    <tr>
      <td style="text-align: center; vertical-align: middle">2</td>
      <td>自研加速算法层🔥🔥</td>
      <td>默认值。该模式兼容了L1, L0能力，并集成了昇腾多项自主研发核心技术成果，可提供全面的性能优化。</td>
    </tr>
  </tbody>
</table>

# 特性介绍

MindSpeed 特性由七大模块组成，分别为：megatron特性支持、并行策略特性、内存优化特性、亲和计算特性、通信优化特性、关键场景特性以及多模态特性。其中【Released】表示是否商用发布，原型特性为非商用发布。

- 特性的介绍中说明了对应特性的应用场景及使用说明。一般而言，在脚本中加入相关参数即可轻松使用对应特性。🛰️

- 当前大模型训练主要使用bf16数据类型，以下特性若无特殊声明原则上兼容fp16, 如遇到fp16场景下相关问题请联系 MindSpeed 团队或提交issue, 我们会快速响应。🛰️

## 特性支持

MindSpore后端对MindSpeed的重要加速特性的支持情况如下表所示，部分不支持的特性将在后续迭代中逐步支持，敬请期待。
<table><thead>
  <tr>
    <th width='450'>特性名称</th>
    <th>介绍</th>
    <th>支持情况</th>
  </tr></thead>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 张量并行</td>
    <td><a href="../features/tensor-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 流水并行</td>
    <td><a href="../features/pipeline-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 虚拟流水并行</td>
    <td><a href="../features/virtual-pipeline-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 分布式优化器</td>
    <td><a href="../features/distributed-optimizer.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 序列并行</td>
    <td><a href="../features/sequence-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 权重更新通信隐藏 </td>
    <td><a href="../features/async-ddp-param-gather.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 重计算</td>
    <td><a href="../features/recomputation.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tr>
    <td rowspan="5"> Megatron 全分片并行</td>
    <td><a href="../features/custom_fsdp.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">暂不支持开启pp及--reuse-fp32-param参数配置</td>
  </tr>

</table>

## 并行策略特性

<table><thead>
  <tr>
    <th width='450'>特性名称</th>
    <th>介绍</th>
    <th>支持情况</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Ulysses 长序列并行</td>
    <td><a href="../features/ulysses-context-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
 <tbody>
  <tr>
    <td rowspan="5"> Ascend Ring Attention 长序列并行</td>
    <td><a href="../features/ring-attention-context-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
   <tr>
    <td rowspan="5"> Ascend Double Ring Attention 长序列并行</td>
    <td><a href="../features/double-ring.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend 混合长序列并行</td>
    <td><a href="../features/hybrid-context-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend 自定义空操作层</td>
    <td><a href="../features/noop-layers.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tr>
    <td rowspan="5"> Ascend DualPipeV</td>
    <td><a href="../features/dualpipev.md">link</a></td>
    <td style="text-align: center; vertical-align: middle"> 暂不支持--dualpipev-dw-detach参数配置 </td>
  </tr>
</table>

## 内存优化特性

<table><thead>
  <tr>
    <th width='450'>特性名称</th>
    <th>介绍</th>
    <th>支持情况</th>  
  </tr></thead>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend 激活函数重计算 </td>
    <td><a href="../features/activation-function-recompute.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend 重计算流水线独立调度 </td>
    <td><a href="../features/recompute_independent_pipelining.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend Mask归一</td>
    <td><a href="../features/generate-mask.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend BF16 参数副本复用</td>
    <td><a href="../features/reuse-fp32-param.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend swap_attention</td>
    <td><a href="../features/swap_attention.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  </tbody>
  <tbody>
    <tr>
    <td rowspan="5">  Ascend Norm重计算</td>
    <td><a href="../features/norm-recompute.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
    <tr>
    <td rowspan="5">  Ascend Hccl Buffer 自适应</td>
    <td><a href="../features/hccl-group-buffer-set.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  </tbody>
  <tbody>
    <tr>
    <td rowspan="5">  O2 BF16 Optimizer</td>
    <td><a href="../features/hccl-group-buffer-set.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  </tbody>
    <tr>
    <td rowspan="5">  SWAP优化器</td>
    <td><a href="../features/swap-optimizer.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</table>

## 亲和计算特性

<table><thead>
  <tr>
    <th width='450'>特性名称</th>
    <th>介绍</th>
    <th>支持情况</th>
  </tr></thead>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend rms_norm 融合算子 </td>
    <td><a href="../features/rms_norm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend swiglu 融合算子 </td>
    <td><a href="../features/swiglu.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend flash attention</td>
    <td><a href="../features/flash-attention.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend ring attention update</td>
    <td><a href="../features/ring-attention-context-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5">  Ascend Moe Token Permute and Unpermute 融合算子</td>
    <td><a href="../features/moe-token-permute-and-unpermute.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend npu_matmul_add_fp32 梯度累加融合算子</td>
    <td><a href="../features/npu_matmul_add.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
    </tbody>
    <tbody>
    <tr>
    <td rowspan="5">  Ascend 计算通信并行优化</td>
    <td><a href="../features/communication-over-computation.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
    </tbody>
    <tbody>
    <tr>
    <td rowspan="5"> Ascend MC2</td>
    <td><a href="../features/mc2.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
    </tbody>
    <tr>
    <td rowspan="5">  Ascend fusion_attention_v2 </td>
    <td><a href="../features/fusion-attn-v2.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</table>

## 通信优化特性

<table><thead>
  <tr>
    <th width='450'>特性名称</th>
    <th>介绍</th>
    <th>支持情况</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Gloo 存档落盘优化 </td>
    <td><a href="../features/hccl-replace-gloo.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</tbody>
  <tr>
    <td rowspan="5"> Ascend 高维张量并行  </td>
    <td><a href="../features/tensor-parallel-2d.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  </table>

## Mcore MoE特性

<table><thead>
  <tr>
    <th width='450'>特性名称</th>
    <th>介绍</th>
    <th>支持情况</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Megatron MoE GMM  </td>
    <td><a href="../features/megatron_moe/megatron-moe-gmm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Megatron MoE Allgather Dispatcher 性能优化  </td>
    <td><a href="../features/megatron_moe/megatron-moe-allgather-dispatcher.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  </tbody>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend Megatron MoE Alltoall Dispatcher 性能优化 </td>
    <td><a href="../features/megatron_moe/megatron-moe-alltoall-dispatcher.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Megatron MoE TP拓展EP </td>
    <td><a href="../features/megatron_moe/megatron-moe-tp-extend-ep.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> Ascend 共享专家  </td>
    <td><a href="../features/shared-experts.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  </tbody>
  <tr>
    <td rowspan="5"> Ascend Megatron MoE alltoall 通信掩盖 </td>
    <td><a href="../features/megatron_moe/megatron-moe-alltoall-overlap-comm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</table>

## 关键场景特性

<table><thead>
  <tr>
    <th width='450'>特性名称</th>
    <th>介绍</th>
    <th>支持情况</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5">  Ascend EOD Reset训练场景   </td>
    <td><a href="../features/eod-reset.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
  <tr>
    <td rowspan="5"> Ascend alibi  </td>
    <td><a href="../features/alibi.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</table>

## 多模态特性

<table><thead>
  <tr>
    <th width='450'>特性名称</th>
    <th>介绍</th>
    <th>支持情况</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> Ascend fused ema adamw优化器   </td>
    <td><a href="../features/fused_ema_adamw_optimizer.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> Ascend PP支持动态形状</td>
    <td><a href="../features/variable_seq_lengths.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> Ascend PP支持多参数传递</td>
    <td><a href="../features/multi_parameter_pipeline.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> Ascend PP支持多参数传递和动态形状</td>
    <td><a href="../features/multi_parameter_pipeline_and_variable_seq_lengths.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> Ascend 非对齐线性层</td>
    <td><a href="../features/unaligned_linear.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</tbody>
  <tr>
    <td rowspan="5"> Ascend 非对齐Ulysses长序列并行</td>
    <td><a href="../features/unaligned-ulysses-context-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</table>

## 其它特性

<table><thead>
  <tr>
    <th width='450'>特性名称</th>
    <th>介绍</th>
    <th>支持情况</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> Ascend TFLOPS计算   </td>
    <td><a href="../features/ops_flops_cal.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Auto Settings 并行策略自动搜索系统 </td>
    <td><a href="../features/auto_settings.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</tbody>
  <tr>
    <td rowspan="5"> Ascend 确定性计算  </td>
    <td><a href="../features/npu_deterministic.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</table>

## 自定义算子

部分自定义算子设置为公开接口，公开接口设置说明请参照 MindSpeed 安全声明中的[公开接口声明](../SECURITYNOTE.md#公开接口声明)，具体对外接口细节参照以下算子对应的手册链接。

<table><thead>
  <tr>
    <th width='450'>自定义算子名称</th>
    <th>介绍</th>
    <th>Released</th>    
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> npu_dropout_add_layer_norm   </td>
    <td><a href="../ops/npu_dropout_add_layer_norm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> npu_rotary_position_embedding  </td>
    <td><a href="../ops/npu_rotary_position_embedding.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> fusion_attention  </td>
    <td><a href="../ops/fusion_attention.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> rms_norm   </td>
    <td><a href="../ops/rms_norm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> swiglu  </td>
    <td><a href="../ops/swiglu.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> npu_mm_all_reduce_add_rms_norm  </td>
    <td><a href="../ops/npu_mm_all_reduce_add_rms_norm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> npu_mm_all_reduce_add_rms_norm_  </td>
    <td><a href="../ops/npu_mm_all_reduce_add_rms_norm_.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> npu_gmm   </td>
    <td><a href="../ops/gmm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> npu_grouped_mat_mul_all_reduce  </td>
    <td><a href="../ops/npu_grouped_mat_mul_all_reduce.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> lcal_coc  </td>
    <td><a href="../ops/lcal_coc.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> npu_apply_fused_ema_adamw  </td>
    <td><a href="../ops/npu_apply_fused_ema_adamw.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> ffn  </td>
    <td><a href="../ops/ffn.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  </tbody>
  <tr>
    <td rowspan="5"> npu_ring_attention_update  </td>
    <td><a href="../ops/npu_ring_attention_update.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> npu_matmul_add_fp32  </td>
    <td><a href="../ops/npu_matmul_add.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  </tbody>
<tbody>
  <tr>
    <td rowspan="5"> npu_groupmatmul_add_fp32 </td>
    <td><a href="../ops/npu_groupmatmul_add.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> npu_all_to_all_all_gather_bmm  </td>
    <td><a href="../ops/npu_all_to_all_all_gather_bmm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5"> npu_bmm_reduce_scatter_all_to_all  </td>
    <td><a href="../ops/npu_bmm_reduce_scatter_all_to_all.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</tbody>
  <tr>
    <td rowspan="5"> quant_gmm  </td>
    <td><a href="../ops/quant_gmm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</table>

---

# MindSpeed 中采集Profile数据

📝 MindSpeed 支持命令式开启Profile采集数据，命令配置介绍如下：

| 配置命令                    | 命令含义                                                                              | 
|-------------------------|-----------------------------------------------------------------------------------|
| --profile               | 打开profile开关                                                                       |
| --profile-step-start    | 配置开始采集步，未配置时默认为10, 配置举例: --profile-step-start 30                                 |
| --profile-step-end      | 配置结束采集步，未配置时默认为12, 配置举例: --profile-step-end 35                                   |
| --profile-level         | 配置采集等级，未配置时默认为level0, 可选配置: level0, level1, level2, 配置举例: --profile-level level1 |
| --profile-with-cpu      | 打开cpu信息采集开关                                                                       |
| --profile-with-stack    | 打开stack信息采集开关                                                                     |
| --profile-with-memory   | 打开memory信息采集开关，配置本开关时需打开--profile-with-cpu                                       |
| --profile-record-shapes | 打开shapes信息采集开关                                                                    |
| --profile-save-path     | 配置采集信息保存路径, 未配置时默认为./profile_dir, 配置举例: --profile-save-path ./result_dir          |
| --profile-ranks         | 配置待采集的ranks，未配置时默认为-1，表示采集所有rank的profiling数据，配置举例: --profile-ranks 0 1 2 3, 需注意: 该配置值为每个rank在单机/集群中的全局值   |

---

# 常见问题

| 现象                                 | 介绍                                    |
|------------------------------------|---------------------------------------|
| Data helpers 数据预处理出错  ❗             | [data_helpers](../faq/data_helpers.md)      |
| Torch extensions 编译卡住     ❗         | [torch_extensions](../faq/torch_extensions.md)  |
| megatron0.7.0版本长稳测试出现grad norm为nan ❗| [megatron070_grad_norm_nan](../faq/megatron070_grad_norm_nan.md)  |
| Gloo建链失败Gloo connectFullMesh failed with ... ❗| [hccl-replace-gloo](../features/hccl-replace-gloo.md)  |

# 技术文章

- [MindSpeed 加速百万级超长序列大模型训练](https://mp.weixin.qq.com/s/8q4MxCkosLn0yoneuxzynw)  🚀🚀
- [MindSpeed 加速万亿MoE大模型训练](https://mp.weixin.qq.com/s/HQRzYzSUNNMonv5d1AP0OQ)  🚀🚀
- [大模型训练内存优化难？MindSpeed 帮你来支招](https://mp.weixin.qq.com/s/lwjVgM67hwsgtOKp06zYPg) 🚀🚀
