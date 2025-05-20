  <p align="center"> <img src="docs/LOGO.png" height="172px" width="598px"> </p>

<p align="center">
    <a> <img src="https://img.shields.io/badge/python-3.8%7C3.9%7C3.10-green"> </a>
    <a> <img src="https://img.shields.io/badge/build-passing-green"> </a>
    <a href="https://gitee.com/ascend/MindSpeed/blob/master/LICENSE">
        <img alt="Badge" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
    <a href="https://www.hiascend.com/software/mindspeed">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>

# 简介

MindSpeed 是针对华为[昇腾设备](https://www.hiascend.com/)的大模型加速库。

大模型训练是一种非常复杂的过程，涉及到许多技术和挑战，其中大模型训练需要大量的显存资源是一个难题，对计算卡提出了不小的挑战。
为了在单个计算卡显存资源不足时，可以通过多张计算卡进行计算，业界出现了类似 Megatron、DeepSpeed 等第三方大模型加速库，对模型、输入数据等进行切分并分配到不同的计算卡上，最后再通过集合通信对结果进行汇总。

昇腾提供 MindSpeed 加速库，使能客户大模型业务快速迁移至昇腾设备，并且支持昇腾专有算法，确保开箱可用。

---
# 安装

### 1. 安装依赖

☀️ 在安装 **MindSpeed** 之前，请参考[版本配套表](#版本配套表)，安装最新昇腾软件栈。

<table border="0">
  <tr>
    <th>依赖软件</th>
    <th>软件安装指南</th>
  </tr>

  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">《 <a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
  </tr>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
    <td rowspan="3">《 <a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td rowspan="3">《 <a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
  </tr>
  <tr>
    <td>torch_npu插件</td>
  </tr>
  <tr>
    <td>apex</td>
  </tr>
</table>




### 2. 安装 MindSpeed

☀️ 下载源码安装：

 ```shell
 git clone https://gitee.com/ascend/MindSpeed.git
 pip install -e MindSpeed
 ```

如需使用Ascend Transformer Boost（ATB）加速库算子，请先安装 CANN-NNAL 并初始化添加环境，例如：
 ```shell
# CANN-NNAL默认安装路径为：/usr/local/Ascend/nnal
# 运行CANN-NNAL默认安装路径下atb文件夹中的环境配置脚本set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 
 ```

### 3. 获取 Megatron-LM 并指定分支

☀️ 获取并切换 Megatron-LM 版本至 core_r0.10.0 的release版本，如下所示：
 ```shell
 git clone https://github.com/NVIDIA/Megatron-LM.git
 cd Megatron-LM
 git checkout core_r0.10.0
 ```



# 版本变更说明（自 core_r0.10.0 起）

## ⚠️ 重要更新

自 **core_r0.10.0** 开始，为了支持更广泛的用户需求，并解决之前用户意图运行 `--transformer-impl transformer_engine` 实现却意外使用了 `--transformer-impl local` 的问题，需要额外配置 Transformer 实现。

## ✅ 新增配置要求

请在启动命令中添加以下参数以明确指定使用本地（Local）Transformer 实现：

```bash
--transformer-impl local
```

此设置确保系统按预期执行特定的 Transformer 实现。

## 📌 注意事项

- 缺少该参数时，系统将依据默认配置选择实现方式，即`--transformer-impl transformer_engine`。
- 使用 `--transformer-impl transformer_engine` 的用户应确保未错误配置为 `--transformer-impl local` 实现。

我们建议所有用户升级后检查并调整相关配置，以匹配实际需求。


# 快速上手

1. 仅仅一行代码就可以轻松使能 MindSpeed 的各项功能。以 GPT 模型为例：在 Megatron-LM 目录下修改`pretrain_gpt.py`文件，在`import torch`下新增一行：`import mindspeed.megatron_adaptor`，即如下修改：

  ```diff
    import os
    import torch
  +import mindspeed.megatron_adaptor
    from functools import partial
    from typing import Union
  ```

2. 从core_r0.10.0版本开始，Megatron大量使用高版本语法的类型注解（Type Annotations），如:
  ```
    hierarchical_context_parallel_sizes: Optional[list[int]] = None
  ```

因此，若出现以下报错：
  ```
    TypeError: 'type' object is not subscriptable.
  ```
则需修改Megatron相应代码，或使用python 3.9及以上版本以适应Megatron原生接口的变动。

3. （可选）若未准备好相应训练数据，则需进行数据集的下载及处理供后续使用。数据集准备流程可参考
<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/Mindspeedguide/mindspeed_0003.html">数据集处理</a>。

4. 在 Megatron-LM 目录下，准备好训练数据，并在示例脚本中填写对应路径，然后执行。以下示例脚本可供参考。
  ```shell
  MindSpeed/tests_extend/example/train_distributed.sh
  ```

---
# 自定义优化级别
MindSpeed 提供了多层次的优化解决方案，并划分为三个层级，用户可根据实际需求灵活启用任意层级。高层级兼容低层级的能力，确保了整个系统的稳定性和扩展性。
用户可以通过设置启动脚本中的 `--optimization-level {层级}` 参数来自定义开启的优化层级。该参数支持以下配置：

<table><thead>
  <tr>
    <th width='50'>层级</th>
    <th width='180'>层级名称</th>
    <th width='600'>介绍</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5" style="text-align: center; vertical-align: middle"> 0 </td>
    <td>基础兼容层</a></td>
    <td>提供Megatron-LM框架对NPU的支持，确保无缝集成。该层包含基础功能集patch，保证可靠性和稳定性，为高级优化奠定基础。</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5" style="text-align: center; vertical-align: middle"> 1 </td>
    <td>亲和性增强层🔥</a></td>
    <td>兼容L0能力，集成高性能融合算子库，结合昇腾亲和的计算优化，充分释放昇腾算力，显著提升计算效率。</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5" style="text-align: center; vertical-align: middle"> 2 </td>
    <td>自研加速算法层🔥🔥</a></td>
    <td>默认值。该模式兼容了L1, L0能力，并集成了昇腾多项自主研发核心技术成果，可提供全面的性能优化。</td>
  </tr>
</table>


# 特性介绍
MindSpeed 特性由七大模块组成，分别为：megetron特性支持、并行策略特性、内存优化特性、亲和计算特性、通信优化特性、关键场景特性以及多模态特性。其中【Released】表示是否商用发布，原型特性为非商用发布。

-  特性的介绍中说明了对应特性的应用场景及使用说明。一般而言，在脚本中加入相关参数即可轻松使用对应特性。🛰️

-  MindSpeed 加速特性仅支持mcore，这也是megatron在v0.6.0版本后主推分支，也是当前版本的默认分支。🛰️

-  当前大模型训练主要使用bf16数据类型，以下特性若无特殊声明原则上兼容fp16, 如使用其它数据类型遇到问题可提交issue, 我们会快速响应。🛰️

## Megatron特性支持

<table><thead>
  <tr>
    <th width='250'>特性名称</th>
    <th>介绍</th>
    <th>Released</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> Megatron 数据并行</td>
    <td><a href="docs/features/data-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Megatron 张量并行</td>
    <td><a href="docs/features/tensor-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 流水并行</td>
    <td><a href="docs/features/pipeline-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 虚拟流水并行</td>
    <td><a href="docs/features/virtual-pipeline-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 分布式优化器</td>
    <td><a href="docs/features/distributed-optimizer.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 序列并行</td>
    <td><a href="docs/features/sequence-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 异步DDP</td>
    <td><a href="docs/features/async-ddp.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 权重更新通信隐藏 </td>
    <td><a href="docs/features/async-ddp-param-gather.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Megatron 重计算</td>
    <td><a href="docs/features/recomputation.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>

</table>


## 并行策略特性

<table><thead>
  <tr>
    <th width='250'>特性名称</th>
    <th>介绍</th>
    <th>Released</th>    
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Ulysses 长序列并行</td>
    <td><a href="docs/features/ulysses-context-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Ring Attention 长序列并行</td>
    <td><a href="docs/features/ring-attention-context-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
   <tr>
    <td rowspan="5"> Ascend Double Ring Attention 长序列并行</td>
    <td><a href="docs/features/double-ring.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend 混合长序列并行</td>
    <td><a href="docs/features/hybrid-context-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend 自定义空操作层</td>
    <td><a href="docs/features/noop-layers.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>

</table>

## 内存优化特性

<table><thead>
  <tr>
    <th width='250'>特性名称</th>
    <th>介绍</th>
    <th>Released</th>    
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> Ascend 激活函数重计算 </td>
    <td><a href="docs/features/activation-function-recompute.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend 重计算流水线独立调度 </td>
    <td><a href="docs/features/recompute_independent_pipelining.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend Mask归一</td>
    <td><a href="docs/features/generate-mask.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend BF16 参数副本复用</td>
    <td><a href="docs/features/reuse-fp32-param.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend swap_attention</td>
    <td><a href="docs/features/swap_attention.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  <tbody>
    <tr>
    <td rowspan="5">  Ascend Norm重计算</td>
    <td><a href="docs/features/norm-recompute.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  <tbody>
    <tr>
    <td rowspan="5">  Ascend Hccl Buffer 自适应</td>
    <td><a href="docs/features/hccl-group-buffer-set.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</table>


## 亲和计算特性

<table><thead>
  <tr>
    <th width='250'>特性名称</th>
    <th>介绍</th>
    <th>Released</th>    
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> Ascend rms_norm 融合算子 </td>
    <td><a href="docs/features/rms_norm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend swiglu 融合算子 </td>
    <td><a href="docs/features/swiglu.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend rotary_embedding 融合算子 </td>
    <td><a href="docs/features/rotary-embedding.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend flash attention</td>
    <td><a href="docs/features/flash-attention.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend npu_matmul_add_fp32 梯度累加融合算子</td>
    <td><a href="docs/features/npu_matmul_add.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
   <tr>
    <td rowspan="5">  Ascend Moe BMM通算融合算子</td>
    <td><a href="docs/features/megatron_moe/megatron-moe-bmm-fused.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
    <tbody>
    <tr>
    <td rowspan="5">  Ascend 计算通信并行优化</td>
    <td><a href="docs/features/communication-over-computation.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
    <tbody>
    <tr>
    <td rowspan="5"> Ascend MC2（存在已知问题⚠️）</td>
    <td><a href="docs/features/mc2.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
    <tbody>
    <tr>
    <td rowspan="5">  Ascend fusion_attention_v2 </td>
    <td><a href="docs/features/fusion-attn-v2.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</table>


## 通信优化特性

<table><thead>
  <tr>
    <th width='250'>特性名称</th>
    <th>介绍</th>
    <th>Released</th>    
  </tr></thead>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend Gloo 存档落盘优化 </td>
    <td><a href="docs/features/hccl-replace-gloo.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend 高维张量并行  </td>
    <td><a href="docs/features/tensor-parallel-2d.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  </table>

## Mcore MoE特性

<table><thead>
  <tr>
    <th width='250'>特性名称</th>
    <th>介绍</th>
    <th>Released</th>    
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Megatron MoE GMM  </td>
    <td><a href="docs/features/megatron_moe/megatron-moe-gmm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Megatron MoE Allgather Dispatcher 性能优化  </td>
    <td><a href="docs/features/megatron_moe/megatron-moe-allgather-dispatcher.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend Megatron MoE Alltoall Dispatcher 性能优化 </td>
    <td><a href="docs/features/megatron_moe/megatron-moe-alltoall-dispatcher.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Megatron MoE TP拓展EP </td>
    <td><a href="docs/features/megatron_moe/megatron-moe-tp-extend-ep.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend 共享专家  </td>
    <td><a href="docs/features/shared-experts.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend Megatron MoE 负载感知内存均衡算 </td>
    <td><a href="docs/features/megatron_moe/megatron-moe-adaptive-recompute-activation.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</table>


## 关键场景特性

<table><thead>
  <tr>
    <th width='250'>特性名称</th>
    <th>介绍</th>
    <th>Released</th>    
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5">  Ascend EOD Reset训练场景   </td>
    <td><a href="docs/features/eod-reset.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend alibi  </td>
    <td><a href="docs/features/alibi.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</table>

## 多模态特性

<table><thead>
  <tr>
    <th width='250'>特性名称</th>
    <th>介绍</th>
    <th>Released</th>    
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> Ascend fused ema adamw优化器   </td>
    <td><a href="docs/features/fused_ema_adamw_optimizer.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend PP支持动态形状</td>
    <td><a href="docs/features/variable_seq_lengths.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend PP支持多参数传递</td>
    <td><a href="docs/features/multi_parameter_pipeline.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend PP支持多参数传递和动态形状</td>
    <td><a href="docs/features/multi_parameter_pipeline_and_variable_seq_lengths.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend 非对齐线性层</td>
    <td><a href="docs/features/unaligned_linear.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend 非对齐Ulysses长序列并行</td>
    <td><a href="docs/features/unaligned-ulysses-context-parallel.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</table>

## 其它特性

<table><thead>
  <tr>
    <th width='250'>特性名称</th>
    <th>介绍</th>
    <th>Released</th>    
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> Ascend TFLOPS计算   </td>
    <td><a href="docs/features/ops_flops_cal.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Auto Settings 并行策略自动搜索系统 </td>
    <td><a href="docs/features/auto_settings.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend 确定性计算  </td>
    <td><a href="docs/features/npu_deterministic.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</table>


## 自定义算子

昇腾训练自定义算子统一由torch_npu提供API，以下API预计2025年q4起不维护，请优先使用torch_npu提供的自定义算子，如有新需求或问题可提issue反馈，我们会尽快回复。

部分自定义算子设置为公开接口，公开接口设置说明请参照 MindSpeed 安全声明中的[公开接口声明](SECURITYNOTE.md#公开接口声明)，具体对外接口细节参照以下算子对应的手册链接。

<table><thead>
  <tr>
    <th width='250'>自定义算子名称</th>
    <th>介绍</th>
    <th>Released</th>    
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> npu_dropout_add_layer_norm   </td>
    <td><a href="docs/ops/npu_dropout_add_layer_norm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> npu_rotary_position_embedding  </td>
    <td><a href="docs/ops/npu_rotary_position_embedding.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> fusion_attention  </td>
    <td><a href="docs/ops/fusion_attention.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> rms_norm   </td>
    <td><a href="docs/ops/rms_norm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> swiglu  </td>
    <td><a href="docs/ops/swiglu.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> npu_mm_all_reduce_add_rms_norm  </td>
    <td><a href="docs/ops/npu_mm_all_reduce_add_rms_norm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> npu_mm_all_reduce_add_rms_norm_  </td>
    <td><a href="docs/ops/npu_mm_all_reduce_add_rms_norm_.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> npu_gmm   </td>
    <td><a href="docs/ops/gmm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> npu_grouped_mat_mul_all_reduce  </td>
    <td><a href="docs/ops/npu_grouped_mat_mul_all_reduce.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> npu_fused_moe_token_permute  </td>
    <td><a href="docs/ops/npu_fused_moe_token_permute.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> npu_fused_moe_token_unpermute  </td>
    <td><a href="docs/ops/npu_fused_moe_token_unpermute.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> npu_ring_attention_update  </td>
    <td><a href="docs/ops/npu_ring_attention_update.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> npu_matmul_add_fp32  </td>
    <td><a href="docs/ops/npu_matmul_add.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> npu_groupmatmul_add_fp32 </td>
    <td><a href="docs/ops/npu_groupmatmul_add.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> npu_apply_fused_ema_adamw  </td>
    <td><a href="docs/ops/npu_apply_fused_ema_adamw.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> lcal_coc  </td>
    <td><a href="docs/ops/lcal_coc.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> ffn  </td>
    <td><a href="docs/ops/ffn.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> npu_all_to_all_all_gather_bmm  </td>
    <td><a href="docs/ops/npu_all_to_all_all_gather_bmm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> npu_bmm_reduce_scatter_all_to_all  </td>
    <td><a href="docs/ops/npu_bmm_reduce_scatter_all_to_all.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> quant_gmm  </td>
    <td><a href="docs/ops/quant_gmm.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
</table>

---
# MindSpeed 中采集Profile数据

📝 MindSpeed 支持命令式开启Profile采集数据，命令配置介绍如下：

| 配置命令                    | 命令含义                                                                                                    | 
|-------------------------|---------------------------------------------------------------------------------------------------------|
| --profile               | 打开profile开关                                                                                             |
| --profile-step-start    | 配置开始采集步，未配置时默认为10, 配置举例: --profile-step-start 30                                                        |
| --profile-step-end      | 配置结束采集步，未配置时默认为12, 配置举例: --profile-step-end 35                                                          |
| --profile-level         | 配置采集等级，未配置时默认为level0, 可选配置: level0, level1, level2, 配置举例: --profile-level level1                        |
| --profile-with-cpu      | 打开cpu信息采集开关                                                                                             |
| --profile-with-stack    | 打开stack信息采集开关                                                                                           |
| --profile-with-memory   | 打开memory信息采集开关，配置本开关时需打开--profile-with-cpu                                                              |
| --profile-record-shapes | 打开shapes信息采集开关                                                                                          |
| --profile-save-path     | 配置采集信息保存路径, 未配置时默认为./profile_dir, 配置举例: --profile-save-path ./result_dir                                |
| --profile-ranks         | 配置待采集的ranks,未配置时默认为-1，表示采集所有rank的profiling数据，配置举例: --profile-ranks 0 1 2 3, 需注意: 该配置值为每个rank在单机/集群中的全局值 |

---
# 版本配套表

💡 **PyTorch Extension**版本号采用`{PyTorch版本}-{昇腾版本}`命名规则，前者为**PyTorch Extension**匹配的PyTorch版本，后者用于匹配CANN版本，详细匹配如下：

| MindSpeed版本             | Megatron版本      | PyTorch版本   | torch_npu版本 | CANN版本  | Python版本                               | 硬件型态     |
|-------------------------|-----------------|------------- |-------------|---------|----------------------------------------|----------|
| master（主线）              | Core 0.10.0      |   2.1.0     | 在研版本        | 在研版本    | Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| core_r0.9.0（主线）         | Core 0.9.0      |   2.1.0     | 在研版本        | 在研版本    | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| core_r0.8.0（主线）         | Core 0.8.0      |   2.1.0     | 在研版本        | 在研版本    | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| core_r0.7.0（主线）         | Core 0.7.0      |   2.1.0     | 在研版本        | 在研版本    | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| core_r0.6.0（主线）         | Core 0.6.0      |   2.1.0     | 在研版本        | 在研版本    | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| 1.0.0_core_r0.7.0（商用）   | Core 0.7.0      |  2.1.0     | 6.0.0       | 8.0.0   | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| 1.0.0_core_r0.6.0（商用）   | Core 0.6.0      |  2.1.0     | 6.0.0       | 8.0.0   | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| 1.0.RC3_core_r0.7.0（商用） | Core 0.7.0      |  2.1.0     | 6.0.RC3     | 8.0.RC3 | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| 1.0.RC3_core_r0.6.0（商用） | Core 0.6.0      |  2.1.0     | 6.0.RC3     | 8.0.RC3 | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| 1.0.RC2（商用）             | Core 0.6.0      |  2.1.0     | 6.0.RC2     | 8.0.RC2 | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| 1.0.RC1（商用）             | commitid bcce6f |  2.1.0     | 6.0.RC1     | 8.0.RC1 | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |

[昇腾辅助软件](https://gitee.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中有更多关于PyTorch和CANN的版本信息。

# 分支维护策略

🛠️ MindSpeed 版本分支的维护阶段如下：

| **状态**            | **时间** | **说明**                                                               |
| ------------------- | -------- |----------------------------------------------------------------------|
| 计划 🕐                | 1-3 个月 | 计划特性                                                                 |
| 开发 🕔              | 3 个月   | 开发特性                                                                 |
| 维护 🕚             | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的MindSpeed 版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护 🕛          | 0-3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布                                             |
| 生命周期终止（EOL）🚫 | N/A      | 分支不再接受任何修改                                                           |

🛠️ MindSpeed 版本维护策略：

| **MindSpeed版本**     | **维护策略** | **当前状态** | **发布时间**   | **后续状态**           | **EOL日期** |
|---------------------|-----------|----------|------------|--------------------|-----------|
| 2.0.0_core_r0.8.0   |  常规版本  | 维护       | 2025/03/30 | 预计2025/9/30起无维护	   |           |
| 1.0.0_core_r0.7.0   |  常规版本  | 维护       | 2024/12/30 | 预计2025/6/30起无维护	   |           |
| 1.0.0_core_r0.6.0   |  常规版本  | 维护       | 2024/12/30 | 预计2025/6/30起无维护	   |           |
| 1.0.RC3_core_r0.7.0 |  常规版本  | 维护       | 2024/09/30 | 预计2025/3/30起无维护	   |           |
| 1.0.RC3_core_r0.6.0 |  常规版本  | 维护       | 2024/09/30 | 预计2025/3/30起无维护	   |           |
| 1.0.RC2             |  常规版本  | 维护       | 2024/06/30 | 预计2024/12/30起无维护	   |           |
| 1.0.RC1             |  常规版本  | 停止维护     | 2024/03/30 | 2024/9/30起无维护           |           |

---

# 常见问题

| 现象                                 | 介绍                                    |
|------------------------------------|---------------------------------------|
| Data helpers 数据预处理出错  ❗             | [link](docs/faq/data_helpers.md)      |
| Torch extensions 编译卡住     ❗         | [link](docs/faq/torch_extensions.md)  |
| megatron0.7.0版本长稳测试出现grad norm为nan ❗| [link](docs/faq/megatron070_grad_norm_nan.md)  |
| Gloo建链失败Gloo connectFullMesh failed with ... ❗| [link](docs/features/hccl-replace-gloo.md)  |

# 技术文章
- [MindSpeed 加速百万级超长序列大模型训练](https://mp.weixin.qq.com/s/8q4MxCkosLn0yoneuxzynw)  🚀🚀
- [MindSpeed 加速万亿MoE大模型训练](https://mp.weixin.qq.com/s/HQRzYzSUNNMonv5d1AP0OQ)  🚀🚀
- [大模型训练内存优化难？MindSpeed 帮你来支招](https://mp.weixin.qq.com/s/lwjVgM67hwsgtOKp06zYPg) 🚀🚀

# 安全声明

⚠️ [MindSpeed 安全声明](SECURITYNOTE.md)

---

# 致谢

🔎 MindSpeed-Core 由华为公司的下列部门联合贡献 ：

华为公司：

- 昇腾计算产品部
- 计算算法部
- 计算软件平台部 
- 计算技术开发部
- 公共开发部：NAIE
- 网络技术实验室

此外，MindSpeed-Core 感谢以下团队对项目的贡献：

- 微信基础架构中心

感谢来自社区的每一个PR，欢迎贡献 MindSpeed-Core！
