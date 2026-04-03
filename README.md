# <p align="center"> <img src="docs/LOGO.png" height="172px" width="598px"> </p>

<p align="center">
    <a> <img src="https://img.shields.io/badge/python-3.8%7C3.9%7C3.10-green"> </a>
    <a> <img src="https://img.shields.io/badge/build-passing-green"> </a>
    <a href="https://gitcode.com/Ascend/MindSpeed/blob/master/LICENSE">
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

---

MindSpeed Core是针对华为[昇腾设备](https://www.hiascend.com/)的大模型加速库。

大模型训练是一种非常复杂的过程，涉及到许多技术和挑战，其中大模型训练需要大量的显存资源是一个难题，对计算卡提出了不小的挑战。
为了在单个计算卡显存资源不足时，可以通过多张计算卡进行计算，业界出现了类似 Megatron、DeepSpeed 等第三方大模型加速库，对模型、输入数据等进行切分并分配到不同的计算卡上，最后再通过集合通信对结果进行汇总。

昇腾提供 MindSpeed Core 加速库，使客户大模型业务能快速迁移至昇腾设备，并且支持昇腾专有算法，确保开箱可用。

此外在 MindSpeed Core 加速库的基础之上也提供了大语言模型、多模态模型以及强化学习模型套件加速库:

- 📝 大语言模型库: [MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM)
- 🖼️ 多模态模型库: [MindSpeed MM](https://gitcode.com/Ascend/MindSpeed-MM)
- 🖥️ 强化学习加速库: [MindSpeed RL](https://gitcode.com/Ascend/MindSpeed-RL)

# 最新消息

---

- [May 21, 2025]: 🚀 MindSpeed Core 支持Mcore 0.12.1版本。

> 注： 当前版本初步支持两种版本的Transformer实现。如需回溯老版本Transformer实现，需要用户配置参数`--transformer-impl local`。

# 社区会议

---

- MindSpeed系列TC及SIG会议安排请查看[Ascend会议中心](https://meeting.ascend.osinfra.cn/)

# 版本说明

---

当前版本推荐配套表如下：

| 软件               | 版本                       |
|------------------|--------------------------|
| MindSpeed Core分支 | 26.0.0_core_r0.12.1                  |
| Mcore版本          | 0.12.1                   |
| CANN版本           | 9.0.0                  |
| PyTorch          | 2.7.1             |
| torch_npu版本      | 26.0.0                  |
| Python版本         | Python3.10.x |

更多具体说明请参考：[版本配套表](./docs/zh/user-guide/installation.md#版本配套表)。

# 安装

---

MindSpeed Core拉取源码后使用pip命令行安装`pip install -e MindSpeed`，具体请参考 [部署文档](./docs/zh/user-guide/installation.md) 安装 MindSpeed Core 指定分支及其依赖软件。

获取并切换Megatron-LM版本至 core_v0.12.1 版本，可参考：

 ```shell
 git clone https://github.com/NVIDIA/Megatron-LM.git
 cd Megatron-LM
 git checkout core_v0.12.1
 ```

# 快速上手

---

## 概述

使用MindSpeed Core仅须增加一行代码，即可在昇腾训练设备上运行Megatron-LM，并进一步参考[特性介绍](#特性介绍) 使能MindSpeed的各项加速特性。

## 操作方法

以 GPT 模型为例：在 Megatron-LM 目录下修改`pretrain_gpt.py`文件，在`import torch`下新增一行：`import mindspeed.megatron_adaptor`，即如下修改：

  ```python
    import torch
    import mindspeed.megatron_adaptor # 新增代码行
    from functools import partial
    from contextlib import nullcontext
    import inspect
  ```

具体操作可以参考[快速上手指导](./docs/zh/user-guide/getting_started.md)。

# 加速特性分级说明

---

MindSpeed Core 加速特性分为三个层级，用户可根据实际需求选择通过设置启动脚本中的 `--optimization-level {层级}` 参数来自定义开启的优化层级。该参数支持以下配置：

<table>
  <thead>
    <tr>
      <th width="50">层级</th>
      <th width="180">层级名称</th>
      <th width="600">介绍</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center; vertical-align: middle">0</td>
      <td>基础功能兼容</td>
      <td>提供Megatron-LM框架对NPU的基本功能适配。</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td style="text-align: center; vertical-align: middle">1</td>
      <td>亲和性增强🔥</td>
      <td>在L0基础上使能部分融合算子与昇腾亲和计算改写。</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td style="text-align: center; vertical-align: middle">2</td>
      <td>加速特性使能🔥🔥</td>
      <td>默认值。在L0、L1基础上开启更丰富的加速特性，加速特性通常通过具体参数使能，可参考"特性介绍"章节。</td>
    </tr>
  </tbody>
</table>

# 特性介绍

---

MindSpeed 特性由七大模块组成，分别为：Megatron特性支持、并行策略特性、内存优化特性、亲和计算特性、通信优化特性、关键场景特性以及多模态特性。其中【Released】表示是否商用发布，原型特性为非商用发布。

- 特性的介绍中说明了对应特性的应用场景及使用说明。一般而言，在脚本中加入相关参数即可轻松使用对应特性。🛰️

- MindSpeed 加速特性仅支持mcore，这也是Megatron在v0.6.0版本后主推分支，也是当前版本的默认分支。🛰️

- 当前大模型训练主要使用bf16数据类型，以下特性若无特殊声明原则上兼容fp16, 如使用其它数据类型遇到问题可提交issue, 我们会快速响应。🛰️

- 注意❗：在Megatron_core_r0.9.0后，alltoall dispatcher进行了调整，原版本alltoall dispatcher重命名为alltoall_seq。MindSpeed MoE特性对各分支的支持情况，见各特性说明。

## Megatron特性支持

<table>
  <thead>
    <tr>
      <th width="250">特性名称</th>
      <th>介绍</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Megatron 数据并行</td>
      <td><a href="docs/zh/features/data-parallel.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Megatron 张量并行</td>
      <td><a href="docs/zh/features/tensor-parallel.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Megatron 流水并行</td>
      <td><a href="docs/zh/features/pipeline-parallel.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Megatron 虚拟流水线并行</td>
      <td><a href="docs/zh/features/virtual-pipeline-parallel.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Megatron 分布式优化器</td>
      <td><a href="docs/zh/features/distributed-optimizer.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Megatron 序列并行</td>
      <td><a href="docs/zh/features/sequence-parallel.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Megatron 异步DDP</td>
      <td><a href="docs/zh/features/async-ddp.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Megatron 权重更新通信隐藏</td>
      <td><a href="docs/zh/features/async-ddp-param-gather.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Megatron 重计算</td>
      <td><a href="docs/zh/features/recomputation.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td> Megatron 分布式权重</td>
      <td><a href="docs/zh/features/dist_ckpt.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td> Megatron 全分片并行</td>
      <td><a href="docs/zh/features/custom_fsdp.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td> Megatron Transformer Engine</td>
      <td><a href="docs/zh/features/transformer_engine.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td> Megatron Multi-head Latent Attention</td>
      <td><a href="docs/zh/features/multi-head-latent-attention.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
</tbody>
</table>

## 并行策略特性

<table>
  <thead>
    <tr>
      <th width="250">特性名称</th>
      <th>介绍</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Ascend Ulysses 长序列并行</td>
      <td><a href="docs/zh/features/ulysses-context-parallel.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend Ring Attention 长序列并行</td>
      <td><a href="docs/zh/features/ring-attention-context-parallel.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend Double Ring Attention 长序列并行</td>
      <td><a href="docs/zh/features/double-ring.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend 混合长序列并行</td>
      <td><a href="docs/zh/features/hybrid-context-parallel.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend 自定义空操作层</td>
      <td><a href="docs/zh/features/noop-layers.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend DualPipeV</td>
      <td><a href="docs/zh/features/dualpipev.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
  </tbody>
</table>

## 内存优化特性

<table>
  <thead>
    <tr>
      <th width="250">特性名称</th>
      <th>介绍</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Ascend 激活函数重计算</td>
      <td><a href="docs/zh/features/activation-function-recompute.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend 重计算流水线独立调度</td>
      <td><a href="docs/zh/features/recompute_independent_pipelining.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend Mask归一</td>
      <td><a href="docs/zh/features/generate-mask.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend BF16 参数副本复用</td>
      <td><a href="docs/zh/features/reuse-fp32-param.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend swap_attention</td>
      <td><a href="docs/zh/features/swap_attention.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend Norm重计算</td>
      <td><a href="docs/zh/features/norm-recompute.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend Hccl Buffer 自适应</td>
      <td><a href="docs/zh/features/hccl-group-buffer-set.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend Swap Optimizer</td>
      <td><a href="docs/zh/features/swap-optimizer.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Virtual Optimizer</td>
      <td><a href="docs/zh/features/virtual-optimizer.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
  </tbody>
</table>

## 亲和计算特性

<table>
  <thead>
    <tr>
      <th width="250">特性名称</th>
      <th>介绍</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Ascend rms_norm 融合算子</td>
      <td><a href="docs/zh/features/rms_norm.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend swiglu 融合算子</td>
      <td><a href="docs/zh/features/swiglu.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend rotary_embedding 融合算子</td>
      <td><a href="docs/zh/features/rotary-embedding.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend flash attention</td>
      <td><a href="docs/zh/features/flash-attention.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend Moe Token Permute and Unpermute 融合算子</td>
      <td><a href="docs/zh/features/moe-token-permute-and-unpermute.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend npu_matmul_add_fp32 梯度累加融合算子</td>
      <td><a href="docs/zh/features/npu_matmul_add.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend 计算通信并行优化</td>
      <td><a href="docs/zh/features/communication-over-computation.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
    <tr>
      <td>Ascend MC2</td>
      <td><a href="docs/zh/features/mc2.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
    <tr>
      <td>Ascend fusion_attention_v2</td>
      <td><a href="docs/zh/features/fusion-attn-v2.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
  </tbody>
</table>

## 通信优化特性

<table>
  <thead>
    <tr>
      <th width="250">特性名称</th>
      <th>介绍</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Ascend Gloo 存档落盘优化</td>
      <td><a href="docs/zh/features/hccl-replace-gloo.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend 高维张量并行</td>
      <td><a href="docs/zh/features/tensor-parallel-2d.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
  </tbody>
</table>

## Mcore MoE特性

<table>
  <thead>
    <tr>
      <th width="250">特性名称</th>
      <th>介绍</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Ascend Megatron MoE GMM</td>
      <td><a href="docs/zh/features/megatron_moe/megatron-moe-gmm.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend Megatron MoE Allgather Dispatcher 性能优化</td>
      <td><a href="docs/zh/features/megatron_moe/megatron-moe-allgather-dispatcher.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend Megatron MoE Alltoall Dispatcher 性能优化</td>
      <td><a href="docs/zh/features/megatron_moe/megatron-moe-alltoall-dispatcher.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend Megatron MoE TP拓展EP</td>
      <td><a href="docs/zh/features/megatron_moe/megatron-moe-tp-extend-ep.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Megatron MoE alltoall dispatcher分支通信隐藏优化</td>
      <td><a href="docs/zh/features/megatron_moe/megatron-moe-alltoall-overlap-comm.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
    <tr>
      <td>Megatron MoE allgather dispatcher分支通信隐藏优化</td>
      <td><a href="docs/zh/features/megatron_moe/megatron-moe-allgather-overlap-comm.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend 共享专家</td>
      <td><a href="docs/zh/features/shared-experts.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>1F1B Overlap</td>
      <td><a href="docs/zh/features/megatron_moe/megatron-moe-fb-overlap.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>专家并行动态负载均衡(数参互寻)</td>
      <td><a href="docs/zh/features/balanced_moe.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
  </tbody>
</table>

## 关键场景特性

<table>
  <thead>
    <tr>
      <th width="250">特性名称</th>
      <th>介绍</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Ascend EOD Reset训练场景</td>
      <td><a href="docs/zh/features/eod-reset.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend alibi</td>
      <td><a href="docs/zh/features/alibi.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
  </tbody>
</table>

## 多模态特性

<table>
  <thead>
    <tr>
      <th width="250">特性名称</th>
      <th>介绍</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Ascend fused ema adamw优化器</td>
      <td><a href="docs/zh/features/fused_ema_adamw_optimizer.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
    <tr>
      <td>Ascend PP支持动态形状</td>
      <td><a href="docs/zh/features/variable_seq_lengths.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend PP支持多参数传递</td>
      <td><a href="docs/zh/features/multi_parameter_pipeline.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend PP支持多参数传递和动态形状</td>
      <td><a href="docs/zh/features/multi_parameter_pipeline_and_variable_seq_lengths.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend 非对齐线性层</td>
      <td><a href="docs/zh/features/unaligned_linear.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend 非对齐Ulysses长序列并行</td>
      <td><a href="docs/zh/features/unaligned-ulysses-context-parallel.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
  </tbody>
</table>

## 其它特性

<table>
  <thead>
    <tr>
      <th width="250">特性名称</th>
      <th>介绍</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Ascend TFLOPS计算</td>
      <td><a href="docs/zh/features/ops_flops_cal.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>Ascend Auto Settings 并行策略自动搜索系统</td>
      <td><a href="docs/zh/features/auto_settings.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
    <tr>
      <td>Ascend 确定性计算</td>
      <td><a href="docs/zh/features/npu_deterministic.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
    <tr>
      <td>Ascend MindStudio Training Tools 精度对照</td>
      <td><a href="docs/zh/features/npu_datadump.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
  </tbody>
</table>

## 自定义算子

昇腾训练自定义算子统一由torch_npu提供API，以下API预计2025年Q4起不维护，请优先使用torch_npu提供的自定义算子，如有新需求或问题可提issue反馈，我们会尽快回复。

部分自定义算子设置为公开接口，公开接口设置说明请参照 MindSpeed 安全声明中的[公开接口声明](./docs/zh/SECURITYNOTE.md#公开接口声明)，具体对外接口细节参照以下算子对应的手册链接。

<table>
  <thead>
    <tr>
      <th width="250">自定义算子名称</th>
      <th>介绍</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>npu_dropout_add_layer_norm</td>
      <td><a href="docs/zh/ops/npu_dropout_add_layer_norm.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>npu_rotary_position_embedding</td>
      <td><a href="docs/zh/ops/npu_rotary_position_embedding.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>fusion_attention</td>
      <td><a href="docs/zh/ops/fusion_attention.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>rms_norm</td>
      <td><a href="docs/zh/ops/rms_norm.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>swiglu</td>
      <td><a href="docs/zh/ops/swiglu.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>npu_mm_all_reduce_add_rms_norm</td>
      <td><a href="docs/zh/ops/npu_mm_all_reduce_add_rms_norm.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>npu_mm_all_reduce_add_rms_norm_</td>
      <td><a href="docs/zh/ops/npu_mm_all_reduce_add_rms_norm_.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>npu_gmm</td>
      <td><a href="docs/zh/ops/gmm.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>npu_grouped_mat_mul_all_reduce</td>
      <td><a href="docs/zh/ops/npu_grouped_mat_mul_all_reduce.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>npu_ring_attention_update</td>
      <td><a href="docs/zh/ops/npu_ring_attention_update.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>npu_matmul_add_fp32</td>
      <td><a href="docs/zh/ops/npu_matmul_add.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>npu_groupmatmul_add_fp32</td>
      <td><a href="docs/zh/ops/npu_groupmatmul_add.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
    <tr>
      <td>npu_apply_fused_ema_adamw</td>
      <td><a href="docs/zh/ops/npu_apply_fused_ema_adamw.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
    <tr>
      <td>lcal_coc</td>
      <td><a href="docs/zh/ops/lcal_coc.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
    <tr>
      <td>ffn</td>
      <td><a href="docs/zh/ops/ffn.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
    <tr>
      <td>npu_all_to_all_all_gather_bmm</td>
      <td><a href="docs/zh/ops/npu_all_to_all_all_gather_bmm.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
    <tr>
      <td>npu_bmm_reduce_scatter_all_to_all</td>
      <td><a href="docs/zh/ops/npu_bmm_reduce_scatter_all_to_all.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
    <tr>
      <td>quant_gmm</td>
      <td><a href="docs/zh/ops/quant_gmm.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">❌</td>
    </tr>
    <tr>
      <td>npu_apply_fused_adamw_v2</td>
      <td><a href="docs/zh/ops/npu_apply_fused_adamw_v2.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
  </tbody>
</table>

# 分支维护策略

---

🛠️ MindSpeed 版本分支的维护阶段如下：

| **状态**            | **时间** | **说明**                                                               |
| ------------------- | -------- |----------------------------------------------------------------------|
| 计划 🕐                | 1-3 个月 | 计划特性                                                                 |
| 开发 🕔              | 3 个月   | 开发特性                                                                 |
| 维护 🕚             | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的MindSpeed 版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护 🕛          | 0-3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布                                             |
| 生命周期终止（EOL）🚫 | N/A      | 分支不再接受任何修改                                                           |

🛠️ MindSpeed 版本维护策略：

| **MindSpeed版本**     | **维护策略** | **当前状态** | **发布时间**   | **后续状态**          | **EOL日期** |
|---------------------|----------|----------|------------|-------------------|-----------|
| 2.3.0_core_r0.12.1  | 常规版本     | 维护       | 2025/12/30 | 预计2026/6/30起无维护   |           |
| 2.2.0_core_r0.12.1  | 常规版本     | 维护       | 2025/09/30 | 预计2026/3/30起无维护  |           |
| 2.1.0_core_r0.12.1  | 常规版本     | 维护       | 2025/06/30 | 预计2025/12/30起无维护  |           |
| 2.1.0_core_r0.8.0   | 常规版本     | 维护       | 2025/06/30 | 预计2025/12/30起无维护  |           |
| 2.0.0_core_r0.8.0   | 常规版本     | 停止维护     | 2025/03/30 | 2025/9/30起无维护   |           |
| 1.0.0_core_r0.7.0   | 常规版本     | 停止维护     | 2024/12/30 | 2025/6/30起无维护     |           |
| 1.0.0_core_r0.6.0   | 常规版本     | 停止维护     | 2024/12/30 | 2025/6/30起无维护     |           |
| 1.0.RC3_core_r0.7.0 | 常规版本     | 停止维护     | 2024/09/30 | 2025/3/30起无维护     |           |
| 1.0.RC3_core_r0.6.0 | 常规版本     | 停止维护     | 2024/09/30 | 2025/3/30起无维护     |           |
| 1.0.RC2             | 常规版本     | 停止维护     | 2024/06/30 | 2024/12/30起无维护    |           |
| 1.0.RC1             | 常规版本     | 停止维护     | 2024/03/30 | 2024/9/30起无维护     |           |

# 常见问题

---

| 现象                                 | 介绍                                    |
|------------------------------------|---------------------------------------|
| Data helpers 数据预处理出错  ❗             | [data_helpers数据预处理出错](docs/zh/faq/data_helpers.md)      |
| Torch extensions 编译卡住     ❗         | [Torch extensions卡住](docs/zh/faq/torch_extensions.md)  |
| megatron0.7.0版本长稳测试出现grad norm为nan ❗| [grad_norm_nan](docs/zh/faq/megatron070_grad_norm_nan.md)  |
| Gloo建链失败Gloo connectFullMesh failed with ... ❗| [hccl-replace-gloo](docs/zh/features/hccl-replace-gloo.md)  |

# 技术文章

---

- [MindSpeed 加速百万级超长序列大模型训练](https://mp.weixin.qq.com/s/8q4MxCkosLn0yoneuxzynw)  🚀🚀
- [MindSpeed 加速万亿MoE大模型训练](https://mp.weixin.qq.com/s/HQRzYzSUNNMonv5d1AP0OQ)  🚀🚀
- [大模型训练内存优化难？MindSpeed 帮你来支招](https://mp.weixin.qq.com/s/lwjVgM67hwsgtOKp06zYPg) 🚀🚀

# 安全声明

---

⚠️ [MindSpeed 安全声明](./docs/zh/SECURITYNOTE.md)

# 免责声明

---

## 致MindSpeed使用者

1. MindSpeed提供的所有内容仅供您用于非商业目的。
2. 对于MindSpeed测试用例以及示例文件中所涉及的各模型和数据集，平台仅用于功能测试，华为不提供任何模型权重和数据集，如您使用这些数据进行训练，请您特别注意应遵守对应模型和数据集的License，如您因使用这些模型和数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用MindSpeed过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。
4. MindSpeed功能依赖的Megatron等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，MindSpeed仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。

## 致数据所有者

如果您不希望您的模型或数据集在MindSpeed中被提及，或希望更新MindSpeed中有关的描述，请在Gitee提交issue，我们将根据您的issue要求删除或更新您相关描述。衷心感谢您对MindSpeed的理解和贡献。

## License声明

Ascend MindSpeed中涉及的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可，对应许可证文本可查阅Ascend MindSpeed根目录。

# 致谢

---

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
- 科大讯飞AI工程院内核技术部

感谢来自社区的每一个PR，欢迎贡献 MindSpeed-Core！
