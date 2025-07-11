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

MindSpeed Core 是针对华为[昇腾设备](https://www.hiascend.com/)的大模型加速库。

大模型训练是一种非常复杂的过程，涉及到许多技术和挑战，其中大模型训练需要大量的显存资源是一个难题，对计算卡提出了不小的挑战。
为了在单个计算卡显存资源不足时，可以通过多张计算卡进行计算，业界出现了类似 Megatron、DeepSpeed 等第三方大模型加速库，对模型、输入数据等进行切分并分配到不同的计算卡上，最后再通过集合通信对结果进行汇总。

昇腾提供 MindSpeed Core 加速库，使能客户大模型业务快速迁移至昇腾设备，并且支持昇腾专有算法，确保开箱可用。

此外在 MindSpeed Core 加速库的基础之上也提供了大语言模型、多模态模型以及强化学习模型套件加速库:

- 📝 大语言模型库: [MindSpeed LLM](https://gitee.com/ascend/MindSpeed-LLM)
- 🖼️ 多模态模型库: [MindSpeed MM](https://gitee.com/ascend/MindSpeed-MM)
- 🖥️ 强化学习加速库: [MindSpeed RL](https://gitee.com/ascend/MindSpeed-RL)

---

# 📣 Latest News
- [May 21, 2025]: 🚀 MindSpeed Core 支持Mcore 0.12.1版本。

> 注： 当前版本仅支持local后端的transformer实现，需要用户配置参数`--transformer-impl local`。te后端实现正在筹备中，敬请期待。

---

# 安装

MindSpeed Core拉取源码后使用pip命令行安装`pip install -e MindSpeed`，具体请参考 [部署文档](./docs/user-guide/installation.md) 安装 MindSpeed Core 指定分支及其依赖软件。

获取并切换 Megatron-LM 版本至 core_v0.12.1 版本，可参考：
 ```shell
 git clone https://github.com/NVIDIA/Megatron-LM.git
 cd Megatron-LM
 git checkout core_v0.12.1
 ```

当前版本配套表如下：

| 软件               | 版本                       |
|------------------|--------------------------|
| MindSpeed Core分支 | master                   |
| Mcore版本          | 0.12.1                   |
| CANN版本           | 8.2.RC1                  |
| PyTorch          | 2.1.0、2.6.0              |
| torch_npu版本      | 7.1.RC1                  |
| Python版本         | Python3.9.x、Python3.10.x |


# 快速上手

使用MindSpeed Core仅须增加一行代码，即可在昇腾训练设备上运行Megatron-LM，并进一步参考[特性介绍](#特性介绍) 使能MindSpeed的各项加速特性。

以 GPT 模型为例：在 Megatron-LM 目录下修改`pretrain_gpt.py`文件，在`import torch`下新增一行：`import mindspeed.megatron_adaptor`，即如下修改：

  ```Python
    import torch
    import mindspeed.megatron_adaptor # 新增代码行
    from functools import partial
    from contextlib import nullcontext
    import inspect
  ```


具体操作可以参考[快速上手指导](./docs/user-guide/getting_started.md)。

---
# 加速特性分级说明

MindSpeed Core 加速特性分为三个层级，用户可根据实际需求选择通过设置启动脚本中的 `--optimization-level {层级}` 参数来自定义开启的优化层级。该参数支持以下配置：

<table><thead>
  <tr>
    <th width='50'>层级</th>
    <th width='180'>层级名称</th>
    <th width='600'>介绍</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5" style="text-align: center; vertical-align: middle"> 0 </td>
    <td>基础功能兼容</a></td>
    <td>提供Megatron-LM框架对NPU的基本功能适配。</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5" style="text-align: center; vertical-align: middle"> 1 </td>
    <td>亲和性增强🔥</a></td>
    <td>在L0基础上使能部分融合算子与昇腾亲和计算改写。</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5" style="text-align: center; vertical-align: middle"> 2 </td>
    <td>加速特性使能🔥🔥</a></td>
    <td>默认值。在L0、L1基础上开启更丰富的加速特性，加速特性通常通过具体参数使能，可参考“特性介绍”章节。</td>
  </tr>
</table>


# 特性介绍
MindSpeed 特性由七大模块组成，分别为：megetron特性支持、并行策略特性、内存优化特性、亲和计算特性、通信优化特性、关键场景特性以及多模态特性。其中【Released】表示是否商用发布，原型特性为非商用发布。

-  特性的介绍中说明了对应特性的应用场景及使用说明。一般而言，在脚本中加入相关参数即可轻松使用对应特性。🛰️

-  MindSpeed 加速特性仅支持mcore，这也是megatron在v0.6.0版本后主推分支，也是当前版本的默认分支。🛰️

-  当前大模型训练主要使用bf16数据类型，以下特性若无特殊声明原则上兼容fp16, 如使用其它数据类型遇到问题可提交issue, 我们会快速响应。🛰️

-  注意❗：在megatron_core_r0.9.0后，alltoall dispatcher进行了调整，原版本alltoall dispatcher重命名为alltoall_seq。MindSpeed MoE特性对各分支的支持情况，见各特性说明。

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
    <td rowspan="5"> Megatron 虚拟流水线并行</td>
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
    <td style="text-align: center; vertical-align: middle">✅</td>
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
    <tr>
      <td rowspan="5"> Ascend DualPipeV</td>
      <td><a href="docs/features/dualpipev.md">link</a></td>
      <td style="text-align: center; vertical-align: middle">✅</td>
    </tr>
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
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend 重计算流水线独立调度 </td>
    <td><a href="docs/features/recompute_independent_pipelining.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
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
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend swap_attention</td>
    <td><a href="docs/features/swap_attention.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
    <tr>
    <td rowspan="5">  Ascend Norm重计算</td>
    <td><a href="docs/features/norm-recompute.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
    <tr>
    <td rowspan="5">  Ascend Hccl Buffer 自适应</td>
    <td><a href="docs/features/hccl-group-buffer-set.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
    <tr>
    <td rowspan="5">  Ascend Swap Optimizer</td>
    <td><a href="docs/features/swap-optimizer.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
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
    <td rowspan="5">  Ascend Moe Token Permute and Unpermute 融合算子</td>
    <td><a href="docs/features/moe-token-permute-and-unpermute.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend npu_matmul_add_fp32 梯度累加融合算子</td>
    <td><a href="docs/features/npu_matmul_add.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
    <tbody>
    <tr>
    <td rowspan="5">  Ascend 计算通信并行优化</td>
    <td><a href="docs/features/communication-over-computation.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">❌</td>
  </tr>
    <tbody>
    <tr>
    <td rowspan="5"> Ascend MC2</td>
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
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend 高维张量并行  </td>
    <td><a href="docs/features/tensor-parallel-2d.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
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
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
  <tr>
    <td rowspan="5"> Ascend Megatron MoE Alltoall Dispatcher 性能优化 </td>
    <td><a href="docs/features/megatron_moe/megatron-moe-alltoall-dispatcher.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend Megatron MoE TP拓展EP </td>
    <td><a href="docs/features/megatron_moe/megatron-moe-tp-extend-ep.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend 共享专家  </td>
    <td><a href="docs/features/shared-experts.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
  <tbody>
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
    <td style="text-align: center; vertical-align: middle">✅</td>
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
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend PP支持动态形状</td>
    <td><a href="docs/features/variable_seq_lengths.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend PP支持多参数传递</td>
    <td><a href="docs/features/multi_parameter_pipeline.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> Ascend PP支持多参数传递和动态形状</td>
    <td><a href="docs/features/multi_parameter_pipeline_and_variable_seq_lengths.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
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
    <td style="text-align: center; vertical-align: middle">✅</td>
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
    <td style="text-align: center; vertical-align: middle">❌</td>
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
<tbody>
  <tr>
    <td rowspan="5"> npu_apply_fused_adamw_v2  </td>
    <td><a href="docs/ops/npu_apply_fused_adamw_v2.md">link</a></td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</table>

---


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

| **MindSpeed版本**     | **维护策略** | **当前状态** | **发布时间**   | **后续状态**          | **EOL日期** |
|---------------------|----------|----------|------------|-------------------|-----------|
| 2.1.0_core_r0.12.1  | 常规版本     | 维护       | 2025/06/30 | 预计2025/12/30起无维护	 |           |
| 2.1.0_core_r0.8.0   | 常规版本     | 维护       | 2025/06/30 | 预计2025/12/30起无维护	 |           |
| 2.0.0_core_r0.8.0   | 常规版本     | 维护       | 2025/03/30 | 预计2025/9/30起无维护	  |           |
| 1.0.0_core_r0.7.0   | 常规版本     | 停止维护     | 2024/12/30 | 2025/6/30起无维护	    |           |
| 1.0.0_core_r0.6.0   | 常规版本     | 停止维护     | 2024/12/30 | 2025/6/30起无维护	    |           |
| 1.0.RC3_core_r0.7.0 | 常规版本     | 停止维护     | 2024/09/30 | 2025/3/30起无维护	    |           |
| 1.0.RC3_core_r0.6.0 | 常规版本     | 停止维护     | 2024/09/30 | 2025/3/30起无维护	    |           |
| 1.0.RC2             | 常规版本     | 停止维护     | 2024/06/30 | 2024/12/30起无维护	   |           |
| 1.0.RC1             | 常规版本     | 停止维护     | 2024/03/30 | 2024/9/30起无维护     |           |

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

---

# 安全声明

⚠️ [MindSpeed 安全声明](SECURITYNOTE.md)

---

# 免责声明

## 致MindSpeed使用者
1. MindSpeed提供的所有内容仅供您用于非商业目的。
2. 对于MindSpeed测试用例以及示例文件中所涉及的各模型和数据集，平台仅用于功能测试，华为不提供任何模型权重和数据集，如您使用这些数据进行训练，请您特别注意应遵守对应模型和数据集的License，如您因使用这些模型和数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用MindSpeed过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。

## 致数据所有者
如果您不希望您的模型或数据集在MindSpeed中被提及，或希望更新MindSpeed中有关的描述，请在Gitee提交issue，我们将根据您的issue要求删除或更新您相关描述。衷心感谢您对MindSpeed的理解和贡献。

## License声明
Ascend MindSpeed中涉及的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可，对应许可证文本可查阅Ascend MindSpeed根目录。

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
- 科大讯飞AI工程院内核技术部

感谢来自社区的每一个PR，欢迎贡献 MindSpeed-Core！
