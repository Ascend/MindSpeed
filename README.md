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

## 架构说明

MindSpeed Core 当前版本（基于 Megatron-LM 0.17.x）对架构进行了分层解耦，将原有 Megatron/TE 原生适配内容拆分至独立组件，使 MindSpeed 聚焦于昇腾 NPU 独有加速特性：

| 组件 | 定位 | 说明 |
|------|------|------|
| [MegatronAdaptor](https://gitcode.com/Ascend/MegatronAdaptor)（MA） | NPU 基础适配层 | 提供 RNG、softmax、设备检测、TE 虚拟模块等基础 NPU 适配，使 Megatron-LM 能在昇腾设备上运行 |
| [TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU)（TENPU） | TE API 的 NPU 实现 | 提供 Linear、LayerNorm、DotProductAttention、FP8 基础等 Transformer Engine 核心算子的 NPU 实现 |
| MindSpeed Core | NPU 独有增强特性 | 提供 FP8 增强 recipe、通信融合、权重复用、亲和融合算子等昇腾独有加速能力 |

```text
Megatron-LM
  → MegatronAdaptor（基础 NPU 适配）
  → TransformerEngineNPU（TE API 的 NPU 实现）
  → MindSpeed Core（NPU 独有加速）
```

更多信息请参考[MindSpeed Core简介](./docs/zh/introduction.md)。

此外在 MindSpeed Core 加速库的基础之上也提供了大语言模型、多模态模型套件加速库:

- 📝 大语言模型库: [MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM)
- 🖼️ 多模态模型库: [MindSpeed MM](https://gitcode.com/Ascend/MindSpeed-MM)

## 代码仓目录结构

关键目录结构如下，详细目录介绍请参考[目录结构文档](./docs/zh/dir_structure.md)。

```plaintext
MindSpeed/
├── mindspeed/                    # 核心代码目录
│   ├── core/                     # 核心功能模块，包含并行策略、内存管理、优化器等核心能力
│   ├── features_manager/         # 特性管理模块，统一管理各种优化特性的注册与配置
│   ├── functional/               # 功能特性模块，包含NPU数据转储、确定性计算、性能分析等
│   ├── op_builder/               # 算子构建模块，提供算子编译和注册工具
│   ├── ops/                      # 算子模块，包含融合算子、自定义算子等高效实现
│   ├── args_utils.py             # 参数工具，提供参数解析和验证功能
│   ├── arguments.py              # 参数定义，包含分布式训练相关参数
│   ├── megatron_adapter.py       # MindSpeed增强适配器入口，配合MegatronAdaptor使用
│   ├── patch_utils.py            # 补丁工具，提供动态代码补丁功能
│   ├── train.py                  # 训练模块，提供训练流程控制
│   └── ...                       # 其他模块和功能
├── docs/                         # 文档目录，包含中英文特性文档、用户指南等
├── tests-extend/                 # 测试目录，包含扩展测试用例
└── tools/                        # 工具目录，提供辅助开发和性能分析工具
```

# 最新消息

---

- [May 21, 2025]: 🚀 MindSpeed Core 支持Mcore 0.12.1版本。
- [Jun 16, 2026]: 🚀 MindSpeed Core 完成Megatron-LM 0.17.1基础适配，架构分层解耦为MA + TENPU + MindSpeed，配合MegatronAdaptor 和 TransformerEngineNPU使用。

> 注： 当前版本初步支持两种版本的Transformer实现。如需回溯老版本Transformer实现，需要用户配置参数`--transformer-impl local`。

# 社区会议

---

- MindSpeed系列TC及SIG会议安排请查看[Ascend会议中心](https://meeting.ascend.osinfra.cn/)

# 版本说明

---

当前版本推荐配套表如下：

| 软件                   | 版本           |
|----------------------|--------------|
| MindSpeed Core 分支    | master       |
| MegatronAdaptor      | core_r0.17.0 |
| TransformerEngineNPU | main         |
| Megatron-LM 版本       | 0.17.1       |
| CANN 版本              | 9.0.0        |
| PyTorch              | 2.7.1        |
| torch_npu 版本         | 26.0.0       |
| Python 版本            | Python3.10.x |

更多具体说明请参考：[版本配套表](./docs/zh/release_notes_core.md#版本配套说明)。

# 安装

---

MindSpeed Core 当前版本基于 MA + TENPU + MindSpeed 三层架构，依赖关系如下：

```text
MegatronAdaptor (MA)           ← NPU 基础适配，使 Megatron-LM 能运行
  └→ TransformerEngineNPU      ← TE 算子的 NPU 实现，依赖 MA 完成设备初始化
       └→ MindSpeed Core       ← 昇腾独有加速，依赖 MA + TENPU 提供基础能力
```

> **关键约束**：
>
> - **安装顺序必须为 MA → TENPU → MindSpeed**，三者缺一不可。
> - TransformerEngineNPU 与原生 [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) 库模块名相同（均为 `transformer_engine`），**严禁在同一环境中同时安装两者**。如果已安装原生 TE，请先执行 `pip uninstall transformer_engine` 卸载。
> - MA 和 TENPU 各自有版本分支要求，请按下方步骤拉取对应分支，版本不匹配会导致运行时错误。

## 使用源码安装

获取并切换Megatron-LM版本至 core_r0.17.1 版本，可参考：

```shell
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.17.1
```

### 安装 MegatronAdaptor（NPU 基础适配层）

MA 负责完成 RNG、softmax、设备检测、TE 虚拟模块等基础适配，是 TENPU 和 MindSpeed 的运行前提。

```shell
git clone https://gitcode.com/Ascend/MegatronAdaptor.git
cd MegatronAdaptor
git checkout core_r0.17.0
cd ..
pip install -e MegatronAdaptor
```

### 安装 TransformerEngineNPU（TE API 的 NPU 实现）

TENPU 提供 Linear、LayerNorm、DotProductAttention、FP8 基础等 TE 核心算子在昇腾设备上的实现。**安装前确认环境中不存在原生 TransformerEngine**：

```shell
# 卸载原生 TE（如存在）
pip uninstall -y transformer_engine

# 安装 TENPU
git clone https://gitcode.com/Ascend/TransformerEngineNPU.git
pip install -e TransformerEngineNPU
```

> **验证**：执行 `pip show transformer_engine`，确认显示的路径为 `TransformerEngineNPU` 目录。

### 安装 MindSpeed Core

在MA和TENPU安装完成后，安装MindSpeed：

```shell
git clone https://gitcode.com/Ascend/MindSpeed.git
pip install -e MindSpeed
```

### 环境验证

安装完成后，可在Python中验证各组件是否正确安装：

```python
# 验证 MA
import megatron_adaptor     # 无报错即 MA 正常加载

# 验证 TENPU
import transformer_engine   # 无报错即 TENPU 正常（非原生 TE）

# 验证 MindSpeed
import mindspeed.megatron_adaptor  # 无报错即 MindSpeed 正常加载
```

具体请参考 [部署文档](./docs/zh/user-guide/install_guide.md) 了解硬件/OS 兼容性、CANN/PyTorch/torch_npu 安装等详细前置步骤。

# 快速上手

---

## 概述

使用MindSpeed Core仅须增加一行代码，即可在昇腾训练设备上以MA + TENPU + MindSpeed分层架构运行Megatron-LM，并进一步参考[特性介绍](#特性介绍) 使能MindSpeed的各项加速特性。

## 操作方法

以 GPT 模型为例：在 Megatron-LM 目录下修改`pretrain_gpt.py`文件，在`import torch`下新增一行：`import mindspeed.megatron_adaptor`，即如下修改：

  ```python
    import torch
    import mindspeed.megatron_adaptor # 新增代码行
    from functools import partial
    from contextlib import nullcontext
    import inspect
  ```

具体操作可以参考[快速上手指导](./docs/zh/user-guide/quickstart.md)。

MindSpeed LLM和MindSpeed MM的快速上手指导可参考：

- 大语言模型训练
  - [基于PyTorch框架](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/training/quick_start.md)
- 多模态模型训练
  - [基于PyTorch框架](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/quickstart.md)

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

各特性支持情况请查看[MindSpeed Core 特性支持情况](./docs/zh/features/feature_list.md)。

## 自定义算子

昇腾训练自定义算子统一由torch_npu提供API，以下API预计2025年Q4起不维护，请优先使用torch_npu提供的自定义算子，如有新需求或问题可提issue反馈，我们会尽快回复。

部分自定义算子设置为公开接口，公开接口设置说明请参照 MindSpeed 安全声明中的[公开接口声明](./docs/zh/SECURITYNOTE.md#公开接口声明)，具体对外接口细节参照以下算子对应的手册链接。

自定义算子支持情况请查看[MindSpeed Core 自定义算子支持情况](./docs/zh/ops/ops_list.md)。

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
| 26.0.0_core_r0.12.1 | 常规版本     | 维护        | 2026/03/30 | 预计2026/09/30起无维护   |           |
| 2.3.0_core_r0.12.1  | 常规版本     | 维护        | 2025/12/30 | 预计2026/06/30起无维护   |           |
| 2.2.0_core_r0.12.1  | 常规版本     | 停止维护     | 2025/09/30 | 2026/03/30起无维护  |           |
| 2.1.0_core_r0.12.1  | 常规版本     | 停止维护     | 2025/06/30 | 2025/12/30起无维护  |           |
| 2.1.0_core_r0.8.0   | 常规版本     | 停止维护     | 2025/06/30 | 2025/12/30起无维护  |           |
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

# 贡献指南

---

欢迎贡献 MindSpeed-Core！请查看[贡献指南](./CONTRIBUTING.md)了解如何参与项目贡献。

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
MindSpeed产品的使用许可证，具体请参见[LICENSE](./LICENSE)文件。
MindSpeed docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)文件。

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
