# <p align="center"> <img src="docs/LOGO.png" height="172px" width="598px"> </p>

<p align="center">
   <a href="./README.md"> 简体中文 </a> | English
</p>

<p align="center">
    <a> <img src="https://img.shields.io/badge/python-3.12-green"> </a>
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

# Introduction

---

MindSpeed Core is a large model acceleration library for Huawei [Ascend devices](https://www.hiascend.com/en).

Large model training is a highly complex process involving many technologies and challenges, among which the large amount of memory resources required by large model training poses considerable challenges to computing cards.
To enable computation across multiple computing cards when the memory resources of a single computing card are insufficient, the industry has introduced third-party large model acceleration libraries such as Megatron and DeepSpeed, which partition models, input data, and so on, distribute them to different computing cards, and finally aggregate the results through collective communication.

Ascend provides MindSpeed Core, enabling customers' large model workloads to be quickly migrated to Ascend devices, while also supporting Ascend-proprietary algorithms to ensure out-of-the-box usability.

## Architecture

The current version of MindSpeed Core (based on Megatron-LM 0.18.x) decouples the architecture into layers and extracts the original Megatron/TE native adaptation content into independent components, so that MindSpeed focuses on Ascend NPU-specific acceleration features:

| Component | Role | Description |
|------|------|------|
| [MegatronAdaptor](https://gitcode.com/Ascend/MegatronAdaptor) (MA) | Basic NPU adaptation layer | Provides basic NPU adaptation such as RNG, softmax, device detection, and the TE virtual module, enabling Megatron-LM to run on Ascend devices |
| [TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU) (TENPU) | NPU implementation of the TE API | Provides NPU implementations of core Transformer Engine operators such as Linear, LayerNorm, DotProductAttention, and FP8 basic |
| MindSpeed Core | NPU-specific enhancement features | Provides Ascend-specific acceleration capabilities such as enhanced FP8 recipes, communication fusion, weight reuse, and affinity fusion operators |

```text
Megatron-LM
  → MegatronAdaptor (basic NPU adaptation)
  → TransformerEngineNPU (NPU implementation of the TE API)
  → MindSpeed Core (NPU-specific acceleration)
```

For more information, refer to [MindSpeed Core Introduction](./docs/en/introduction.md).

In addition, based on MindSpeed Core, acceleration libraries for large language model and multimodal model suites are also provided:

- 📝 Large language model library: [MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM)
- 🖼️ Multimodal model library: [MindSpeed MM](https://gitcode.com/Ascend/MindSpeed-MM)

## Repository Directory Structure

The key directory structure is as follows. For a detailed directory introduction, refer to the [Directory Structure](./docs/en/dir_structure.md).

```plaintext
MindSpeed/
├── mindspeed/                    # Core code directory
│   ├── core/                     # Core functional modules, including parallel strategies, memory management, optimizers, and other core capabilities
│   ├── features_manager/         # Feature management module, unifying registration and configuration of various optimization features
│   ├── functional/               # Functional modules, including NPU data dumping, deterministic computation, performance analysis, etc.
│   ├── op_builder/               # Operator builder module, providing tools for operator compilation and registration
│   ├── ops/                      # Operator modules, including efficient implementations of fusion operators and custom operators
│   ├── args_utils.py             # Argument utilities, providing parameter parsing and validation functionality
│   ├── arguments.py              #  Argument definitions, including distributed training-related parameters
│   ├── megatron_adapter.py       # MindSpeed enhanced adapter entry, used together with MegatronAdaptor
│   ├── patch_utils.py            # Patch utilities, providing dynamic code patching functionality
│   ├── train.py                  # Training module, providing training flow control
│   └── ...                       # Other modules and utilities
├── docs/                         # Documentation directory, containing feature docs, user guides, etc. in both Chinese and English
├── tests-extend/                 # Test directory, containing extended test cases
└── tools/                        # Tools directory, providing development support and performance analysis tools
```

# Latest News

---

- [May 21, 2025]: 🚀 MindSpeed Core now supports Mcore 0.12.1.
- [Aug 7, 2026]: 🚀 MindSpeed Core completes basic adaptation for Megatron-LM 0.18.0, with the architecture decoupled into MA + TENPU + MindSpeed, and used together with MegatronAdaptor and TransformerEngineNPU.

> Note: The current version provides preliminary support for two Transformer implementations. To fall back to the legacy Transformer implementation, you need to configure the parameter `--transformer-impl local`.

# Community Meeting

---

- For the MindSpeed TC and SIG meeting schedules, see [Ascend Meeting Center](https://meeting.ascend.osinfra.cn/)

# Version Notes

---

The table below lists recommended versions:

| Software               | Version                |
|----------------------|----------------------|
| MindSpeed Core Branch    | 26.2.0_core_r0.18.0  |
| MegatronAdaptor      | 26.2.0_core_r0.18.0  |
| TransformerEngineNPU | 26.2.0_release_v2.17 |
| Megatron-LM Version     | core_v0.18.2               |
| CANN Version            | 9.2.0                |
| PyTorch              | 2.10.0               |
| torch_npu Version       | 26.2.0               |
| Python Version          | Python3.12.x         |

For more details, refer to: [Version Compatibility Table](./docs/en/release_notes_core.md#version-compatibility-information).

# Installation

---

The current version of MindSpeed Core is based on the three-layer architecture of MA + TENPU + MindSpeed, with the following dependencies:

```text
MegatronAdaptor (MA)           ← Basic NPU adaptation, enabling Megatron-LM to run
  └→ TransformerEngineNPU      ← NPU implementation of TE operators, relying on MA to complete device initialization
       └→ MindSpeed Core       ← Ascend-specific acceleration, relying on MA + TENPU to provide basic capabilities
```

> **Key Constraints**:
>
> - **The installation order must be MA → TENPU → MindSpeed**, and none of the three can be omitted.
> - TransformerEngineNPU and the native [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) library have the same module name (both are `transformer_engine`); **it is strictly forbidden to install both in the same environment**. If the native TE is already installed, first run `pip uninstall transformer_engine` to uninstall it.
> - MA and TENPU each have their own version branch requirements. Pull the corresponding branches as described in the steps below; a version mismatch will cause runtime errors.

## Installation from Source

To obtain and switch the Megatron-LM version to core_v0.18.2, refer to the following:

```shell
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.18.2
```

### Installing MegatronAdaptor (Basic NPU Adaptation Layer)

```shell
git clone https://gitcode.com/Ascend/MegatronAdaptor.git
cd MegatronAdaptor
git checkout 26.2.0_core_r0.18.0
cd ..
pip install -e MegatronAdaptor
```

### Installing TransformerEngineNPU (NPU Implementation of the TE API)

TENPU provides implementations of TE core operators such as Linear, LayerNorm, DotProductAttention, and FP8 basic on Ascend devices. **Before installation, confirm that the native TransformerEngine does not exist in the environment**:

```shell
# Uninstall the native TE (if present)
pip uninstall -y transformer_engine

# Install TENPU
git clone https://gitcode.com/Ascend/TransformerEngineNPU.git
cd TransformerEngineNPU
git checkout 26.2.0_release_v2.17
cd ..
pip install -e TransformerEngineNPU --no-build-isolation
```

> **Verification**: Run `pip show transformer_engine` and confirm that the displayed path is the `TransformerEngineNPU` directory.

### Installing MindSpeed Core

After MA and TENPU are installed, install MindSpeed:

```shell
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 26.2.0_core_r0.18.0
cd ..
pip install -e MindSpeed
```

### Environment Verification

After installation, you can verify in Python whether each component is correctly installed:

```python
# Verify MA
import megatron_adaptor     # If no error is reported, MA is loaded correctly

# Verify TENPU
import transformer_engine   # If no error is reported, TENPU is working correctly (not the native TE)

# Verify MindSpeed
import mindspeed.megatron_adaptor  # If no error is reported, MindSpeed is loaded correctly
```

For details, refer to the [Installation Guide](./docs/en/user-guide/install_guide.md) to learn about hardware/OS compatibility, CANN/PyTorch/torch_npu installation, and other detailed prerequisite steps.

# Quick Start

---

## Overview

To use MindSpeed Core, you only need to add one line of code to run Megatron-LM on Ascend training devices with the layered MA + TENPU + MindSpeed architecture, and then refer to [Feature Introduction](#feature-introduction) to enable the various acceleration features of MindSpeed.

## Usage

Take the GPT model as an example: in the Megatron-LM directory, modify the `pretrain_gpt.py` file and add a new line under `import torch`: `import megatron_adaptor`, as shown in the following modification:

  ```python
    import torch
    import megatron_adaptor # Add the new code line.
    from functools import partial
    from contextlib import nullcontext
    import inspect
  ```

For specific operations, refer to [Quick Start](./docs/en/user-guide/quickstart.md).

For the quick start guides of MindSpeed LLM and MindSpeed MM, refer to:

- LLM training
  - [Based on the PyTorch framework](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.2.0/docs/en/pytorch/training/quick_start.md)
- Multimodal model training
<!--  - [Based on the PyTorch framework](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.2.0/docs/en/guides/practices/quickstart.md) -->

# Acceleration Feature Layers

---

The MindSpeed Core acceleration feature is divided into three layers. You can select an optimization layer by setting the `--optimization-level {Layer}` parameter in the startup script according to your actual needs. This parameter supports the following configurations:

<table>
  <thead>
    <tr>
      <th width="50">Layer No.</th>
      <th width="180">Layer Name</th>
      <th width="600">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center; vertical-align: middle">0</td>
      <td>Basic function compatibility</td>
      <td>Provides basic functional adaptation of the Megatron-LM framework for NPU.</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td style="text-align: center; vertical-align: middle">1</td>
      <td>Affinity enhancement 🔥</td>
      <td>On top of L0, enables some fused operators and Ascend-affinity computation rewrites.</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td style="text-align: center; vertical-align: middle">2</td>
      <td>Acceleration enabling 🔥🔥</td>
      <td>This the default value. On top of L0 and L1, enables a richer set of acceleration features. These acceleration features are usually enabled through specific parameters. For details, refer to the "Feature Introduction" chapter.</td>
    </tr>
  </tbody>
</table>

# Feature Introduction

---

MindSpeed features consist of seven major modules: Megatron, parallel strategy, memory optimization, affinity computation, communication optimization, key scenario, and multimodal features. Among them, [released] indicates whether the feature is commercially released; [prototype] features are not commercially released.

- The application scenarios and usage instructions of the corresponding feature are provided. Generally, adding the relevant parameters to the script allows you to easily use the feature. 🛰️

- MindSpeed acceleration features support only Mcore, which is the branch promoted by Megatron after v0.6.0 and is also the default branch of the current version. 🛰️

- Current large model training mainly uses the bf16 data type. Unless otherwise stated, the following features are in principle compatible with fp16. If you encounter problems when using other data types, you can submit an issue, and we will respond in a timely manner. 🛰️

- Note❗: After `Megatron_core_r0.18.0`, `alltoall dispatcher` has been adjusted, and `alltoall_seq` is no longer supported. For the support status of MindSpeed MoE features across branches, see the description of each feature.

For details, see [MindSpeed Core Feature Support](./docs/en/features/feature_list.md).

## Custom Operators

Custom operators for Ascend training are uniformly provided through TorchNPU APIs. The following APIs are expected to be no longer maintained starting from Q4, 2025. Please prioritize using the custom operators provided by TorchNPU. If you have new requirements or encounter problems, you can submit an issue for feedback, and we will respond as soon as possible.

Some custom operators are set as public APIs. For instructions on public API settings, please refer to the [Public API Statement](./docs/en/SECURITYNOTE.md) in the *MindSpeed Security Statement*. For specific external API details, refer to the manual links corresponding to the operators.

For custom operator support, please refer to [Custom Operators Supported by MindSpeed Core](./docs/en/ops/ops_list.md).

# Branch Maintenance Policy

---

🛠️ Maintenance period of MindSpeed branches:

| **Status**            | **Time** | **Description**                                                               |
| ------------------- | -------- |----------------------------------------------------------------------|
| Planned 🕐                | 1-3 months | Planned features                                                                 |
| Development 🕔              | 3 months   | Feature development                                                                 |
| Maintenance 🕚             | 6-12 months| Merge all resolved issues and release versions. Different maintenance policies are adopted for different MindSpeed versions. The maintenance cycles for Regular Releases and Long-Term Support releases are 6 months and 12 months, respectively. |
| No Maintenance 🕛          | 0-3 months | Merge all resolved issues, with no dedicated maintenance personnel and no version releases.                                             |
| End of Life (EOL) 🚫 | N/A      | The branch no longer accepts any modifications.                                                           |

🛠️ MindSpeed version maintenance policy:

| **MindSpeed Version**     | **Maintenance Policy** | **Current Status** | **Release Date**   | **Subsequent Status**          | **EOL Date** |
|---------------------|----------|----------|------------|-------------------|-----------|
| 26.1.0_core_r0.12.1 | Regular Release     | Maintenance        | 2026/06/30 | Estimated No Maintenance Since 2026/12/30   |           |
| 26.0.0_core_r0.12.1 | Regular Release     | Maintenance        | 2026/03/30 | Estimated No Maintenance Since 2026/09/30   |           |
| 2.3.0_core_r0.12.1  | Regular Release     | End of Maintenance     | 2025/12/30 | Estimated No Maintenance Since 2026/06/30  |           |
| 2.2.0_core_r0.12.1  | Regular Release     | End of Maintenance     | 2025/09/30 | No Maintenance Since 2026/03/30  |           |
| 2.1.0_core_r0.12.1  | Regular Release     | End of Maintenance     | 2025/06/30 | No Maintenance Since 2025/12/30  |           |
| 2.1.0_core_r0.8.0   | Regular Release     | End of Maintenance     | 2025/06/30 | No Maintenance Since 2025/12/30  |           |
| 2.0.0_core_r0.8.0   | Regular Release     | End of Maintenance     | 2025/03/30 | No Maintenance Since 2025/09/30  |           |
| 1.0.0_core_r0.7.0   | Regular Release     | End of Maintenance     | 2024/12/30 | No Maintenance Since 2025/06/30  |           |
| 1.0.0_core_r0.6.0   | Regular Release     | End of Maintenance     | 2024/12/30 | No Maintenance Since 2025/06/30  |           |
| 1.0.RC3_core_r0.7.0 | Regular Release     | End of Maintenance     | 2024/09/30 | No Maintenance Since 2025/03/30  |           |
| 1.0.RC3_core_r0.6.0 | Regular Release     | End of Maintenance     | 2024/09/30 | No Maintenance Since 2025/03/30  |           |
| 1.0.RC2             | Regular Release     | End of Maintenance     | 2024/06/30 | No Maintenance Since 2024/12/30  |           |
| 1.0.RC1             | Regular Release     | End of Maintenance     | 2024/03/30 | No Maintenance Since 2024/09/30  |           |

# FAQs

---

For FAQs, see [MindSpeed FAQs](./docs/en/FAQ.md).

# Reference

---

- [MindSpeed Accelerates Million-Level Ultra-Long Sequence Large Model Training](https://mp.weixin.qq.com/s/8q4MxCkosLn0yoneuxzynw)  🚀🚀
- [MindSpeed Accelerates Trillion-Parameter MoE Large Model Training](https://mp.weixin.qq.com/s/HQRzYzSUNNMonv5d1AP0OQ)  🚀🚀
- [Struggling with Large Model Training Memory Optimization? MindSpeed Has Solutions](https://mp.weixin.qq.com/s/lwjVgM67hwsgtOKp06zYPg) 🚀🚀

# Security Statement

---

⚠️ [MindSpeed Security Statement](./docs/en/SECURITYNOTE.md)

# Contribution Guide

---

Welcome to contribute to MindSpeed Core! Please refer to the [Contribution Guide](./CONTRIBUTING_en.md) to learn how to participate in project contributions.

# Disclaimer

---

## To MindSpeed Users

1. All content provided by MindSpeed is intended solely for your non-commercial use.
2. Regarding the models and datasets involved in MindSpeed test cases and sample files, the platform uses them only for functional testing. Huawei does not provide any model weights or datasets. If you use such data for training, please pay special attention to complying with the licenses of the corresponding models and datasets. Huawei assumes no responsibility for any infringement disputes arising from your use of these models and datasets.
3. If you encounter any issues (including but not limited to functional issues and compliance issues) while using MindSpeed, please submit an issue on Gitee, and we will review and resolve it in a timely manner.
4. The third-party open-source software that MindSpeed features depend on, such as Megatron, is provided and maintained by their respective third-party communities. Fixes for issues caused by third-party open-source software depend on the contributions and feedback of the relevant communities. You should understand that the MindSpeed repository does not guarantee fixes for issues in the third-party open-source software itself, nor does it guarantee testing or correcting all vulnerabilities and errors in third-party open-source software.

## To Data Owners

In addition, we appreciate the following teams for their contributions to the project:

## License Statement

For models involved in Ascend MindSpeed, if a license exists in the model directory, that license prevails. If no lLicense exists in the model directory, the model is licensed under the Apache 2.0 license, and the corresponding license content can be found in the root directory of Ascend MindSpeed.
For the license governing the use of MindSpeed, see [LICENSE](./LICENSE) file.
Documents in the MindSpeed docs directory are licensed under the CC-BY 4.0 license. For details, see the [LICENSE](./docs/LICENSE) file.

# Acknowledgments

---

🔎 MindSpeed-Core is jointly contributed by the following departments of Huawei:

Huawei:

- Ascend Computing Product Department
- Computing Algorithm Department
- Computing Software Platform Department
- Computing Technology Development Department
- Public Development Department: NAIE
- Network Technology Laboratory

In addition, we appreciate the following teams for their contributions to the project:

- WeChat Infrastructure Center
- Kernel Technology Department, iFLYTEK AI Engineering Institute

Thanks to every PR from the community. Contributions to MindSpeed Core are welcome!
