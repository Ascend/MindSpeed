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

# Introduction

---

MindSpeed Core is a large model training acceleration library for Huawei [Ascend devices](https://www.hiascend.com/en).

Large model training is a highly complex process involving numerous technologies and challenges. Among them, the substantial video memory resources required for large model training pose a significant difficulty and a considerable challenge for compute cards.
To address the issue of insufficient video memory on a single compute card, computation can be distributed across multiple cards. This has led to the emergence of third-party large model acceleration libraries in the industry, such as Megatron and DeepSpeed, which partition models and input data, distribute them across different compute cards, and finally aggregate the results through collective communication.

Ascend provides the MindSpeed Core acceleration library, enabling customers to quickly migrate their large model businesses to Ascend devices, and supports Ascend-specific algorithms to ensure out-of-the-box usability. For more information, refer to [MindSpeed Core Introduction](./docs/en/introduction.md).

In addition, based on the MindSpeed Core acceleration library, it also provides acceleration libraries for large language model and multimodal model suites:

- 📝 Large Language Model Library: [MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM)
- 🖼️ Multimodal Model Library: [MindSpeed MM](https://gitcode.com/Ascend/MindSpeed-MM)

## Repository Directory Structure

The key directory structure is as follows. For a detailed directory introduction, please refer to the [Directory Structure Document](./docs/en/dir_structre.md).

```plaintext
MindSpeed/
├── mindspeed/                    # Core code directory
│   ├── core/                     # Core function module, containing core capabilities such as parallelism strategies, memory management, and optimizers
│   ├── features_manager/         # Feature management module, uniformly managing the registration and configuration of various optimization features
│   ├── functional/               # Functional feature module, including NPU data dumping, deterministic computation, performance analysis, etc.
│   ├── op_builder/               # Operator builder module, providing operator compilation and registration tools
│   ├── ops/                      # Operator module, containing efficient implementations such as fused operators and custom operators
│   ├── args_utils.py             # Argument utility, providing argument parsing and validation functions
│   ├── arguments.py              # Argument definitions, containing distributed training related arguments
│   ├── megatron_adapter.py       # Megatron-LM adapter, implementing integration with the Megatron framework
│   ├── patch_utils.py            # Patch tool, providing dynamic code patching functionality
│   ├── train.py                  # Training module, providing training process control
│   └── ...                       # Other modules and features
├── docs/                         # Documentation directory, containing Chinese and English feature documentation, user guides, etc.
├── tests-extend/                 # Test directory, containing extended test cases
└── tools/                        # Tool directory, providing auxiliary development and performance analysis tools
```

# Latest News

---

- [May 21, 2025]: 🚀 MindSpeed Core supports Mcore 0.12.1.

> Note: The current version provides preliminary support for two versions of the Transformer implementation. To fall back to the older Transformer implementation, users need to configure the parameter `--transformer-impl local`.

# Community Meetings

---

- For MindSpeed TC and SIG meeting schedules, please check [Ascend Meeting Center](https://meeting.ascend.osinfra.cn/)

# Release Notes

---

The recommended version compatibility table for the current version is as follows:

| Software               | Version                       |
|------------------|--------------------------|
| MindSpeed Core Branch | master                   |
| Mcore Version          | 0.12.1                   |
| CANN Version           | 9.0.0                  |
| PyTorch          | 2.7.1             |
| torch_npu Version      | 26.0.0                  |
| Python Version         | Python3.10.x |

For more details, please refer to: [Version Compatibility Table](docs/en/release_notes_core.md#version-compatibility-information).

# Installation

---

## Installation from Source Code

After pulling the source code of MindSpeed Core, install it using the pip command `pip install -e MindSpeed`. For details, refer to the [Deployment Document](./docs/en/user-guide/install_guide.md) to install the specified branch of MindSpeed Core and its dependent software.

Obtain and switch the Megatron-LM version to core_v0.12.1. Refer to:

 ```shell
 git clone https://github.com/NVIDIA/Megatron-LM.git
 cd Megatron-LM
 git checkout core_v0.12.1
 ```

# Quick Start

---

## Overview

To use MindSpeed Core, you only need to add one line of code to run Megatron-LM on Ascend training devices, and then refer to [Feature Introduction](#feature-introduction) to enable various acceleration features of MindSpeed.

## How to Use

Taking the GPT model as an example: In the Megatron-LM directory, modify the `pretrain_gpt.py` file and add a new line under `import torch`: `import mindspeed.megatron_adaptor`, as shown in the following modification:

  ```python
    import torch
    import mindspeed.megatron_adaptor # New code line added
    from functools import partial
    from contextlib import nullcontext
    import inspect
  ```

For specific operations, refer to the [Quick Start Guide](./docs/en/user-guide/quickstart.md).

Quick start guides for MindSpeed LLM and MindSpeed MM can be found at:

- Large Language Model Training
  - [Based on PyTorch Framework](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/en/pytorch/training/quick_start.md)
  - [Based on MindSpore Framework](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/en/mindspore/quick_start.md)
- Multimodal model training
  - [Based on PyTorch framework](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.0.0/docs/zh/pytorch/quickstart.md)
  - [Based on MindSpore framework](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.0.0/docs/zh/mindspore/quickstart_ms.md)

# Acceleration Hierarchy

---

MindSpeed Core acceleration features are divided into three hierarchies. You can customize the hierarchy to enable acceleration by setting the `--optimization-level {hierarchy}` parameter in the launch script according to your needs. This parameter supports the following configurations:

<table>
  <thead>
    <tr>
      <th width="50">Level</th>
      <th width="180">Feature Name</th>
      <th width="600">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center; vertical-align: middle">0</td>
      <td>Basic Feature Compatibility</td>
      <td>Provides basic functional adaptation of the Megatron-LM framework for the NPU.</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td style="text-align: center; vertical-align: middle">1</td>
      <td>Affinity Enhancement🔥</td>
      <td>Enables some fusion operators and Ascend affinity computation rewrites on top of L0.</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td style="text-align: center; vertical-align: middle">2</td>
      <td>Acceleration Enablement🔥🔥</td>
      <td>Default value. Enables richer acceleration features on top of L0 and L1. Acceleration features are typically enabled through specific parameters. Refer to the "Feature Introduction" section for details.</td>
    </tr>
  </tbody>
</table>

# Feature Introduction

---

MindSpeed features consist of seven major modules: Megatron support, parallel strategies, memory optimization, affinity computation, communication optimization, key scenario enablement, and multimodalibility. [Released] indicates whether the feature is commercially released. Prototype features are not commercially released.

- The feature introduction describes the app scenarios and usage instructions for the corresponding feature. Generally, you can easily use the corresponding feature by adding relevant parameters to the script. 🛰️

- MindSpeed acceleration features only support mcore, which is the main branch promoted by Megatron after version v0.6.0 and is also the default branch for the current version. 🛰️

- Current large model training primarily uses the bf16 data type. The following features are compatible with fp16 in principle unless otherwise stated. If you encounter issues when using other data types, please submit an issue, and we will respond quickly. 🛰️

- Note❗: After Megatron_core_r0.9.0, the `alltoall dispatcher` has been adjusted, and the original version of the `alltoall dispatcher` has been renamed to `alltoall_seq`. For the support status of MindSpeed MoE across different branches, refer to the respective feature descriptions.

For the support status of each feature, refer to [MindSpeed Core Feature List](./docs/en/feature_list.md).

## Custom Operator

Custom operators for Ascend training are uniformly provided through the torch_npu API. The following APIs are estimated to enter End of Maintenance starting Q4 2025. Prioritize using the custom operators provided by torch_npu. If you have new requirements or encounter issues, you can submit an issue for feedback, and we will respond as soon as possible.

Some custom operators are set as Public APIs. For instructions on Public API settings, refer to the [Public API Statement](docs/en/SECURITYNOTE.md#Open API Statement) in the MindSpeed Security Statement. For specific external API details, refer to the links corresponding to the following operators.

For the supported custom operators, check [MindSpeed Core Custom Operator List](./docs/en/ops_list.md).

# Branch Maintenance Strategy

---

🛠️ The maintenance phases for MindSpeed version branches are as follows:

| **Status** | **Time** | **Description** |
| ------------------- | -------- | --- |
| Planned 🕐 | 1-3 Months | Planned feature |
| Development 🕔 | 3 Months | Feature under development |
| Maintenance 🕚 | 6-12 Months | Merge all resolved issues and release versions. Different maintenance strategies are adopted for different MindSpeed versions. The maintenance cycles for Regular Versions and Long-Term Support versions are 6 months and 12 months, respectively. |
| Unmaintained 🕛 | 0-3 Months | Merge all resolved issues, no dedicated maintenance personnel, no version releases |
| End of Life (EOL) 🚫 | N/A | The branch no longer accepts any modifications |

🛠️ MindSpeed Version Maintenance Strategy:

| **MindSpeed Version**     | **Maintenance Strategy** | **Current Status** | **Release Date**   | **Subsequent Status**          | **EOL Date** |
|---------------------|----------|----------|------------|-------------------|-----------|
| 26.0.0_core_r0.12.1 | Regular Version     | In maintenance        | 2026/03/30 | Estimated No maintenance from 2026/09/30   |           |
| 2.3.0_core_r0.12.1  | Regular Version     | In maintenance        | 2025/12/30 | Estimated No maintenance from 2026/06/30   |           |
| 2.2.0_core_r0.12.1  | Regular Version     | End of maintenance     | 2025/09/30 | No maintenance from 2026/03/30  |           |
| 2.1.0_core_r0.12.1  | Regular Version     | End of maintenance     | 2025/06/30 | No maintenance from 2025/12/30  |           |
| 2.1.0_core_r0.8.0   | Regular Version     | End of maintenance     | 2025/06/30 | No maintenance from 2025/12/30  |           |
| 2.0.0_core_r0.8.0   | Regular Version     | End of maintenance     | 2025/03/30 | No maintenance from 2025/9/30   |           |
| 1.0.0_core_r0.7.0   | Regular Version     | End of maintenance     | 2024/12/30 | No maintenance from 2025/6/30     |           |
| 1.0.0_core_r0.6.0   | Regular Version     | End of maintenance     | 2024/12/30 | No maintenance from 2025/6/30     |           |
| 1.0.RC3_core_r0.7.0 | Regular Version     | End of maintenance     | 2024/09/30 | No maintenance from 2025/3/30     |           |
| 1.0.RC3_core_r0.6.0 | Regular Version     | End of maintenance     | 2024/09/30 | No maintenance from 2025/3/30     |           |
| 1.0.RC2             | Regular Version     | End of maintenance     | 2024/06/30 | No maintenance from 2024/12/30    |           |
| 1.0.RC1             | Regular Version     | End of maintenance     | 2024/03/30 | No maintenance from 2024/9/30     |           |

# FAQs

---

| Symptom                                 | Introduction                                    |
|------------------------------------|---------------------------------------|
| Data helpers data preprocessing error ❗             | [data_helpers data preprocessing error](docs/en/faq/data_helpers.md)      |
| Torch extensions compilation stuck ❗         | [Torch extensions stuck](docs/en/faq/torch_extensions.md)  |
| grad norm is nan in megatron0.7.0 long-term stability test ❗| [grad_norm_nan](docs/en/faq/megatron070_grad_norm_nan.md)  |
| Gloo connection failed Gloo connectFullMesh failed with ... ❗| [hccl-replace-gloo](docs/en/features/hccl-replace-gloo.md)  |

# Technical Articles

---

- [MindSpeed Accelerates Million-Level Ultra-Long Sequence Large Model Training](https://mp.weixin.qq.com/s/8q4MxCkosLn0yoneuxzynw) 🚀🚀
- [MindSpeed Accelerates Trillion-Parameter MoE Large Model Training](https://mp.weixin.qq.com/s/HQRzYzSUNNMonv5d1AP0OQ) 🚀🚀
- [Struggling with Large Model Training Memory Optimization? MindSpeed Has the Solution](https://mp.weixin.qq.com/s/lwjVgM67hwsgtOKp06zYPg) 🚀🚀

# Security Statement

---

⚠️ [MindSpeed Security Statement](./docs/en/SECURITYNOTE.md)

# Contribution Guide

---

Welcome to contribute to MindSpeed-Core! Please refer to the [Contribution Guide](https://gitcode.com/Ascend/MindSpeed/blob/master/CONTRIBUTING.md) to learn how to participate in project contributions.

# Disclaimer

---

## To MindSpeed Users

1. All content provided by MindSpeed is for non-commercial use only.
2. For the models and datasets involved in MindSpeed test cases and example files, the platform is used solely for functional testing. Huawei does not provide any model weights or datasets. If you use such data for training, please pay special attention to complying with the corresponding model and dataset licenses. Huawei assumes no responsibility for any infringement disputes arising from your use of these models and datasets.
3. If you discover any issues (including but not limited to functional issues or compliance issues) while using MindSpeed, please submit an issue on Gitee, and we will promptly review and resolve it.
4. Third-party open-source software that MindSpeed features depend on, such as Megatron, is provided and maintained by their respective third-party communities. The resolution of issues caused by third-party open-source software depends on the contributions and feedback of the relevant communities. You should understand that the MindSpeed repository does not guarantee fixes for issues within the third-party open-source software itself, nor does it guarantee testing and correcting all vulnerabilities and errors in third-party open-source software.

## To Data Owners

If you do not wish for your model or dataset to be mentioned in MindSpeed, or wish to update the relevant descriptions in MindSpeed, please submit an issue on Gitee. We will remove or update your relevant descriptions according to your issue requirements. We sincerely appreciate your understanding and contribution to MindSpeed.

## License Statement

For models involved in Ascend MindSpeed, if a License exists in the model directory, that License shall prevail. If no License exists in the model directory, the Apache 2.0 license applies, and the corresponding license text can be found in the Ascend MindSpeed root directory.
For the usage license of the MindSpeed product, please refer to the [LICENSE](./LICENSE) file.
Documents under the MindSpeed docs directory are subject to the CC-BY 4.0 license. For details, please refer to the [LICENSE](./LICENSE) file.

# Acknowledgments

---

🔎 MindSpeed-Core is jointly contributed by the following departments of Huawei:

Huawei:

- Ascend Computing Product Department
- Computing Algorithm Department
- Computing Software Platform Department
- Computing Technology Development Department
- Public Development Department: NAIE
- Network Technology Lab

Additionally, MindSpeed-Core thanks the following teams for their contributions to the project:

- WeChat Infrastructure Center
- Kernel Technology Department, iFLYTEK AI Engineering Institute

Thank you for every PR from the community. Contributions to MindSpeed-Core are welcome!
