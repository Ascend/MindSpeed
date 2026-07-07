# MindSpeed MindSpore Introduction

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-30T12:38:53.337Z pushedAt=2026-06-30T12:41:27.779Z -->

MindSpeed now supports integration with Huawei's AI framework MindSpore, aiming to provide an easy-to-use, full-stack, end-to-end large model training solution for Huawei, thereby achieving a more extreme performance experience. The MindSpore backend provides a set of APIs aligned with PyTorch, allowing users to switch seamlessly without additional code adaptation.

---

## Installation

### Installing Dependencies

<table border="0">
  <tr>
    <th>Dependency Software</th>
    <th>Software Installation Guide</th>
  </tr>

  <tr>
    <td>Ascend NPU driver</td>
    <td rowspan="2"><a href="https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Debian&Software=cannToolKit">Driver and Firmware Installation Guide</a></td>
  </tr>
  <tr>
    <td>Ascend NPU firmware</td>
  </tr>
  <tr>
    <td>Toolkit (development suite)</td>
    <td rowspan="3"><a href="https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0008.html?Mode=PmIns&OS=Debian&Software=cannToolKit">CANN Software Installation Guide</a></td>
  </tr>
  <tr>
    <td>ops (operator package)</td>
  </tr>
  <tr>
    <td>NNAL (Ascend Transformer Boost acceleration library)</td>
  </tr>
  <tr>
    <td>MindSpore</td>
    <td rowspan="1">*<a href="https://gitee.com/mindspore/mindspore#%E5%AE%89%E8%A3%85">MindSpore AI Framework Installation Guide</a>*</td>
  </tr>
</table>

### Obtaining the [MindSpeed-Core-MS](https://gitcode.com/Ascend/MindSpeed-Core-MS/) Code Repository

Run the following command to pull the MindSpeed-Core-MS code repository and install the Python third-party dependency libraries, as shown below:

```shell
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master
cd MindSpeed-Core-MS
pip install -r requirements.txt
```

You can refer to the [one-click adaptation command script](https://gitcode.com/Ascend/MindSpeed-Core-MS/#one-click-adaptation) provided in the MindSpeed-Core-MS directory to pull and adapt the corresponding versions of MindSpeed, Megatron-LM, and MSAdapter.

If you use the one-click adaptation command script in the MindSpeed-Core-MS directory (such as [auto_convert.sh](https://gitcode.com/Ascend/MindSpeed-Core-MS/blob/master/auto_convert.sh)), you can skip the subsequent steps.

### Obtaining and Adapting the Corresponding Versions of MindSpeed, Megatron-LM, and MSAdapter

1. After entering the MindSpeed-Core-MS directory, obtain the source code of the specified version repository:

   ```shell
   # Obtain the source code of the specified version of MindSpeed:
   git clone https://gitcode.com/Ascend/MindSpeed.git -b master

   # Obtain the source code of the specified version of Megatron-LM:
   git clone https://gitee.com/mirrors/Megatron-LM.git -b core_v0.12.1

   # Obtain the source code of the specified version of MSAdapter:
   git clone https://openi.pcl.ac.cn/OpenI/MSAdapter.git -b master
   ```

   For specific version mapping, refer to the [one-click adaptation command script](https://gitcode.com/Ascend/MindSpeed-Core-MS/#one-click-adaptation) under MindSpeed-Core-MS, such as [auto_convert.sh](https://gitcode.com/Ascend/MindSpeed-Core-MS/blob/master/auto_convert.sh).

2. Set environment variables:

   ```shell
   # Execute in the MindSpeed-Core-MS directory
   # If environment variables such as PYTHONPATH become invalid in the environment (for example, after exiting and re-entering the container), you need to set the environment variables again
   MindSpeed_Core_MS_PATH=$(pwd)
   export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MSAdapter:${MindSpeed_Core_MS_PATH}/MSAdapter/msa_thirdparty:${MindSpeed_Core_MS_PATH}/MindSpeed:$PYTHONPATH
   echo $PYTHONPATH
   ```

3. If you need to use Ascend Transformer Boost (ATB) acceleration library operators, install CANN-NNAL first and initialize the environment, for example:

   ```shell
   # The default installation path of CANN-NNAL is: /usr/local/Ascend/nnal
   # Run the environment configuration script set_env.sh in the atb folder under the default CANN-NNAL installation path
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

## Quick Start

1. You can easily enable various MindSpeed features with just one line of code. Taking the GPT model as an example: modify the `pretrain_gpt.py` file in the Megatron-LM directory, and add a new line under `import torch`: `import mindspeed.megatron_adaptor`, as shown in the following modification:

    ```diff
     import os
     import torch
    +import mindspeed.megatron_adaptor
     from functools import partial
     from typing import Union
    ```

2. (Optional) If the corresponding training data is not ready, you need to download and process the dataset for subsequent use. For the dataset preparation process, refer to
<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/Mindspeedguide/mindspeed_0003.html">Dataset Processing</a>.

3. In the Megatron-LM directory, prepare the training data, fill in the corresponding paths in the example script, and then execute it. The following examples are for reference.

    ```shell
    bash ./train_distributed_ms.sh
    ```

---

## Custom Optimization Level

MindSpeed provides multi-level optimization solutions, divided into three levels. Users can flexibly enable any level based on actual requirements. Higher levels are compatible with the capabilities of lower levels, ensuring the stability and scalability of the entire system.
Users can customize the optimization level to enable by setting the `--optimization-level {level}` parameter in the launch script. This parameter supports the following configurations:

<table><thead>
  <tr>
    <th width='50'>Level</th>
    <th width='300'>Name</th>
    <th width='600'>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5" style="text-align: center; vertical-align: middle"> 0 </td>
    <td>Basic compatibility layer</td>
    <td>Provides Megatron-LM framework support for NPU, ensuring seamless integration. This layer includes a basic feature set patch to guarantee reliability and stability, laying the foundation for advanced optimization.</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5" style="text-align: center; vertical-align: middle"> 1 </td>
    <td>Affinity enhancement layer🔥</td>
    <td>Compatible with L0 capabilities, integrates high-performance fused operator libraries, and combines Ascend-affinity computation optimizations to fully unleash Ascend computing power and significantly improve computational efficiency.</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5" style="text-align: center; vertical-align: middle"> 2 </td>
    <td>Acceleration algorithm layer🔥🔥</td>
    <td>Default value. This mode is compatible with L1 and L0 capabilities, and integrates multiple core self-developed technical achievements of Ascend, providing comprehensive performance optimization.</td>
  </tr>
</tbody>
</table>

## Data Profiling Data in MindSpeed

📝 MindSpeed supports command-based data profiling. The configuration commands are described as follows:

| Command Configuration                    | Command Description                                                                              |
|-------------------------|-----------------------------------------------------------------------------------|
| --profile               | Turns on the profile switch                                                                       |
| --profile-step-start    | Configures the start step for profiling. When not configured, defaults to 10. Configuration example: `--profile-step-start 30`                                 |
| --profile-step-end      | Configures the end step for profiling. When not configured, defaults to 12. Configuration example: `--profile-step-end 35`                                   |
| --profile-level         | Configures the profiling level. When not configured, defaults to level0. Optional configurations: level0, level1, level2. Configuration example: `--profile-level level1` |
| --profile-with-cpu      | Turns on the CPU information profiling switch                                                                       |
| --profile-with-stack    | Turns on the stack information profiling switch                                                                     |
| --profile-with-memory   | Turns on the memory information profiling switch. When configuring this switch, `--profile-with-cpu` must be enabled                                       |
| --profile-record-shapes | Turns on the shapes information profiling switch                                                                    |
| --profile-save-path     | Configures the save path for collected information. When not configured, defaults to `./profile_dir.` Configuration example: `--profile-save-path ./result_dir`          |
| --profile-ranks         | Configures the ranks to be collected. When not configured, defaults to `-1`, indicating that profiling data for all ranks is collected. Configuration example: `--profile-ranks 0 1 2 3`. Note: This configuration value is the global value for each rank in a single node/cluster   |
