# Installation Guide

This document explains how to quickly install MindSpeed Core, the LLM training acceleration library, on the MindSpore framework.

## Hardware and Supported OSs

**Table 1** Product hardware support list

| Product | Supported for Training |
|--|:-:|
| <term>Atlas A3 training products</term> | √ |
| <term>Atlas A3 inference products</term> | x |
| <term>Atlas A2 training products</term> | √ |
| <term>Atlas A2 inference products</term> | x |
| <term>Atlas 200I/500 A2 inference products</term> | x |
| <term>Atlas inference products</term> | x |
| <term>Atlas training products</term> | x |

> [!NOTE]
> The "√" in the table indicates support, and "x" indicates no support.

- For the OSs supported by each hardware product in physical machine deployment scenarios, see the [Compatibility Query Assistant](https://www.hiascend.com/hardware/compatibility).

- For the OSs supported by each hardware product in VM and container deployment scenarios, see [OS Compatibility Description](https://www.hiascend.com/document/detail/en/canncommercial/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum) in the CANN Software Installation Guide for the commercial edition, or [OS Compatibility Description](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum) for the community edition.

## Installing Dependencies

See [Related Product Version Mapping](../release_notes_core.md#related-product-version-mapping) in the *Release Notes* to download and install the corresponding software version.

### Installing the NPU Driver and Firmware

Download the [Firmware and Drivers](https://www.hiascend.com/hardware/firmware-drivers/community). Choose the community or commercial firmware and driver package according to the OS and hardware model. Run the following commands to install them.

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

For more detailed information about the driver and firmware, see [Install the NPU Driver and Firmware](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=openEuler) in the CANN Software Installation Guide for the commercial edition, or [Install the NPU Driver and Firmware](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=netconda&OS=openEuler) for the community edition.

### Installing CANN

Refer to [CANN Quick Installation](https://www.hiascend.com/cann/download) to install the CANN software, including the Toolkit, ops, and NNAL packages, and configure the environment variables.

```shell
# Set environment variables based on the MindSpore framework
source /usr/local/Ascend/cann/set_env.sh  # Change this to the actual Toolkit installation path
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0  # Change this to the actual NNAL installation path
```

> [!NOTICE]
> You are advised to install and run MindSpeed-Core-MS as a non-root user. You are also advised to control permissions for the installer directory and files. Set directory permissions to 750 and file permissions to 640. You can control the permissions after installation by setting `umask`, for example `umask 0027`.
> For more security-related information, see the description of "file permission control" for each component in [Security Statement](../SECURITYNOTE.md).

### Installing MindSpeed

> [!NOTE]
>
> If you have an older version of MindSpeed, uninstall the old version of MindSpeed first, and then install the new version of MindSpeed.

1. Download the MindSpeed source code from the `26.0.0_core_r0.12.1` branch. Pay attention to the letter case in the following commands.

   ```shell
   git clone https://gitcode.com/Ascend/MindSpeed.git
   git checkout 26.0.0_core_r0.12.1
   ```

2. Install MindSpeed.

   ```shell
   pip install -e MindSpeed
   ```

3. Obtain the Megatron-LM source code and switch to version `core_v0.12.1`.

   Perform the following operations:

   ```shell
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout core_v0.12.1
   ```

## Installing the MindSpore Framework

Refer to the official MindSpore installation guide. Choose the installation command for MindSpore 2.9.0 based on the OS type, CANN version, and Python version. Ensure that network access is available before installation.

## Installing MindSpeed-Core-MS

For MindSpore, we provide a one-click conversion tool, MindSpeed-Core-MS, which helps users automatically obtain the relevant source code and adapt `torch` code with one click.

### One-Click Installation

1. Run the following commands to obtain the MindSpeed-Core-MS code repository and install third-party dependencies.

   ```shell
   git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master
   cd MindSpeed-Core-MS
   pip install -r requirements.txt
   ```

2. Use the one-click adaptation script [auto_convert.sh](https://gitcode.com/Ascend/MindSpeed-Core-MS/blob/master/auto_convert.sh) in the MindSpeed-Core-MS directory to complete the installation.

   ```bash
   bash auto_convert.sh
   ```

### Manual Installation

Complete the manual installation by following these steps.

1. Obtain the source code.

   ```shell
   # Obtain the source code of the specified version of MindSpeed-Core-MS
   git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master
   cd MindSpeed-Core-MS

   # In the MindSpeed-Core-MS directory, obtain the specified versions of MindSpeed, Megatron-LM, and MSAdapter
   git clone https://gitcode.com/Ascend/MindSpeed.git -b 26.0.0_core_r0.12.1
   git clone https://gitee.com/mirrors/Megatron-LM.git -b core_v0.12.1
   git clone https://openi.pcl.ac.cn/OpenI/MSAdapter.git -b master
   ```

   If you have an older version of MindSpeed, uninstall the old version of MindSpeed first, and then install the new version of MindSpeed.

2. Set environment variables.

   ```shell
   # If environment variables such as PYTHONPATH become invalid in the environment, for example after you exit and re-enter the container, set the environment variables again
   MindSpeed_Core_MS_PATH=$(pwd)
   export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MSAdapter:${MindSpeed_Core_MS_PATH}/MSAdapter/msa_thirdparty:${MindSpeed_Core_MS_PATH}/MindSpeed:$PYTHONPATH
   echo $PYTHONPATH
   ```

3. Set the CANN environment variables.

   ```shell
   # The default NNAL installation path is /usr/local/Ascend/nnal
   # Run the set_env.sh environment configuration script in the atb folder in the default NNAL installation path
   source /usr/local/Ascend/nnal/atb/set_env.sh
   source /usr/local/Ascend/cann/set_env.sh --cxx_abi=0
   ```

## Uninstallation

Run the following command to uninstall MindSpeed.

```shell
pip uninstall -y mindspeed  # The command name is lowercase mindspeed
```
