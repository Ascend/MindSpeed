# Installation Guide

This document explains how to quickly install MindSpeed Core, the LLM training acceleration library, with the PyTorch framework.

## Hardware and Supported OSs

**Table 1** Product hardware support

| Product | Supported for Training |
|--|:-:|
|<term>Ascend 950PR&950DT products</term>|√|
|<term>Atlas A3 training products</term>|√|
|<term>Atlas A3 inference products</term>|x|
|<term>Atlas A2 training products</term>|√|
|<term>Atlas A2 inference products</term>|x|
|<term>Atlas 200I/500 A2 inference products</term>|x|
|<term>Atlas inference products</term>|x|
|<term>Atlas training products</term>|x|

> [!NOTE]
>
> The "√" in the table indicates support, and "x" indicates no support.

<!-- - For the OS supported by each hardware product in physical machine deployment scenarios, see the [Compatibility Query Assistant](<>). -->

For the OSs supported by each hardware product in virtual machine and container deployment scenarios, see the "OS Compatibility" section in [CANN Quick Installation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum).

## Preparation Before Installation

See [Related Product Version Mapping](../release_notes_core.md#related-product-version-mapping) in the Release Notes to download and install the corresponding software version.

Click [Driver and Firmware](https://hiascend.com/en/hardware/firmware-drivers) to install driver and firmware as prompted.

> [!NOTE]
>
> It is recommended to use a non-root user for installation and execution, and to enforce proper permission controls on the installation directory: set folder permissions to 750 and file permissions to 640. You can control the permissions of installed files by setting the umask, e.g., `umask 0027`.
> For more security-related information, please refer to the "File Permission Control" section for each component in the [Security Statement](../SECURITYNOTE.md).

## MindSpeed Installation

### Method 1: Image Installation

> [!NOTE]
>
> - Before using an image, confirm the machine model. The latest images support both AArch64 and X86_64 architectures. Run the `uname -a` command to check whether the current environment meets the requirements.
> - The matching images have CANN 9.2.0 and TorchNPU 26.2.0 preinstalled. Select an image as required.
> - If your current environment is incompatible with the provided image, choose [Method 2: Installation from Source](#method-2-installation-from-source).
> - The master branch will have new images updated in the future. If you need to build custom images, please refer to [Image Overview](../../../docker/OVERVIEW.md).

1. Pull the image.

   The latest image bundles the [26.1.0_core_r0.12.1 branch of MindSpeed Core](https://gitcode.com/Ascend/MindSpeed/tree/26.1.0_core_r0.12.1). Pull the image as needed from [Ascend Hub](https://www.hiascend.com/en/developer/ascendhub/detail/4ad248a439a44b4bb72e0534bfda8e2a).
   <!-- npu="950" id5 -->
   - <term>Ascend 950PR&950DT products</term>: v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-950-openeuler24.03-py3.12

   - <term>Ascend 950PR&950DT products</term>: v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-950-ubuntu22.04-py3.12
   <!-- end id5 -->
   <!-- npu="A3" id4 -->
   - <term>Atlas A3 training products</term>: v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.12

   - <term>Atlas A3 training products</term>: v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-ubuntu22.04-py3.12
   <!-- end id4 -->
   <!-- npu="910b" id3 -->
   - <term>Atlas A2 training products</term>: v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12

   - <term>Atlas A2 training products</term>: v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.12
   <!-- end id3 -->
   Take the image v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.12 as an example:

   ```bash
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-core:v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.12
   ```

2. Check whether the image is pulled successfully.

   ```bash
   docker image list
   ```

3. Run the image.

   Before copying the startup command, replace the two paths `{path-to-data}` and `{path-to-weights}` in the `-v` parameters with the actual local host directories. Otherwise, the container will fail to start.

   ```bash
   docker run -it -d \
      --name mindspeed-core \
      --privileged \
      --network host \
      --ipc=host \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v {path-to-data}:/data \
      -v {path-to-weights}:/weights \
      swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-core:v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.12 \
      /bin/bash
   ```

4. Load the container and verify the environment status.

   ```bash
    # Enter the container
    docker exec -it container name bash
    # Check whether the NPU is available
    npu-smi info
   ```

### Method 2: Installation from Source

1. Install CANN.

   Install the matching NPU driver/firmware and CANN (Toolkit, ops, and NNAL) and configure the CANN environment variables. For details, see [CANN Quick Installation](https://www.hiascend.com/en/cann/download?versionId=793&ids=d806%2Ch0501%2Ch0601%2Ch0703&currentTab=0).

   The CANN software provides a script that sets process-level environment variables. Before you run service code on an NPU in training or inference scenarios, source this script. Otherwise, the service code cannot run.

   ```shell
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

   The preceding commands use the default paths after a root installation. Replace them with the actual path to `set_env.sh`.

2. Install PyTorch and TorchNPU.

   See [TorchNPU Quick Installation](https://www.hiascend.com/en/developer/software/ai-frameworks/pytorch/download?versionId=180&ids=89dda9ba9de741349efa03687a487678%2C204%2C200%2C1%2C6%2C177%2C) to obtain the matching PyTorch and TorchNPU packages.

   >[!NOTE]
   >
   > If you have an older version of MindSpeed installed, please [uninstall](#uninstallation) the old version before installing the new one.
   >
   > For more TorchNPU plugin versions, see [GitCode](https://gitcode.com/ascend/pytorch/releases).

3. Clone the master branch of MindSpeed (note the case sensitivity of the following commands).

      ```shell
        git clone https://gitcode.com/Ascend/MindSpeed.git
        cd MindSpeed
        git checkout 26.2.0_core_r0.12.1
        cd ..
      ```

4. Install MindSpeed.

      ```shell
      pip install -e MindSpeed
      ```

5. Obtain the Megatron-LM source code and switch to version `core_v0.12.1`.

      The procedure is as follows:

      ```shell
      git clone https://github.com/NVIDIA/Megatron-LM.git
      cd Megatron-LM
      git checkout core_v0.12.1
      cd ..
      ```

6. Install Triton-Ascend

   To install the matching version of Triton-Ascend, see the [Triton-Ascend Installation Guide](https://triton-ascend.readthedocs.io/en/latest/installation_guide.html#piptriton-ascend) to obtain the installation command for the matching Triton-Ascend version.

   You can use the following installation command:

   ```shell
   # Note: For triton-ascend 3.2.0 and earlier, Triton-Ascend and Triton cannot coexist. You need to uninstall the community Triton first, and then install Triton-Ascend.
   pip install triton-ascend==3.6.0 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
   ```

## Uninstallation

Run the following command to uninstall MindSpeed.

```shell
pip uninstall -y mindspeed # Note that the command uses lowercase mindspeed
```
