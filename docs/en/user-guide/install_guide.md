# Installation Guide

This document explains how to quickly install MindSpeed Core, the LLM training acceleration library, with the PyTorch framework.

## Hardware and Supported OSs

**Table 1** Product hardware support

| Product | Supported for Training |
|--|:-:|
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

- For the OSs supported by each hardware product in physical machine deployment scenarios, see the [Compatibility Query Assistant](https://www.hiascend.com/hardware/compatibility).

- For the OSs supported by each hardware product in VM and container deployment scenarios, see the "OS Compatibility" section in [CANN Software Installation](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html) for the commercial edition or the "OS Compatibility" section in [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html) for the community edition.

## Preparation Before Installation

See [Related Product Version Mapping](../release_notes_core.md#related-product-version-mapping) in the Release Notes to download and install the corresponding software version.

> [!NOTICE]
> You are advised to use a non-root user to install and run the program, and you are advised to control permissions on the installation directory and files. Set directory permissions to 750 and file permissions to 640. You can control file permissions after installation by setting `umask`, for example, to 0027.
> For more security information, see the descriptions of "file permission control" for each component in [Security Statement](../SECURITYNOTE.md).

Download the [Firmware and Drivers](https://hiascend.com/hardware/firmware-drivers/community). Choose the community or commercial firmware and driver package according to the OS and hardware model. Run the following commands to install them.

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

## MindSpeed Installation

### Method 1: Image Installation

> [!NOTE]
>
> - Before you use the image, first confirm the machine model. The latest image supports only the `aarch64` architecture. Run the `uname -a` command to verify that the current environment meets the requirement.
> - The bundled image already includes CANN 9.0.0 software and the Ascend Extension for PyTorch 26.0.0 plugin. You can use it as needed.
> - If your current environment is incompatible with the provided image, choose [Method 2: Installation from Source](#method-2-installation-from-source).

1. Pull the image.

   The latest image bundles the [26.0.0_core_r0.12.1 branch of MindSpeed Core](https://gitcode.com/Ascend/MindSpeed/tree/26.0.0_core_r0.12.1). Pull the image as needed from [Ascend Hub](https://www.hiascend.com/developer/ascendhub/detail/4ad248a439a44b4bb72e0534bfda8e2a).

   - <term>Atlas A2 training products</term>: `26.0.0_core_r0.12.1-910b-openeuler24.03-py3.11-aarch64`

   - <term>Atlas A3 training products</term>: `26.0.0_core_r0.12.1-a3-openeuler24.03-py3.11-aarch64`

   ```bash
   # Check whether the image is pulled successfully
   docker image list
   ```

2. Create a container.

   ```bash
    # Mount the image
    docker run -dit --ipc=host --network host --name 'container name' --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ -v /data/:/data image name:tag /bin/bash
   ```

   By default, the driver and firmware are installed in `/usr/local/Ascend`. If the paths differ, modify the command paths.

   The container initializes the NPU driver and CANN environment information by default. If you need a different setup, replace it or source it manually. See `~/.bashrc` of the container for details.

    Example:

      ```bash
      docker run -itd \
         --name mindspeed \
         --privileged \
         --network host \
         --ipc=host \
         -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
         -v /usr/local/dcmi:/usr/local/dcmi \
         -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
         -v /etc/ascend_install.info:/etc/ascend_install.info \
         -v /home:/home \
         -v /data:/data \
         -v /mnt:/mnt \
         mindspeed-core:26.0.0_core_r0.12.1-a3-openeuler24.03-py3.11-aarch64
      ```

3. Load the container and verify the environment status.

   ```bash
    # Enter the container
    docker exec -it container name bash
    # Check whether the NPU is available
    npu-smi info
   ```

### Method 2: Installation from Source

1. Install CANN.

   Install the matching NPU driver firmware and CANN software, including Toolkit, ops, and NNAL, and configure the CANN environment variables. For details, see [CANN Software Installation](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html) for the commercial edition or [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html) for the community edition.

   The CANN software provides a script that sets process-level environment variables. Before you run service code on an NPU in training or inference scenarios, source this script. Otherwise, the service code cannot run.

   ```shell
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

   The preceding commands use the default paths after a root installation. Replace them with the actual path to `set_env.sh`.

2. Install PyTorch and `torch_npu`.

   See [Install PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md) to obtain the matching PyTorch and `torch_npu` packages.
   Use the following installation commands:

   ```shell
   # See https://gitcode.com/ascend/pytorch/releases for torch and torch_npu build references
   pip3 install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
   pip3 install torch_npu-2.7.1post4-cp310-cp310-manylinux_2_28_aarch64.whl
   ```

   >[!NOTE]
   >If an older version of MindSpeed is installed, first uninstall it, then install the new version.

3. Download the MindSpeed source code from the `26.0.0_core_r0.12.1` branch. Pay attention to the letter case in the following commands.

      ```shell
        git clone https://gitcode.com/Ascend/MindSpeed.git
        git checkout 26.0.0_core_r0.12.1
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
        ```

## Uninstallation

Run the following command to uninstall MindSpeed.

   ```shell
   # Note that the command uses lowercase mindspeed
   pip uninstall -y mindspeed
   ```
