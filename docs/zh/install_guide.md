# 软件安装

本文主要向用户介绍如何快速基于PyTorch框架以及MindSpore框架完成MindSpeed Core（亲和加速模块）的安装。

## 安装前准备

请参见《版本说明》中的“[相关产品版本配套说明](./release_notes.md#相关产品版本配套说明)”章节，下载安装对应的软件版本。

### 硬件配套和支持的操作系统

**表 1**  产品硬件支持列表

|产品|是否支持（训练场景）|
|--|:-:|
|<term>Atlas A3 训练系列产品</term>|√|
|<term>Atlas A3 推理系列产品</term>|x|
|<term>Atlas A2 训练系列产品</term>|√|
|<term>Atlas A2 推理系列产品</term>|x|
|<term>Atlas 200I/500 A2 推理产品</term>|x|
|<term>Atlas 推理系列产品</term>|x|
|<term>Atlas 训练系列产品</term>|√|

> [!NOTE]  
> 本节表格中“√”代表支持，“x”代表不支持。

- 各硬件产品对应物理机部署场景支持的操作系统请参考[兼容性查询助手](https://www.hiascend.com/hardware/compatibility)。

- 各硬件产品对应虚拟机部署场景支持的操作系统请参考《CANN 软件安装指南》的“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0101.html?Mode=VmIns&InstallType=local&OS=openEuler)”章节（商用版）或“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0101.html?Mode=VmIns&InstallType=local&OS=openEuler)”章节（社区版）。

- 各硬件产品对应容器部署场景支持的操作系统请参考《CANN 软件安装指南》的“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0101.html?Mode=DockerIns&InstallType=local&OS=openEuler)”章节（商用版）或“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0101.html?Mode=DockerIns&InstallType=local&OS=openEuler)”章节（社区版）。

### 安装驱动固件

下载[驱动固件](https://www.hiascend.com/hardware/firmware-drivers/community)，请根据系统和硬件产品型号选择对应版本的社区版本或商用版本的驱动与固件。执行以下命令安装：

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

更多驱动与固件的详细信息请参考《CANN软件安装指南》中的“[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=openEuler)”章节（商用版）或“[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=netconda&OS=openEuler)”章节（社区版），安装NPU驱动和固件。

### 安装CANN

获取[CANN](https://www.hiascend.com/cann/download)，安装配套版本的Toolkit、ops和NNAL并配置CANN环境变量。具体请参考《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html)》（商用版）或《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html)》（社区版）。

```shell
#基于PyTorch框架设置环境变量
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
source /usr/local/Ascend/nnal/atb/set_env.sh # 修改为实际安装的nnal包路径
```

```shell
#基于MindSpore框架设置环境变量
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0 # 修改为实际安装的nnal包路径
```

> [!NOTICE]  
> 建议使用非root用户安装运行torch\_npu，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。
> 更多安全相关内容请参见《[安全声明](SECURITYNOTE.md)》中各组件关于“文件权限控制”的说明。

## 基于PyTorch框架

### 安装PyTorch以及torch_npu

请参考《Ascend Extension for PyTorch 软件安装指南》中的“[安装PyTorch框架](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)”章节，获取配套版本的PyTorch以及torch_npu软件包。
可参考如下安装命令：

```shell
# 安装torch和torch_npu 构建参考 https://gitcode.com/ascend/pytorch/releases
pip3 install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl 
pip3 install torch_npu-2.7.1rc1-cp310-cp310-manylinux_2_28_aarch64.whl
```

### 安装MindSpeed

   1. 下载MindSpeed源码master分支（请注意下列命令的大小写）
   
      ```shell  
        git clone https://gitcode.com/Ascend/MindSpeed.git
      ```

   2. 安装MindSpeed

      ```shell
      pip install -e MindSpeed
      ```

      如有旧版本MindSpeed，请先[卸载](#卸载mindspeed)旧版本MindSpeed，再安装新版本MindSpeed。

   3. 获取Megatron-LM源码切换 core_v0.12.1 版本

       具体操作如下所示：
  
        ```shell
        git clone https://github.com/NVIDIA/Megatron-LM.git
        cd Megatron-LM
        git checkout core_v0.12.1
        ```

## 基于MindSpore框架

### 安装MindSpore框架 

参考[MindSpore官方安装指导](https://www.mindspore.cn/install)，根据系统类型、CANN版本及Python版本选择相应的安装命令进行安装，安装前请确保网络畅通。

### 安装MindSpeed-Core-MS

针对MindSpore，我们提供了一键转换工具MindSpeed-Core-MS，旨在帮助用户自动获取相关源码并对torch代码一键适配。

+ 一键安装 
 
  执行以下命令获取MindSpeed-Core-MS代码仓，并安装第三方依赖，如下所示：

  ```shell
  git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master
  cd MindSpeed-Core-MS
  pip install -r requirements.txt
  ```

  > [!NOTICE]  
  > 若使用MindSpeed-Core-MS目录下的一键适配命令脚本（如[auto_convert.sh](https://gitcode.com/Ascend/MindSpeed-Core-MS/blob/master/auto_convert.sh)）可忽略后面步骤。

+ 手动安装

  参考如下步骤完成手动安装。
  
1. 获取源码

    ```shell
    #获取指定版本的MindSpeed-Core-MS源码
    git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master
    cd MindSpeed-Core-MS

    # 在MindSpeed-Core-MS目录下，获取指定版本的MindSpeed，Megatron-LM和MSAdapter源码
    git clone https://gitcode.com/Ascend/MindSpeed.git -b master
    git clone https://gitee.com/mirrors/Megatron-LM.git -b core_v0.12.1
    git clone https://openi.pcl.ac.cn/OpenI/MSAdapter.git -b master
    ```

    如有旧版本MindSpeed，请先[卸载](#卸载mindspeed)旧版本MindSpeed，再安装新版本MindSpeed。

2. 设置环境变量

    ```shell
    # 若环境中，PYTHONPATH等环境变量失效（例如退出容器后再进入等），需要重新设置环境变量
    MindSpeed_Core_MS_PATH=$(pwd)
    export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MSAdapter:${MindSpeed_Core_MS_PATH}/MSAdapter/msa_thirdparty:${MindSpeed_Core_MS_PATH}/MindSpeed:$PYTHONPATH
    echo $PYTHONPATH
    ```
  
3. 设置CANN环境变量

   ```shell
    # NNAL默认安装路径为：/usr/local/Ascend/nnal
    # 运行NNAL默认安装路径下atb文件夹中的环境配置脚本set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    source /usr/local/Ascend/cann/set_env.sh
   ```

## 卸载MindSpeed

执行以下命令卸载MindSpeed。

```shell
pip uninstall -y mindspeed
```
