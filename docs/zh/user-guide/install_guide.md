# MindSpeed软件安装

本文主要向用户介绍如何快速基于PyTorch框架完成MindSpeed Core（大模型训练加速库）的安装。

## 硬件配套和支持的操作系统

**表 1**  产品硬件支持列表

|产品|是否支持（训练场景）|
|--|:-:|
|<term>Ascend 950PR&950DT系列产品</term>|√|
|<term>Atlas A3训练系列产品</term>|√|
|<term>Atlas A3推理系列产品</term>|x|
|<term>Atlas A2训练系列产品</term>|√|
|<term>Atlas A2推理系列产品</term>|x|
|<term>Atlas 200I/500 A2推理产品</term>|x|
|<term>Atlas推理系列产品</term>|x|
|<term>Atlas训练系列产品</term>|x|

> [!NOTE]
>
> 本节表格中“√”代表支持，“x”代表不支持。

- 各硬件产品对应物理机部署场景支持的操作系统请参考[兼容性查询助手](https://www.hiascend.com/hardware/compatibility)。

- 各硬件产品对应虚拟机及容器部署场景支持的操作系统请参考《CANN 软件安装》的“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)”章节。

## 安装前准备

请参见《版本说明》中的“[相关产品版本配套说明](../release_notes_core.md#相关产品版本配套说明)”章节，下载安装对应的软件版本。

请单击[固件与驱动](https://hiascend.com/hardware/firmware-drivers)，并根据引导完成固件与驱动的安装。

> [!NOTE]
>
> 安装运行程序建议使用非root用户，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。
> 更多安全相关内容请参见《[安全声明](../SECURITYNOTE.md)》中各组件关于“文件权限控制”的说明。

## 安装MindSpeed

### 方式一：镜像安装

> [!NOTE]
>
> - 使用镜像前，请先确认机器型号。最新镜像支持aarch64及X86_64架构，可通过uname -a命令确认当前环境是否符合要求。
> - 配套镜像已预装配套的CANN 9.1.0软件及TorchNPU 26.1.0插件，可根据需要选用。
> - 若当前环境与提供的镜像不兼容，请选择[方式二：源码安装](#方式二源码安装)。
> - master分支后续会更新新的镜像，如果需要自定义构建镜像请参见[镜像概述](../../../docker/OVERVIEW.zh.md)。

1. 获取镜像

   最新镜像均配套[MindSpeed Core的26.1.0_core_r0.12.1分支](https://gitcode.com/Ascend/MindSpeed/tree/26.1.0_core_r0.12.1)，请按需[获取镜像](https://www.hiascend.com/developer/ascendhub/detail/4ad248a439a44b4bb72e0534bfda8e2a)。
   <!-- npu="950" id5 -->
   - <term>Ascend 950PR&950DT系列产品</term>：v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-950-openeuler24.03-py3.12

   - <term>Ascend 950PR&950DT系列产品</term>：v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-950-ubuntu22.04-py3.12
   <!-- end id5 -->
   <!-- npu="A3" id4 -->
   - <term>Atlas A3训练系列产品</term>：v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.12

   - <term>Atlas A3训练系列产品</term>：v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-ubuntu22.04-py3.12
   <!-- end id4 -->
   <!-- npu="910b" id3 -->
   - <term>Atlas A2训练系列产品</term>：v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12

   - <term>Atlas A2训练系列产品</term>：v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.12
   <!-- end id3 -->
   以镜像v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.12为例：

   ```bash
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-core:v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.12
   ```

2. 确认是否成功拉取镜像

   ```bash
   docker image list
   ```

3. 运行镜像

   复制启动命令前，请将-v参数内的{path-to-data}、{path-to-weights}两处路径，替换为宿主机本地真实目录，否则容器启动失败。

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

4. 加载容器并确认环境状态

   ```bash
    # 加载容器
    docker exec -it 容器名 bash
    # 确认NPU是否可以正常使用
    npu-smi info
   ```

### 方式二：源码安装

1. 安装CANN

   安装配套版本的NPU驱动固件、CANN软件（Toolkit、ops和NNAL）并配置CANN环境变量，具体请参考《[CANN 快速安装](https://www.hiascend.com/cann/download?versionId=793&ids=d806%2Ch0501%2Ch0601%2Ch0703&currentTab=0)》。

   CANN软件提供进程级环境变量设置脚本，训练或推理场景下使用NPU执行业务代码前需要调用该脚本，否则业务代码将无法执行。

   ```shell
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

   以上命令以root用户安装后的默认路径为例，请用户根据set_env.sh的实际路径进行替换。

2. 安装PyTorch以及TorchNPU

   请参考《[TorchNPU 快速安装](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=180&ids=89dda9ba9de741349efa03687a487678%2C204%2C200%2C1%2C6%2C177%2C)》，获取配套版本的PyTorch以及TorchNPU软件包。

   >[!NOTE]
   >
   > 如有旧版本MindSpeed，请先[卸载](#卸载mindspeed)旧版本MindSpeed，再安装新版本MindSpeed。
   >
   >更多TorchNPU插件版本请单击[Link](https://gitcode.com/ascend/pytorch/releases)。

3. 安装MegatronAdaptor（MA）

   参考[MegatronAdaptor软件安装](https://gitcode.com/Ascend/MegatronAdaptor/blob/core_r0.18.0/docs/zh/install_guide.md)，获取配套分支源码并安装：

   ```shell
   git clone https://gitcode.com/Ascend/MegatronAdaptor.git
   cd MegatronAdaptor
   git checkout core_r0.18.0
   cd ..
   pip install -e MegatronAdaptor
   ```

4. 安装TransformerEngineNPU（TENPU）

   ```shell
   git clone --branch main https://gitcode.com/Ascend/TransformerEngineNPU.git
   pip install -e TransformerEngineNPU --no-build-isolation
   ```

   > [!NOTE]
   >
   > TENPU与原生TransformerEngine共用`transformer_engine`模块名，不能在同一环境中同时安装。若已安装原生TransformerEngine，请先执行`pip uninstall transformer_engine`，再安装TENPU。

5. 安装MindSpeed Core

   需要MindSpeed加速能力时，获取配套分支源码并安装。已有当前分支源码时，直接执行安装命令即可。

   ```shell
   git clone https://gitcode.com/Ascend/MindSpeed.git
   cd MindSpeed
   git checkout core_r0.18.0
   cd ..
   pip install -e MindSpeed
   ```

6. 按场景安装MindSpeed-Ops

   Megatron-LM的权重梯度融合累加接口在NPU上的实现由[MindSpeed-Ops](https://gitcode.com/Ascend/MindSpeed-Ops)的MatmulAdd算子提供，MA负责将该实现补丁到Megatron-LM。MindSpeed-Ops是独立安装包，不会随MindSpeed自动安装。

   | 使用场景 | 是否需要安装 |
   |---------|------------|
   | 默认参数训练（Megatron-LM默认开启`gradient_accumulation_fusion`，且模型包含Megatron原生张量并行Linear，如output layer） | 需要 |
   | 传入`--no-gradient-accumulation-fusion`的训练 | 不需要 |
   | 纯推理或评估（不执行反向传播） | 不需要 |

   默认参数训练请执行以下命令：

   ```shell
   git clone https://gitcode.com/Ascend/MindSpeed-Ops.git
   cd MindSpeed-Ops
   pip install -e . --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple --no-build-isolation --no-deps
   python -c "from mindspeed_ops.api.atb.npu_matmul_add import npu_matmul_add_fp32, npu_matmul_add_fp16; print('MindSpeed-Ops MatmulAdd API loaded successfully')"
   cd ..
   ```

   > [!NOTE]
   >
   > - 未安装MindSpeed-Ops时，安装MindSpeed、导入`megatron_adaptor`以及构建模型均不会报错；默认训练会在首次反向传播调用融合接口时显式报错。此时应安装MindSpeed-Ops，或增加`--no-gradient-accumulation-fusion`关闭融合路径。
   > <!-- npu="A3,910b" id1 -->
   > - <term>Atlas A2训练系列产品</term>、<term>Atlas A3训练系列产品</term>且主梯度为fp32时使用ATB JIT融合路径，需要第1步已安装CANN-NNAL并加载`nnal/atb/set_env.sh`，同时提供C++/Ninja编译工具链。首次调用会执行JIT编译，产物缓存后可复用。
   > <!-- end id1 -->
   ><!-- npu="950" id2 -->
   > - fp16/bf16主梯度和<term>Ascend 950PR&950DT系列产品</term>fp32场景使用`addmm_`路径，但接口仍由`mindspeed_ops`包提供；只要开启权重梯度融合，仍需安装MindSpeed-Ops。
   ><!-- end id2 -->
   > - 完整依赖和芯片编译说明请参见[MindSpeed-Ops软件安装](https://gitcode.com/Ascend/MindSpeed-Ops/blob/master/docs/zh/install_guide.md)。

7. 获取Megatron-LM源码并切换至0.18.0版本

   ```shell
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout core_v0.18.0
   cd ..
   ```

   Megatron-LM上游版本标签为`core_v0.18.0`，与MA配套分支`core_r0.18.0`的命名不同。安装后请参考[快速入门](quickstart.md#环境准备)准备数据并启动训练。

## 卸载MindSpeed

执行以下命令卸载MindSpeed。

```shell
pip uninstall -y mindspeed #注意命令中为小写mindspeed
```

## 卸载MindSpeed Ops

如已安装MindSpeed Ops，可执行以下命令卸载：

```shell
pip uninstall -y mindspeed_ops #注意命令中为小写mindspeed_ops
```
