# MindSpeed Core Docker 镜像概述

## 快速参考

| 项目 | 说明 |
| ------ | ------ |
| 镜像名称 | `mindspeed-core` |
| 源码仓库 | [https://gitcode.com/Ascend/MindSpeed](https://gitcode.com/Ascend/MindSpeed) |
| Dockerfile 路径 | `docker/Dockerfile` |
| 默认场景 | MindSpeed Core 训练与开发 |
| 基础镜像 | 可配置 CANN 镜像，默认 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.11` |
| 默认工作目录 | `/MindSpeed` |

## 镜像 Tag 关键字段描述

推荐 Tag 模板：

`{MindSpeed版本}-cann{CANN版本}-torch_npu{TorchNPU版本}-{NPU类型}-{操作系统}-py{Python版本}-{架构类型}`

镜像 tag 和 CANN 基础镜像名中的“NPU 类型”必须使用小写：`a3`、`910b` 或 `950`。完整 `--base-image` 会原样传入，因此其中的 tag 必须与已发布的 CANN 镜像名完全一致。

示例：

- `v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.11-x86_64`
- `v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11-aarch64`

## 最新版本 CANN 9.1.0

如下所示是支持 CANN 最新发布的 9.1.0 版本的所有镜像，历史版本的所有 Tag 请参考 [Supported Tags](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/support_tags.md)。

| Tag | Dockerfile | Content |
| ------ | ------ | ------ |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11-aarch64 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11-x86_64 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-ubuntu22.04-py3.11-aarch64 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-ubuntu22.04-py3.11-x86_64 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11-aarch64 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11-x86_64 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.11-aarch64 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.11-x86_64 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-950-openeuler24.03-py3.11-aarch64 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-950-openeuler24.03-py3.11-x86_64 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-950-ubuntu22.04-py3.11-aarch64 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-950-ubuntu22.04-py3.11-x86_64 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |

## 构建参数

推荐使用 `docker/build.sh` 作为构建入口。脚本支持按操作系统、NPU 类型、Python 标签、CANN 基础镜像版本和目标架构选择基础镜像。

| 参数 | 说明 | 默认值 |
| ------ | ------ | ------ |
| `-t, --npu-type` | NPU 类型：`a3`、`910b` 或 `950` | `910b` |
| `-o, --os` | 操作系统：`openeuler24.03` 或 `ubuntu22.04` | `openeuler24.03` |
| `-a, --arch` | 目标架构：`aarch64` 或 `x86_64` | 当前宿主机架构 |
| `--base-image-version` | CANN 基础镜像版本 | `9.1.0` |
| `--base-image` | 完整 CANN 基础镜像名，优先级高于 `--base-image-version`；会原样传入 | 空 |
| `--python-version` | CANN 基础镜像中的 Python 标签 | `3.11` |
| `--torch-version` | PyTorch 版本 | `2.7.1` |
| `--torch-npu-version` | torch_npu 版本 | `2.7.1.post8` |
| `--numpy-version` | 所有依赖安装完成后恢复的 NumPy 版本 | `1.26.0` |
| `--mindspeed-branch` | 克隆 MindSpeed 使用的分支、标签或 ref | `v26.1.0_core_r0.12.1` |
| `--megatron-branch` | checkout Megatron-LM 使用的分支、标签或 ref | `core_v0.12.1` |
| `--image-version` | 默认镜像 tag 中使用的 MindSpeed 版本字段 | `v26.1.0_core_r0.12.1` |

宿主机设置了 `http_proxy`、`https_proxy`、`HTTP_PROXY`、`HTTPS_PROXY`、`NO_PROXY` 或 `no_proxy` 时，脚本会自动将其作为 Docker 构建参数转发；这些值不会保留在最终镜像中。

## 快速开始

默认构建：

```bash
cd docker
bash build.sh
```

构建 a3 + openEuler + aarch64 + CANN 9.1.0 基础镜像：

```bash
cd docker
bash build.sh -t a3 -o openeuler24.03 -a aarch64
```

使用完整基础镜像名构建。脚本会尽量从镜像 tag 自动识别 CANN 版本、NPU 类型、操作系统和 Python 版本：

```bash
cd docker
bash build.sh \
  --arch aarch64 \
  --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.11
```

下载镜像：

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-core:v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11-aarch64
```

运行镜像：

复制下方启动命令前，请将参数内的`{path-to-data}`、`{path-to-weights}`两处路径，替换为宿主机真实数据、模型权重存储路径，否则容器启动后无法正确挂载宿主机路径。

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
  swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-core:v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11-aarch64 \
  bin/bash
```

进入已启动容器：

```bash
docker exec -it mindspeed-core /bin/bash
```

## 兼容性说明

- 当前版本采用统一 Dockerfile + 构建脚本结构，支持可配置的 CANN 基础镜像选择。
- 默认基础镜像使用 CANN 9.1.0、910b、openEuler 24.03、Python 3.11。
- 可以通过 `docker/build.sh` 切换操作系统、NPU 类型（`a3`、`910b` 或 `950`）、目标架构或 CANN 基础镜像版本。
- MindSpeed 克隆到 `/MindSpeed`，Megatron-LM 克隆到 `/Megatron-LM`。
- 镜像安装 PyTorch、TorchNPU、MindSpeed Core、Megatron-LM 以及 `requirements.txt` 中的 Python 依赖，随后恢复并校验配置的 PyTorch、TorchNPU 和 NumPy 版本。

## 许可证

MindSpeed 基于 Apache License 2.0 许可证发布。详见 [LICENSE](https://gitcode.com/Ascend/MindSpeed/blob/master/LICENSE) 文件。

与所有 Docker 镜像一样，这些镜像可能还包含受其他许可证约束的其他软件（例如基础发行版中的 Bash，以及所包含主要软件的任何直接或间接依赖项）。

对于预构建镜像的任何使用，镜像用户有责任确保对此镜像的任何使用符合其中包含的所有软件的相关许可证。

## 免责声明

发布的昇腾软件镜像均是社区版本，不对商业负责、仅作为生产实践的参考。
