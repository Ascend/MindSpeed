# MindSpeed Core Docker 镜像概述

## 快速参考

| 项目 | 说明 |
| ------ | ------ |
| 镜像名称 | `mindspeed-core` |
| 源码仓库 | [https://gitcode.com/Ascend/MindSpeed](https://gitcode.com/Ascend/MindSpeed) |
| Dockerfile 路径 | `docker/Dockerfile` |
| 默认场景 | MindSpeed Core 训练与开发 |
| 基础镜像 | 可配置 CANN 镜像，默认 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.2-910b-openeuler24.03-py3.11` |
| 默认工作目录 | `/MindSpeed` |

## 镜像 Tag 关键字段描述

推荐 Tag 模板：

`{版本号}-{芯片信息}-{操作系统}-{Python标签}-{架构类型}`

镜像 tag 和 CANN 基础镜像名中的“芯片信息”必须使用小写，例如 `a3` 和 `910b`。完整 `--base-image` 会原样传入，因此其中的 tag 必须与已发布的 CANN 镜像名完全一致。

示例：

- `0.12.1-910b-ubuntu22.04-py3.11-x86_64`
- `master-a3-openeuler24.03-py3.11-aarch64`

## Dockerfile 归档路径

- `docker/Dockerfile`
- `docker/build.sh`

## 构建参数

推荐使用 `docker/build.sh` 作为构建入口。脚本支持按操作系统、NPU 类型、Python 标签和 CANN 基础镜像版本选择基础镜像。

| 参数 | 说明 | 默认值 |
| ------ | ------ | ------ |
| `-t, --npu-type` | NPU 类型：`a3` 或 `910b` | `910b` |
| `-o, --os` | 操作系统：`openeuler24.03` 或 `ubuntu22.04` | `openeuler24.03` |
| `--base-image-version` | CANN 基础镜像版本 | `9.0.0-beta.2` |
| `--base-image` | 完整 CANN 基础镜像名，优先级高于 `--base-image-version`；会原样传入 | 空 |
| `--python-version` | CANN 基础镜像中的 Python 标签 | `3.11` |
| `--torch-version` | PyTorch 版本 | `2.7.1` |
| `--torch-npu-version` | torch_npu 版本 | `2.7.1` |
| `--mindspeed-branch` | 克隆 MindSpeed 使用的分支、标签或 ref | `master` |
| `--megatron-branch` | checkout Megatron-LM 使用的分支、标签或 ref | `core_v0.12.1` |

## 快速开始

默认构建：

```bash
cd docker
bash build.sh
```

构建 a3 + openEuler + CANN 9.0 基础镜像：

```bash
cd docker
bash build.sh -t a3 -o openeuler24.03 --base-image-version 9.0.0-beta.2
```

使用完整基础镜像名构建。脚本会尽量从镜像 tag 自动识别 NPU 类型、操作系统和 Python 版本：

```bash
cd docker
bash build.sh \
  --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.2-910b-openeuler24.03-py3.11
```

运行示例：

镜像名使用 `docker images` 中的 `REPOSITORY:TAG`，例如 `mindspeed-core:master-910b-openeuler24.03-py3.11-aarch64`。

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
  mindspeed-core:master-910b-openeuler24.03-py3.11-aarch64
```

进入已启动容器：

```bash
docker exec -it mindspeed /bin/bash
```

如果宿主机的 `npu-smi` 安装在 `/usr/local/sbin/npu-smi`，请相应替换 `npu-smi` 挂载路径。

## 兼容性说明

- 当前版本采用统一 Dockerfile + 构建脚本结构，支持可配置的 CANN 基础镜像选择。
- 默认基础镜像使用 CANN 9.0.0-beta.2、910b、openEuler 24.03、Python 3.11。
- 可以通过 `docker/build.sh` 切换 Ubuntu 22.04、a3 或其他 CANN 基础镜像版本。
- MindSpeed 克隆到 `/MindSpeed`，Megatron-LM 克隆到 `/Megatron-LM`。
- 镜像安装 PyTorch、torch_npu、MindSpeed Core、Megatron-LM 以及 `requirements.txt` 中的 Python 依赖。

## 许可证

MindSpeed 基于 Apache License 2.0 许可证发布。详见 [LICENSE](../LICENSE) 文件。

与所有 Docker 镜像一样，这些镜像可能还包含受其他许可证约束的其他软件（例如基础发行版中的 Bash，以及所包含主要软件的任何直接或间接依赖项）。

对于预构建镜像的任何使用，镜像用户有责任确保对此镜像的任何使用符合其中包含的所有软件的相关许可证。
