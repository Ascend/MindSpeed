# MindSpeed CI Docker 镜像概述

## 快速参考

| 项目 | 说明 |
| ------ | ------ |
| 镜像名称 | `mindspeed-ci` |
| 源码仓库 | [https://gitcode.com/Ascend/MindSpeed](https://gitcode.com/Ascend/MindSpeed) |
| Dockerfile 路径 | `ci/Dockerfile` |
| 默认场景 | MindSpeed CI 测试（包含 MindSpeed-LLM、vLLM、vLLM-ascend、verl，用于 ST 测试） |
| 基础镜像 | 可配置 CANN 镜像，默认 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.1-910b-openeuler24.03-py3.12` |
| 默认工作目录 | `/home/ci_deps/MindSpeed` |

## 镜像 Tag 关键字段描述

推荐 Tag 模板：

`{版本号}-{芯片信息}-{操作系统}-{Python标签}-{架构类型}`

镜像 tag 和 CANN 基础镜像名中的"芯片信息"必须使用小写，例如 `a3` 和 `910b`。完整 `--base-image` 会原样传入，因此其中的 tag 必须与已发布的 CANN 镜像名完全一致。

示例：

- `master-910b-openeuler24.03-py3.12-aarch64`
- `master-a3-openeuler24.03-py3.12-aarch64`

## Dockerfile 归档路径

- `ci/Dockerfile`
- `ci/build.sh`
- `ci/configure_apt_repo.sh`
- `ci/configure_yum_repo.sh`
- `ci/configure_repo.sh`

## 包含的组件

CI 镜像在 MindSpeed Core 之上额外包含以下组件，用于集成测试和系统测试：

| 组件 | 默认版本 | 镜像内路径 |
| ------ | ------ | ------ |
| CANN (基础镜像) | 9.0.1 | `/usr/local/Ascend` |
| PyTorch | 2.9.0 | pip |
| torch_npu | 2.9.0 | pip |
| Megatron-LM | core_v0.12.1 | `/home/ci_deps/Megatron-LM` |
| MindSpeed | master | `/home/ci_deps/MindSpeed` |
| MindSpeed-LLM | master | `/home/ci_deps/MindSpeed-LLM` |
| vLLM | v0.18.0 | `/home/ci_deps/vllm` |
| vLLM-ascend | releases/v0.18.0 | `/home/ci_deps/vllm-ascend` |
| verl | v0.7.0 | `/home/ci_deps/verl` |
| mbridge | latest | pip |
| transformers | 4.57.1 | pip |

## 构建参数

推荐使用 `ci/build.sh` 作为构建入口。脚本支持按操作系统、NPU 类型、Python 标签和 CANN 基础镜像版本选择基础镜像，同时支持指定所有组件的版本。

| 参数 | 说明 | 默认值 |
| ------ | ------ | ------ |
| `-t, --npu-type` | NPU 类型：`a3` 或 `910b` | `910b` |
| `-o, --os` | 操作系统：`openeuler24.03` 或 `ubuntu22.04` | `openeuler24.03` |
| `-i, --image-name` | 镜像名称（默认根据分支和配置自动生成） | 自动 |
| `-n, --no-cache` | 不使用缓存构建 | 关闭 |
| `--base-image-version` | CANN 基础镜像版本 | `9.0.1` |
| `--base-image` | 完整 CANN 基础镜像名，优先级高于 `--base-image-version`；会原样传入 | 空 |
| `--python-version` | CANN 基础镜像中的 Python 标签 | `3.12` |
| `--torch-version` | PyTorch 版本 | `2.9.0` |
| `--torch-npu-version` | torch_npu 版本 | `2.9.0` |
| `--mindspeed-branch` | 克隆 MindSpeed 使用的分支、标签或 ref | `master` |
| `--megatron-branch` | checkout Megatron-LM 使用的分支、标签或 ref | `core_v0.12.1` |
| `--mindspeed-llm-branch` | checkout MindSpeed-LLM 使用的分支、标签或 ref | `master` |
| `--vllm-version` | vLLM 版本 | `v0.18.0` |
| `--vllm-ascend-version` | vLLM-ascend 版本 | `releases/v0.18.0` |
| `--verl-version` | verl 版本 | `v0.7.0` |
| `--cleanup-on-fail` | 构建失败时清理悬空镜像和对应容器 | 关闭 |

## 快速开始

默认构建：

```bash
cd ci
bash build.sh
```

构建 910b + openEuler + CANN 9.0.1 基础镜像：

```bash
cd ci
bash build.sh -t 910b -o openeuler24.03
```

使用完整基础镜像名构建。脚本会尽量从镜像 tag 自动识别 NPU 类型、操作系统和 Python 版本：

```bash
cd ci
bash build.sh \
  --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.1-910b-openeuler24.03-py3.12
```

指定自定义组件版本构建：

```bash
cd ci
bash build.sh \
  --mindspeed-branch master \
  --vllm-version v0.18.0 \
  --verl-version v0.7.0
```

> **注意：** CI 环境不允许使用特权容器，请使用下方的 `--device` + `--cap-add` 方式替代 `--privileged`。

运行示例：

镜像名使用 `docker images` 中的 `REPOSITORY:TAG`，例如 `mindspeed-ci:master-910b-openeuler24.03-py3.12-aarch64`。

```bash
docker run -itd \
  --name mindspeed-ci \
  --pid=host \
  --network host \
  --ipc host \
  --cgroupns host \
  --security-opt seccomp=unconfined \
  --cap-add=CAP_SYS_RESOURCE \
  --cap-add=CAP_SYS_ADMIN \
  --cap-add=CAP_MKNOD \
  --cap-add=CAP_SYS_PTRACE \
  --cap-add=CAP_IPC_LOCK \
  -e ASCEND_VISIBLE_DEVICES=0-7 \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --security-opt label=disable \
  --shm-size=32G \
  -v /home/ci_deps/models:/home/ci_deps/models \
  -v /home/dataset:/home/dataset \
  -v /home/ci_deps/MindSpeed:/home/ci_deps/MindSpeed \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin/:/usr/local/sbin/ \
  mindspeed-ci:master-910b-openeuler24.03-py3.12-aarch64
```

进入已启动容器：

```bash
docker exec -it mindspeed-ci /bin/bash
```

## 关键实现细节

### verl 与 vLLM API 兼容性

verl v0.7.0 与 vLLM v0.18.0 之间存在 API 不兼容。Dockerfile 在构建时应用了以下补丁：

- `WorkerWrapperBase` 构造函数签名变更 — 从 `WorkerWrapperBase(vllm_config=self.vllm_config)` 修改为 `WorkerWrapperBase(rpc_rank=0)`
- `execute_method` 方法已移除 — 修改为使用 `getattr(self.inference_engine, method)(...)` 替代

### triton-ascend 版本

verl v0.7.0 的 `requirements-npu.txt` 写死了 `triton-ascend==3.2.0rc4`，但镜像源中该版本不可用。Dockerfile 将其修改为 `triton-ascend==3.2.1`。

### mbridge

verl 运行时依赖 mbridge（Megatron bridge），在 verl 之后安装以避免依赖冲突。

### transformers 版本锁定

transformers 在最后一个 `RUN` 步骤中锁定为 4.57.1，防止任何中间安装过程将其升级。

### flash_attn 命名空间补丁

MindSpeed 的 `requirements_basic.py` 注册了虚拟 `flash_attn` 命名空间包，以避免与 vLLM 的 `find_spec` 检测冲突：

- `flash_attn.flash_attn_interface.flash_attn_unpadded_func`
- `flash_attn.ops.triton.rotary.apply_rotary`

## 兼容性说明

- 当前版本采用统一 Dockerfile + 构建脚本结构，支持可配置的 CANN 基础镜像选择。
- 默认基础镜像使用 CANN 9.0.1、910b、openEuler 24.03、Python 3.12。
- 可以通过 `ci/build.sh` 切换 Ubuntu 22.04、a3 或其他 CANN 基础镜像版本。
- MindSpeed、Megatron-LM、MindSpeed-LLM、vLLM、vLLM-ascend 和 verl 克隆到 `/home/ci_deps/`。
- 环境变量 `ENABLE_ATB=1` 和 `PYTHONPATH` 已预配置，用于 NPU 开发。
- 此镜像专用于 CI/测试场景，不适用于生产部署。

## 许可证

MindSpeed 基于 Apache License 2.0 许可证发布。详见 [LICENSE](https://gitcode.com/Ascend/MindSpeed/blob/master/LICENSE) 文件。

与所有 Docker 镜像一样，这些镜像可能还包含受其他许可证约束的其他软件（例如基础发行版中的 Bash，以及所包含主要软件的任何直接或间接依赖项）。

对于预构建镜像的任何使用，镜像用户有责任确保对此镜像的任何使用符合其中包含的所有软件的相关许可证。
