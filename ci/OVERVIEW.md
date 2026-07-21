# MindSpeed CI Docker Overview

## Quick Reference

| Item | Description |
| ------ | ------ |
| Image Name | `mindspeed-ci` |
| Repository | [https://gitcode.com/Ascend/MindSpeed](https://gitcode.com/Ascend/MindSpeed) |
| Dockerfile Path | `ci/Dockerfile` |
| Default Scenario | MindSpeed CI testing (includes MindSpeed-LLM, vLLM, vLLM-ascend, verl for ST tests) |
| Base Image | Configurable CANN image, default `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.1-910b-openeuler24.03-py3.12` |
| Default Working Directory | `/home/ci_deps/MindSpeed` |

## Key Fields in the Image Tag

Recommended tag template:

`{version}-{chip_info}-{os}-{python_tag}-{arch}`

`chip_info` must use lowercase values in image tags and CANN base image names, for example `a3` and `910b`. A full `--base-image` value is passed through unchanged, so its tag must match the published CANN image name exactly.

Examples:

- `master-910b-openeuler24.03-py3.12-aarch64`
- `master-a3-openeuler24.03-py3.12-aarch64`

## Dockerfile Archive Path

- `ci/Dockerfile`
- `ci/build.sh`
- `ci/configure_apt_repo.sh`
- `ci/configure_yum_repo.sh`
- `ci/configure_repo.sh`

## Included Components

The CI image contains additional components beyond MindSpeed Core for integration and system testing:

| Component | Default Version | Path in Image |
| ------ | ------ | ------ |
| CANN (Base Image) | 9.0.1 | `/usr/local/Ascend` |
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

## Build Options

`ci/build.sh` is the recommended build entry point. It supports selecting the CANN base image by OS, NPU type, Python tag, and CANN base image version, as well as versions for all included components.

| Option | Description | Default |
| ------ | ------ | ------ |
| `-t, --npu-type` | NPU type: `a3` or `910b` | `910b` |
| `-o, --os` | Operating system: `openeuler24.03` or `ubuntu22.04` | `openeuler24.03` |
| `-i, --image-name` | Image name (default: auto-generated from branch + config) | auto |
| `-n, --no-cache` | Build without cache | off |
| `--base-image-version` | CANN base image version | `9.0.1` |
| `--base-image` | Full CANN base image name, higher priority than `--base-image-version`; passed through unchanged | empty |
| `--python-version` | Python tag in the CANN base image | `3.12` |
| `--torch-version` | PyTorch version | `2.9.0` |
| `--torch-npu-version` | torch_npu version | `2.9.0` |
| `--mindspeed-branch` | MindSpeed branch/tag/ref to clone | `master` |
| `--megatron-branch` | Megatron-LM branch/tag/ref to checkout | `core_v0.12.1` |
| `--mindspeed-llm-branch` | MindSpeed-LLM branch/tag/ref to checkout | `master` |
| `--vllm-version` | vLLM version | `v0.18.0` |
| `--vllm-ascend-version` | vLLM-ascend version | `releases/v0.18.0` |
| `--verl-version` | verl version | `v0.7.0` |
| `--cleanup-on-fail` | Clean dangling images/containers if build fails | off |

## Quick Start

Build with defaults:

```bash
cd ci
bash build.sh
```

Build 910b + openEuler with a CANN 9.0.1 base image:

```bash
cd ci
bash build.sh -t 910b -o openeuler24.03
```

Build from a full base image name. The script auto-detects NPU type, OS, and Python version from the tag when possible:

```bash
cd ci
bash build.sh \
  --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.1-910b-openeuler24.03-py3.12
```

Build with custom component versions:

```bash
cd ci
bash build.sh \
  --mindspeed-branch master \
  --vllm-version v0.18.0 \
  --verl-version v0.7.0
```

> **Note:** CI environment does not permit privileged containers. Use the `--device` + `--cap-add` pattern below instead of `--privileged`.

Run example:

Use the `REPOSITORY:TAG` value from `docker images`, such as `mindspeed-ci:master-910b-openeuler24.03-py3.12-aarch64`.

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

Enter the running container:

```bash
docker exec -it mindspeed-ci /bin/bash
```

## Key Implementation Details

### verl ↔ vLLM API Compatibility

verl v0.7.0 has API incompatibilities with vLLM v0.18.0. The Dockerfile applies patches during build:

- `WorkerWrapperBase` constructor signature changed — patched from `WorkerWrapperBase(vllm_config=self.vllm_config)` to `WorkerWrapperBase(rpc_rank=0)`
- `execute_method` method removed — patched to use `getattr(self.inference_engine, method)(...)` instead

### triton-ascend Version

verl v0.7.0's `requirements-npu.txt` pins `triton-ascend==3.2.0rc4`, which is unavailable in the mirror. The Dockerfile patches it to `triton-ascend==3.2.1`.

### mbridge

verl requires mbridge (Megatron bridge) at runtime. It is installed after verl to avoid dependency conflicts.

### transformers Version Pinning

transformers is pinned to 4.57.1 in the final `RUN` step to prevent any intermediate installation from upgrading it.

### flash_attn Namespace Patches

MindSpeed's `requirements_basic.py` registers dummy `flash_attn` namespace packages to prevent conflicts with vLLM's `find_spec` detection:

- `flash_attn.flash_attn_interface.flash_attn_unpadded_func`
- `flash_attn.ops.triton.rotary.apply_rotary`

## Compatibility Notes

- This image uses a unified Dockerfile and build script for configurable CANN base image selection.
- The default base image uses CANN 9.0.1, 910b, openEuler 24.03, and Python 3.12.
- You can switch to Ubuntu 22.04, a3, or a different CANN base image version through `ci/build.sh`.
- MindSpeed, Megatron-LM, MindSpeed-LLM, vLLM, vLLM-ascend, and verl are cloned to `/home/ci_deps/`.
- Environment variables `ENABLE_ATB=1` and `PYTHONPATH` are pre-configured for NPU development.
- This image is intended for CI/testing scenarios, not production deployment.

## License

MindSpeed is released under the Apache License 2.0. See the [LICENSE](https://gitcode.com/Ascend/MindSpeed/blob/master/LICENSE) file for details.

Like all Docker images, these images may also contain other software under other licenses, such as Bash from the base distribution and any direct or indirect dependencies of the included main software.

For any use of pre-built images, it is the responsibility of the image user to ensure that any use of this image complies with the relevant licenses of all software contained therein.
