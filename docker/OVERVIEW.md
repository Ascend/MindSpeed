# MindSpeed Core Docker Image Overview

## Quick Reference

| Name | Description |
| ------ | ------ |
| Image name | `mindspeed-core` |
| Source repository | [https://gitcode.com/Ascend/MindSpeed](https://gitcode.com/Ascend/MindSpeed) |
| Dockerfile path | `docker/Dockerfile` |
| Default scenario | MindSpeed Core training and development |
| Base image | Configurable CANN image, defaulted to `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.11` |
| Default working directory | `/MindSpeed` |

## Key Field Description of Image Tags

Recommended tag template:

`{mindspeed_version}-cann{cann_version}-torch_npu{TorchNPU_version}-{npu_type}-{os}-py{python_version}`

The NPU type in the image tag and CANN base image name must be in lowercase: `a3`, `910b`, or `950`. The complete `--base-image` is passed through as-is, so its tag must exactly match the published CANN image name.

Examples:

- `v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.11`
- `v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11`

## Latest CANN 9.1.0

All images that support the latest CANN 9.1.0 release are listed below.

| Tag | Dockerfile | Content |
| ------ | ------ | ------ |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-ubuntu22.04-py3.11 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.11 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-950-openeuler24.03-py3.11 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |
| v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-950-ubuntu22.04-py3.11 | [Dockerfile](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/Dockerfile) | MindSpeed-Core/Megatron-LM |

> The latest tags in the table above are multi-architecture images (x86 and aarch64 combined). Once actually built, this Dockerfile generates tags with architecture suffixes such as `-aarch64` / `-x86_64` (for example, `v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11-aarch64`). For all tags of historical versions, see [Supported Tags](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docker/support_tags.md).

## Build Parameters

It is recommended to use `docker/build.sh` as the build entry point. The script supports selecting the CANN base image by operating system, NPU type, Python tag, CANN base image version, and target architecture.

| Name | Description | Default Value |
| ------ | ------ | ------ |
| `-t, --npu-type` | NPU type: `a3`, `910b`, or `950` | `910b` |
| `-o, --os` | Operating system: `openeuler24.03` or `ubuntu22.04` | `openeuler24.03` |
| `-a, --arch` | Target architecture: `aarch64` or `x86_64` | Current host architecture |
| `--base-image-version` | CANN base image version | `9.1.0` |
| `--base-image` | Full CANN base image name, which takes precedence over `--base-image-version`; it is passed through as-is | Empty |
| `--python-version` | Python tag in the CANN base image | `3.11` |
| `--torch-version` | PyTorch version | `2.7.1` |
| `--torch-npu-version` | TorchNPU version | `2.7.1.post8` |
| `--numpy-version` | NumPy version restored after all dependency installation steps | `1.26.0` |
| `--mindspeed-branch` | MindSpeed branch/tag/ref to clone | `v26.1.0_core_r0.12.1` |
| `--megatron-branch` | Megatron-LM branch/tag/ref to checkout | `core_v0.12.1` |
| `--image-version` | MindSpeed version field used in the default image tag | `v26.1.0_core_r0.12.1` |

If `http_proxy`, `https_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, or `no_proxy` is set on the host, the script automatically forwards it as a Docker build argument. These values are not persisted in the final image.

## Quick Start

Default:

```bash
cd docker
bash build.sh
```

Build the A3 + openEuler + aarch64 + CANN 9.1.0 base image:

```bash
cd docker
bash build.sh -t a3 -o openeuler24.03 -a aarch64
```

Build using the full base image name. The script will try to automatically identify the CANN version, NPU type, operating system, and Python version from the image tag:

```bash
cd docker
bash build.sh \
  --arch aarch64 \
  --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.11
```

Download the image:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-core:v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11
```

Run image:

Before copying the startup command below, replace the two paths `{path-to-data}` and `{path-to-weights}` in the parameters with the actual host paths for data and model weights. Otherwise, the container will fail to mount the host paths correctly after startup.

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
  swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-core:v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11 \
  bin/bash
```

Enter the started container:

```bash
docker exec -it mindspeed-core /bin/bash
```

## Compatibility Notes

- The current version uses a unified Dockerfile plus build script structure, supporting configurable CANN base image selection.
- The default base image uses CANN 9.1.0, 910B, openEuler 24.03, and Python 3.11.
- You can switch the operating system, NPU type (`a3`, `910b`, or `950`), target architecture, or CANN base image version via `docker/build.sh`.
- MindSpeed is cloned to `/MindSpeed`, and Megatron-LM is cloned to `/Megatron-LM`.
- The image installs PyTorch, TorchNPU, MindSpeed Core, Megatron-LM, and the Python dependencies in `requirements.txt`, then restores and verifies the configured PyTorch, TorchNPU, and NumPy versions.

## License

MindSpeed is released under the Apache License 2.0. For details, see the [LICENSE](https://gitcode.com/Ascend/MindSpeed/blob/master/LICENSE) file.

As with all Docker images, these images may also contain other software that is subject to other licenses (such as Bash from the base distribution, and any direct or indirect dependencies of the primary software being included).

For any use of pre-built images, it is the image user's responsibility to ensure that any use of this image complies with the relevant licenses of all software contained within it.

## Disclaimer

The released Ascend software images are community versions and are not intended for commercial accountability. They are provider solely as references for production practices.
