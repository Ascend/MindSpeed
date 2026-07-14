# MindSpeed Core Docker Overview

## Quick Reference

| Item | Description |
| ------ | ------ |
| Image Name | `mindspeed-core` |
| Repository | [https://gitcode.com/Ascend/MindSpeed](https://gitcode.com/Ascend/MindSpeed) |
| Dockerfile Path | `docker/Dockerfile` |
| Default Scenario | MindSpeed Core training and development |
| Base Image | Configurable CANN image, default `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.2-910b-openeuler24.03-py3.11` |
| Default Working Directory | `/MindSpeed` |

## Key Fields in the Image Tag

Recommended tag template:

`{version}-{chip_info}-{os}-{python_tag}-{arch}`

`chip_info` must use lowercase values in image tags and CANN base image names, for example `a3` and `910b`. A full `--base-image` value is passed through unchanged, so its tag must match the published CANN image name exactly.

Examples:

- `0.12.1-910b-ubuntu22.04-py3.11-x86_64`
- `master-a3-openeuler24.03-py3.11-aarch64`

## Dockerfile Archive Path

- `docker/Dockerfile`
- `docker/build.sh`

## Build Options

`docker/build.sh` is the recommended build entry point. It supports selecting the CANN base image by OS, NPU type, Python tag, and CANN base image version.

| Option | Description | Default |
| ------ | ------ | ------ |
| `-t, --npu-type` | NPU type: `a3` or `910b` | `910b` |
| `-o, --os` | Operating system: `openeuler24.03` or `ubuntu22.04` | `openeuler24.03` |
| `--base-image-version` | CANN base image version | `9.0.0-beta.2` |
| `--base-image` | Full CANN base image name, higher priority than `--base-image-version`; passed through unchanged | empty |
| `--python-version` | Python tag in the CANN base image | `3.11` |
| `--torch-version` | PyTorch version | `2.7.1` |
| `--torch-npu-version` | TorchNPU version | `2.7.1` |
| `--mindspeed-branch` | MindSpeed branch/tag/ref to clone | `master` |
| `--megatron-branch` | Megatron-LM branch/tag/ref to checkout | `core_v0.12.1` |

## Quick Start

Build with defaults:

```bash
cd docker
bash build.sh
```

Build a3 + openEuler with a CANN 9.0 base image:

```bash
cd docker
bash build.sh -t a3 -o openeuler24.03 --base-image-version 9.0.0-beta.2
```

Build from a full base image name. The script auto-detects NPU type, OS, and Python version from the tag when possible:

```bash
cd docker
bash build.sh \
  --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.2-910b-openeuler24.03-py3.11
```

Pull image:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-core:26.0.0_core_r0.12.1-a3-openeuler24.03-py3.11-aarch64
```

Run image:

Use the `REPOSITORY:TAG` value from `docker images`, such as `swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-core:26.0.0_core_r0.12.1-a3-openeuler24.03-py3.11-aarch64`.

Before copying the startup command below, replace the `{path-to-data}` and `{path-to-weights}` placeholders with the actual paths on your host machine where data and model weights are stored. Otherwise, the container won't mount the host paths correctly after startup.

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
  swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-core:26.0.0_core_r0.12.1-a3-openeuler24.03-py3.11-aarch64 \
  bin/bash
```

Enter the running container:

```bash
docker exec -it mindspeed-core /bin/bash
```

## Compatibility Notes

- This image uses a unified Dockerfile and build script for configurable CANN base image selection.
- The default base image uses CANN 9.0.0-beta.2, 910b, openEuler 24.03, and Python 3.11.
- You can switch to Ubuntu 22.04, a3, or a different CANN base image version through `docker/build.sh`.
- MindSpeed is cloned to `/MindSpeed`; Megatron-LM is cloned to `/Megatron-LM`.
- It installs PyTorch, TorchNPU, MindSpeed Core, Megatron-LM, and Python dependencies from `requirements.txt`.

## License

MindSpeed is released under the Apache License 2.0. See the [LICENSE](https://gitcode.com/Ascend/MindSpeed/blob/master/LICENSE) file for details.

Like all Docker images, these images may also contain other software under other licenses, such as Bash from the base distribution and any direct or indirect dependencies of the included main software.

For any use of pre-built images, it is the responsibility of the image user to ensure that any use of this image complies with the relevant licenses of all software contained therein.
