# MindSpeed Ascend HDK Path Batch Replacement

## Background

The Docker mount configurations, run scripts, and other files in the MindSpeed repository contain hardcoded references to the `/usr/local/Ascend/driver/` path.
On some servers, the actual HDK installation path is `/usr/local/npu/driver/`, so a batch replacement must be completed before use to ensure that HDK-related mounts and library loading work properly.

> Note: This replacement **applies only to the HDK path** (`/usr/local/Ascend/driver/`). Other sub-paths such as CANN, Ascend Toolkit, and NNAL/ATB remain unchanged.

This guide provides the complete steps for batch path replacement using the `replace_ascend_path.py` script, and explains the adaptation requirements for the return value of the `dcmi_get_device_chip_info` interface on some versions.

---

## Prerequisites

- Python 3.10+
- Read and write permissions on the repository directory
- It is recommended to commit or back up the current state via git before performing the replacement

---

## Affected File Scope

| File Type | Description | Typical Path Example |
| --------- | ------ | ------------- |
| Shell scripts (`.sh`) | Various run scripts, including but not limited to data preprocessing, weight conversion, pre-training, fine-tuning, evaluation, inference, and testing workflows | `examples/*/*.sh`, `tests/*/*.sh` |
| Markdown documents (`.md`) | All documentation, including but not limited to installation guides, quick start guides, task-specific guides, and feature descriptions | `docs/en/user-guide/install_guide.md`, `docker/OVERVIEW.md` |
| RST documents (`.rst`) | reStructuredText-style documentation | `docs/*/*.rst` |
| TXT documents (`.txt`) | Plain text description files or configuration descriptions | `requirements.txt` |
| Python files (`.py`) | Source code (if path references exist) | Source files of each module |
| Dockerfile | Docker image build scripts | `docker/Dockerfile` |

> Path variant notes: This replacement only covers driver-related path references, for example:
>
> - `/usr/local/Ascend/driver/lib64/` (Docker mount path, the most common)
> - `/usr/local/Ascend/driver/` (HDK installation root path)
>
> The following paths are **not** within the replacement scope and remain unchanged:
>
> - `/usr/local/Ascend/cann/set_env.sh` (environment variable initialization)
> - `/usr/local/Ascend/ascend-toolkit/set_env.sh` (Ascend Toolkit initialization)
> - `/usr/local/Ascend/nnal/atb/set_env.sh` (ATB library initialization)

---

## Usage Steps

1. Enter the repository root directory.

    ```bash
    cd /path/to/MindSpeed
    ```

2. Preview the changes to be made (recommended).

    Before making actual modifications, first confirm the scope of changes in `--dry-run` mode:

    ```bash
    python3 tools/replace_ascend_path.py --dry-run
    ```

    Output example:

    ```bash
    [DRY RUN] Path replacement: /usr/local/Ascend/driver -> /usr/local/npu/driver
    Scan directory : /path/to/MindSpeed
    File types     : .md, .py, .rst, .sh, .txt + Dockerfile
    ------------------------------------------------------------
    Found XXX candidate file(s), processing...

    [would replace   1] docker/Dockerfile
    [would replace   2] docker/OVERVIEW.md
    [would replace   2] docker/OVERVIEW.zh.md
    ...

    ============================================================
    [DRY RUN] XXX file(s) would be modified, XXX replacement(s) total.
            Remove --dry-run to apply changes.
    ```

3. Perform batch replacement.

    After confirming the preview is correct, perform the actual replacement:

    ```bash
    # Default: Replace /usr/local/Ascend/driver with /usr/local/npu/driver.
    python3 tools/replace_ascend_path.py
    ```

    After execution, the script outputs the number of modified files and the total number of replacements.

4. Verify the replacement result.

    ```bash
    # Check whether any driver paths remain unreplaced (the result should be 0).
    grep -r "/usr/local/Ascend/driver" . \
    --include='*.sh' \
    --include='*.md' \
    --include='*.rst' \
    --include='*.py' \
    --include='*.txt' \
    --include='Dockerfile' \
    --exclude='replace_ascend_path.py' \
    --exclude='replace_ascend_path_guide.md' \
    --exclude-dir='.git' \
    | wc -l
    ```

---

## Post-Execution Verification

1. Verify driver path loading.

    ```bash
    # Verify that the driver directory exists under the new path.
    ls /usr/local/npu/driver/lib64/

    # Load the environment variables (the ascend-toolkit path remains unchanged, so the original path is still used).
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # Verify that the environment variables take effect.
    echo $ASCEND_HOME_PATH
    ```

2. Verify component installation.

    ```bash
    # Verify that MindSpeed is installed successfully
    python3 -c "import mindspeed; print('MindSpeed installed successfully')"

    # Verify that NPU is available
    python3 -c "import torch_npu; print('NPU available:', torch_npu.npu.is_available())"
    ```

3. Verify the chip information interface (`dcmi_get_device_chip_info`).

    Some versions require the chip identifier returned by the `dcmi_get_device_chip_info` interface to be `A2G3` or `A2G4`.

    > Note: The current MindSpeed code does not directly call this interface; this is provided only as an adaptation note. If upper-layer services or O&M scripts rely on the return value of this interface to determine the chip model, ensure that the return value is `A2G3` or `A2G4`; otherwise, model-related logic branches may be affected.

    For verification methods, refer to
    [dcmi_get_device_chip_info](https://support.huawei.com/enterprise/zh/doc/EDOC1100568435/8739bb5a).

4. Conduct smoke testing for core feature.

    Configure according to the README of the corresponding model to verify that the training process can start normally.

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # Run the sample script (subject to the specific model)
    bash ./train_distributed.sh
    ```

---

## Script Parameter Description

```bash
usage: replace_ascend_path.py [-h] [--source SOURCE] [--target TARGET]
                               [--dir DIR] [--extensions EXT [EXT ...]]
                               [--dry-run]

选项：
  -h, --help            Help information
  --source SOURCE       Source path (default: /usr/local/Ascend/driver)
  --target TARGET       Target path (default: /usr/local/npu/driver)
  --dir DIR             Directory to scan (default: current directory .)
  --extensions EXT...   File extension whitelist (default: .sh .md .rst .py .txt)
  --dry-run             Preview changes only, without modifying files
```
