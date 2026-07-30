# MindSpeed Dev Container 使用说明

本目录包含 VS Code Dev Container 配置，用于在容器内开发 MindSpeed Core、执行 UT、运行 pre-commit 检查。

## 前置条件

| 条件 | 说明 |
|---|---|
| VS Code | 已安装 `Dev Containers` 扩展（`ms-vscode-remote.remote-containers`） |
| Remote-SSH（远程场景） | 本地需先通过 `Remote - SSH` 连接到 NPU 服务器 |
| Docker | NPU 服务器上 Docker 已安装，当前用户有 `docker` 权限 |
| NPU 驱动 | 宿主机已安装 Ascend NPU 驱动（`/usr/local/Ascend/driver` 存在，`npu-smi` 可执行） |
| CANN 基础镜像 | 能拉取 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.1-910b-ubuntu22.04-py3.12` |

## 环境规格

| 工具 | 版本 |
|---|---|
| 基础镜像 | CANN 9.0.1 / Ubuntu 22.04 / Python 3.12 |
| GCC / G++ | 系统自带 |
| CMake | 系统自带 |
| PyTorch | 2.9.0 |
| torch_npu | 2.9.0 |
| ccache | 系统包（加速 op_builder 的 C++ 扩展编译） |
| clang-format | 18.x（C++ 代码格式化） |
| Ruff | 最新（Python linting + formatting） |
| Pylint | 最新（Python 代码质量） |
| Bandit | 最新（Python 安全扫描） |
| codespell | 最新（拼写检查） |
| gitleaks | 8.20.0（密钥泄露扫描） |

## 两种工作空间模式

Dev Container 支持两种打开方式，初始化脚本会自动发现 MindSpeed 根目录：

### 模式 A：直接打开 MindSpeed 仓库（推荐）

适用于只开发 MindSpeed 本身的场景。

### 模式 B：打开包含 MindSpeed 的父目录

适用于需要同时开发 MindSpeed + Megatron-LM 等多仓库的场景。目录结构示例：

```plaintext
~/projects/                  # 用 VS Code 打开这个目录
├── MindSpeed/               # MindSpeed 仓库（含 .devcontainer/）
│   ├── .devcontainer/
│   ├── setup.py
│   └── ...
└── Megatron-LM/             # 其他依赖仓库
    └── ...
```

> **注意**：模式 B 需要父目录下存在 `.devcontainer/`。最简单的方式是在父目录创建一个符号链接：
>
> ```bash
> cd ~/projects
> ln -s MindSpeed/.devcontainer .devcontainer
> ```

---

## 快速开始

### 1. 连接 NPU 服务器（本地场景）

在 VS Code 中按 `F1` → **Remote-SSH: Connect to Host** → 选择你的 NPU 服务器。

NPU 服务器 SSH 配置示例（`~/.ssh/config`）：

```plaintext
Host npu-server
    HostName 192.168.x.x
    User root
    Port 22
```

### 2. 打开工作目录

- **模式 A**：`File` → `Open Folder` → 选择 MindSpeed 仓库目录
- **模式 B**：`File` → `Open Folder` → 选择包含 MindSpeed 的父目录（需提前创建 `.devcontainer` 符号链接）

### 3. 启动容器

按 `F1` → **Dev Containers: Reopen in Container**，等待镜像构建完成。

- **首次构建**约 5～10 分钟（拉取基础镜像 + 安装依赖）
- **后续打开**秒级恢复

容器启动后会**自动**完成以下初始化：

1. 自动发现 MindSpeed 根目录（`setup.py` 中搜索 `mindspeed` 关键字，最多搜索 5 层）
2. Source CANN + ATB 环境脚本
3. `pip install -r requirements.txt`
4. `pip install -e <mindspeed_root>`（editable 模式，代码修改实时生效）
5. `pre-commit install --install-hooks`（安装 git hooks）
6. 软链 `gitleaks` 到 MindSpeed 根目录（供 pre-commit hook 使用）

### 4. 验证环境

在 VS Code 终端中执行：

```bash
# NPU 可用性
python3 -c "import torch_npu; print(torch_npu.npu.is_available())"
# → True

# MindSpeed 已安装（editable 模式）
pip show mindspeed | grep Editable
# → Editable project location: /workspace          （模式 A）
# → Editable project location: /workspace/MindSpeed（模式 B）

# pre-commit hooks 已就绪
cd $(cat /tmp/.mindspeed_root) && pre-commit run --all-files
```

> **提示**：初始化成功后可通过 `cat /tmp/.mindspeed_root` 查看自动发现的 MindSpeed 根目录路径。模式 B 下还提供了 `cdm` 别名快速跳转到 MindSpeed 目录。

## 日常开发

> **模式 B 用户**：终端中的命令需在 MindSpeed 根目录下执行。可使用 `cdm` 别名快速跳转，或 `cd $(cat /tmp/.mindspeed_root)`。

### 修改 Python 代码

源码通过 bind mount 挂载，修改即时生效，无需重建容器：

```bash
# 编辑代码
vim mindspeed/core/xxx.py

# 运行测试
pytest tests_extend/unit_tests/ -xvs

# 运行指定测试
pytest tests_extend/unit_tests/test_xxx.py::TestClass::test_func -xvs
```

### 修改 C++ 扩展代码

`mindspeed/ops/csrc/` 下的 C++ 代码通过 `op_builder` 在运行时 JIT 编译，修改后下次 `import` 时自动重新编译，**无需手动操作**。

### 提交代码

```bash
git add .
git commit -m "feat: xxx"
# pre-commit hooks 自动运行：
#   ruff-check → ruff-format → pylint → clang-format → codespell → bandit → gitleaks
```

跳过特定 hook：

```bash
SKIP=pylint git commit -m "WIP: xxx"
SKIP=gitleaks-offline-scan git commit -m "WIP: xxx"
```

### 完整编译验证

```bash
bash build.sh
```

## 重建容器

以下情况需要重建（`F1` → **Dev Containers: Rebuild Container**）：

| 改动 | 需要重建？ |
|---|---|
| `requirements.txt` 新增/变更依赖 | ✅ |
| `.devcontainer/Dockerfile` | ✅ |
| `.devcontainer/devcontainer.json` | ✅ |
| `mindspeed/` 下 Python 代码 | ❌ 实时生效 |
| `mindspeed/ops/csrc/` 下 C++ 代码 | ❌ JIT 自动重编译 |
| `.pre-commit-config.yaml` | ❌ 手动运行 `pre-commit clean && pre-commit install` |

## 与生产镜像的区别

| | Dev Container（本目录） | 生产镜像（docker/） |
|---|---|---|
| 用途 | 代码开发、UT、pre-commit | 训练任务部署 |
| 源码 | bind mount 挂载，实时编辑 | 从 git clone，不可编辑 |
| 依赖安装 | `pip install -e`（editable） | `pip install -e`（固化到镜像层） |
| Megatron-LM | 按需安装 | 从 git clone + 安装 |
| 开发工具 | ruff, pylint, bandit, ipython, ipdb, clang-format | 不含 |
| 镜像大小 | 较大（含完整开发工具链） | 较小（仅运行依赖） |

## 常见问题

### Q: 容器启动后 `npu-smi` 报 command not found？

宿主机 NPU 驱动通过 mount 进入容器，检查挂载是否生效：

```bash
ls /usr/local/bin/npu-smi
# 如果不存在，确认宿主机上 npu-smi 路径为 /usr/local/bin/npu-smi
```

### Q: op_builder 编译报找不到 CANN 头文件？

在终端中手动 source CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 确认环境变量
echo $ASCEND_HOME_PATH
# → /usr/local/Ascend/ascend-toolkit/latest
```

> `postAttachCommand` 已自动将 source 命令写入 `~/.bashrc`，新终端会自动加载。如果旧终端报错，关闭后新开一个即可。

### Q: pre-commit 的 gitleaks 报 "No such file or directory"？

初始化脚本已自动创建软链 `<mindspeed_root>/gitleaks → /usr/local/bin/gitleaks`。如果仍然报错：

```bash
MINDSPEED_ROOT=$(cat /tmp/.mindspeed_root 2>/dev/null || echo /workspace)
ls -la ${MINDSPEED_ROOT}/gitleaks
# 若不存在，手动创建：
ln -sf /usr/local/bin/gitleaks ${MINDSPEED_ROOT}/gitleaks
```

### Q: `torch_npu.npu.is_available()` 返回 False？

1. 确认宿主机 NPU 驱动正常：`npu-smi info`
2. 确认容器有 `--privileged`（`runArgs` 已配置）
3. 确认 `ASCEND_HOME_PATH` 已设置：`echo $ASCEND_HOME_PATH`

### Q: 如何切换基础镜像版本（如 910b → a3，Ubuntu → openEuler）？

修改 `.devcontainer/Dockerfile` 第 1 行的 `FROM` 指令，例如：

```dockerfile
# 切换到 a3 + openEuler
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.1-a3-openeuler24.03-py3.12
```

然后 `F1` → **Dev Containers: Rebuild Container**。
