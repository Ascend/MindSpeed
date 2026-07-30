#!/bin/bash
set -e

# ── 0. 自动发现 MindSpeed 根目录 ────────────────────────────────────────────
# 支持两种场景：
#   a) 直接用 MindSpeed 目录打开 devcontainer → /workspace 就是 MindSpeed 根目录
#   b) 用包含 MindSpeed 的父目录打开 → MindSpeed 是 /workspace 的子目录
detect_mindspeed_root() {
    if [ -f /workspace/setup.py ] && grep -q "mindspeed" /workspace/setup.py 2>/dev/null; then
        echo "/workspace"
        return
    fi
    # 在 /workspace 下最多搜索 5 层，找到含 "mindspeed" 的 setup.py
    local found
    found=$(find /workspace -maxdepth 5 -name setup.py -exec grep -l "mindspeed" {} \; 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        dirname "$found"
        return
    fi
    # 兜底：假设就是 /workspace
    echo "/workspace"
}

MINDSPEED_ROOT=$(detect_mindspeed_root)
echo ">>> Detected MindSpeed root: ${MINDSPEED_ROOT}"

# ── 1. Source CANN 环境 ────────────────────────────────────────────────────
echo "=== [1/6] Sourcing CANN environment ==="
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# ── 2. 配置 pip 镜像源（与 ci/Dockerfile 保持一致） ────────────────────────
ARCH=$(uname -m)
PIP_EXTRA="--extra-index-url https://mirrors.aliyun.com/pypi/simple/"
if [ "$ARCH" = "x86_64" ]; then
    PIP_EXTRA="--extra-index-url https://download.pytorch.org/whl/cpu/"
fi
echo ">>> Using pip mirror: ${PIP_EXTRA}"

# ── 3. 安装 Python 依赖 ────────────────────────────────────────────────────
echo "=== [3/6] Installing Python dependencies ==="
cd "${MINDSPEED_ROOT}"
pip install ${PIP_EXTRA} -r requirements.txt

# ── 4. 安装 MindSpeed（editable 模式） ────────────────────────────────────
echo "=== [4/6] Installing MindSpeed (editable mode) ==="
pip install ${PIP_EXTRA} -e "${MINDSPEED_ROOT}"

# ── 5. 安装 pre-commit hooks ───────────────────────────────────────────────
echo "=== [5/6] Installing pre-commit hooks ==="
if [ -f "${MINDSPEED_ROOT}/.pre-commit-config.yaml" ]; then
    cd "${MINDSPEED_ROOT}"
    pre-commit install --install-hooks
    echo "Pre-commit hooks installed."
else
    echo "No .pre-commit-config.yaml found at ${MINDSPEED_ROOT}, skipping."
fi

# ── 6. 软链 gitleaks 到 MindSpeed 根目录（pre-commit hook 期望 ./gitleaks） ──
echo "=== [6/6] Symlinking gitleaks for pre-commit hook ==="
ln -sf /usr/local/bin/gitleaks "${MINDSPEED_ROOT}/gitleaks"

# ── 写入环境变量标记文件（供 postAttach.sh 和新终端使用） ─────────────────
echo "${MINDSPEED_ROOT}" > /tmp/.mindspeed_root

echo ""
echo "============================================"
echo "  MindSpeed dev container is ready!"
echo "  Root: ${MINDSPEED_ROOT}"
echo "  CANN: ${ASCEND_HOME_PATH}"
NPU_OK=$(python3 -c "import torch_npu; print(torch_npu.npu.is_available())" 2>/dev/null || echo "check after CANN source")
echo "  NPU:  ${NPU_OK}"
echo "============================================"
