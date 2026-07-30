# ──────────────────────────────────────────────────────────────────────────────
# MindSpeed Dev Container
# 基于 CANN 基础镜像，提供代码开发、op_builder JIT 编译、UT 所需的完整环境。
# 版本参数与 ci/Dockerfile 保持一致。
# 生产部署请使用 docker/Dockerfile。
# ──────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE_VERSION=9.0.1
ARG NPU_TYPE=910b
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=3.12

FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${BASE_IMAGE_VERSION}-${NPU_TYPE}-${OS}-py${PYTHON_VERSION}

LABEL maintainer="mindspeed-dev"
LABEL description="MindSpeed development container based on CANN / Ubuntu / Python 3.12"

ARG TORCH_VERSION=2.9.0
ARG TORCH_NPU_VERSION=2.9.0

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# ── 1. 系统工具（op_builder JIT 编译 + 开发辅助） ───────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # op_builder JIT 编译依赖
    build-essential \
    cmake \
    gcc \
    g++ \
    ninja-build \
    # 开发工具
    ccache \
    curl \
    git \
    jq \
    libnuma-dev \
    unzip \
    vim \
    wget \
    # pre-commit: clang-format（VS Code C++ 格式化；pre-commit hook 自行下载 v18 二进制）
    clang-format-14 \
    && rm -rf /var/lib/apt/lists/*

# ── 2. 配置 pip 镜像源（arm64 走阿里云，x86_64 走 pytorch cpu 源） ────────
#    与 ci/Dockerfile 策略一致，加速包下载
RUN ARCH=$(uname -m) && \
    PIP_EXTRA="--extra-index-url https://mirrors.aliyun.com/pypi/simple/"; \
    if [ "$ARCH" = "x86_64" ]; then \
        PIP_EXTRA="--extra-index-url https://download.pytorch.org/whl/cpu/"; \
    fi && \
    pip3 config set global.extra-index-url "$(echo ${PIP_EXTRA} | sed 's/--extra-index-url //')" && \
    pip3 install --no-cache-dir --upgrade pip packaging setuptools==80.10.2 && \
    pip3 install --no-cache-dir \
    # 代码质量（pre-commit hooks 后端）
    ruff \
    pylint \
    bandit \
    codespell \
    pre-commit \
    # 调试 & profiling
    ipython \
    ipdb \
    && pip3 cache purge

# ── 3. gitleaks（离线二进制，pre-commit hook 期望 ./gitleaks 存在于仓库根目录）───
RUN ARCH=$(uname -m) && \
    GITLEAKS_VER="8.20.0" && \
    if [ "$ARCH" = "x86_64" ]; then \
        URL="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VER}/gitleaks_${GITLEAKS_VER}_linux_x64.tar.gz"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        URL="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VER}/gitleaks_${GITLEAKS_VER}_linux_arm64.tar.gz"; \
    else \
        echo "ERROR: Unsupported architecture: ${ARCH}"; exit 1; \
    fi && \
    wget -q "$URL" -O /tmp/gitleaks.tar.gz && \
    tar -xzf /tmp/gitleaks.tar.gz -C /usr/local/bin gitleaks && \
    rm -f /tmp/gitleaks.tar.gz && \
    chmod +x /usr/local/bin/gitleaks

# ── 4. PyTorch + torch_npu（若 CANN 基础镜像未预装则安装） ──────────────────
#    与 ci/Dockerfile 保持一致：x86_64 走 pytorch cpu 源，aarch64 走阿里云源
RUN if ! python3 -c "import torch" 2>/dev/null; then \
        ARCH=$(uname -m) && \
        PIP_EXTRA="--extra-index-url https://mirrors.aliyun.com/pypi/simple/"; \
        if [ "$ARCH" = "x86_64" ]; then \
            PIP_EXTRA="--extra-index-url https://download.pytorch.org/whl/cpu/"; \
        fi && \
        pip3 install ${PIP_EXTRA} torch==${TORCH_VERSION} torch_npu==${TORCH_NPU_VERSION} && \
        pip3 cache purge; \
    fi

# ── 5. ccache 加速 op_builder 的 C++ 扩展重复编译 ──────────────────────────
RUN mkdir -p /ccache && ccache -M 5G
ENV CCACHE_DIR=/ccache
ENV PATH="/usr/lib/ccache:${PATH}"

# ── 6. 工作目录 ────────────────────────────────────────────────────────────
WORKDIR /workspace

CMD ["/bin/bash"]
