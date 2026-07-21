# ============================================
# MindSpeed CI Docker Image
# NPU Type: Configurable (910b, a3, etc.)
# Supports: x86_64 and aarch64
# Supports: Ubuntu and openEuler CANN base images
# ============================================

ARG OS=openeuler24.03
ARG BASE_IMAGE_VERSION=9.0.1
ARG NPU_TYPE=910b
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=""

FROM ${BASE_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${BASE_IMAGE_VERSION}-${NPU_TYPE}-${OS}-py${PYTHON_VERSION}}

USER root
SHELL ["/bin/bash", "-c"]

ARG OS_FAMILY=ubuntu
ARG TORCH_VERSION=2.9.0
ARG TORCH_NPU_VERSION=2.9.0
ARG MINDSPEED_BRANCH=master
ARG MEGATRON_BRANCH=core_v0.12.1
ARG MINDSPEED_LLM_BRANCH=master
ARG VLLM_VERSION=v0.18.0
ARG VLLM_ASCEND_VERSION=releases/v0.18.0
ARG VERL_VERSION=v0.7.0
ARG SOC_VERSION=ascend910b1

# ============================================
# System dependencies
# ============================================
COPY configure_repo.sh /tmp/configure_repo.sh
RUN chmod +x /tmp/configure_repo.sh && \
    bash /tmp/configure_repo.sh && \
    rm -f /tmp/configure_repo.sh

RUN ARCH=$(uname -m) && \
    echo "Detected CPU architecture: ${ARCH}" && \
    if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then \
        echo "ERROR: Unsupported architecture: ${ARCH}"; \
        exit 1; \
    fi

RUN echo "Installing system dependencies for ${OS_FAMILY}..." && \
    if [ "$OS_FAMILY" = "ubuntu" ]; then \
        apt-get install -y --no-install-recommends \
            build-essential \
            cmake \
            curl \
            gcc \
            g++ \
            git \
            jq \
            libnuma-dev \
            vim \
            wget && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*; \
    elif [ "$OS_FAMILY" = "openeuler" ]; then \
        yum install -y \
            cmake \
            curl \
            findutils \
            gcc \
            gcc-c++ \
            git \
            jq \
            make \
            numactl-devel \
            tar \
            vim \
            wget \
            which && \
        yum clean all && \
        rm -rf /var/cache/yum; \
    else \
        echo "ERROR: Unsupported OS family: ${OS_FAMILY}"; \
        exit 1; \
    fi && \
    rm -rf /tmp/* /var/tmp/* && \
    pip install --upgrade pip packaging setuptools==80.10.2 && \
    pip cache purge

# ============================================
# Clone repos (with retry for network resilience)
# ============================================
RUN mkdir -p /home/ci_deps && cd /home/ci_deps && \
    git clone https://gitcode.com/Ascend/MindSpeed.git && \
    cd MindSpeed && git checkout ${MINDSPEED_BRANCH} && cd .. && \
    for i in 1 2 3; do \
        git clone https://github.com/NVIDIA/Megatron-LM.git && break || sleep 5; \
    done && \
    cd Megatron-LM && git checkout ${MEGATRON_BRANCH} && cd .. && \
    git clone https://gitcode.com/Ascend/MindSpeed-LLM.git && \
    cd MindSpeed-LLM && git checkout ${MINDSPEED_LLM_BRANCH} && cd .. && \
    for i in 1 2 3; do \
        git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git && break || sleep 5; \
    done && \
    for i in 1 2 3; do \
        git clone --depth 1 --branch ${VLLM_ASCEND_VERSION} https://github.com/vllm-project/vllm-ascend.git && break || sleep 5; \
    done && \
    for i in 1 2 3; do \
        git clone -b ${VERL_VERSION} https://github.com/verl-project/verl.git && break || sleep 5; \
    done && \
    cd /home/ci_deps/verl && \
    sed -i 's/WorkerWrapperBase(vllm_config=self.vllm_config)/WorkerWrapperBase(rpc_rank=0)/' \
        verl/workers/rollout/vllm_rollout/vllm_rollout.py && \
    grep -q 'rpc_rank=0' verl/workers/rollout/vllm_rollout/vllm_rollout.py || { echo "ERROR: WorkerWrapperBase patch failed"; exit 1; } && \
    sed -i 's/return self.inference_engine.execute_method(method, \*args, \*\*kwargs)/return getattr(self.inference_engine, method)(*args, **kwargs)/' \
        verl/workers/rollout/vllm_rollout/vllm_rollout.py && \
    grep -q 'getattr(self.inference_engine, method)' verl/workers/rollout/vllm_rollout/vllm_rollout.py || { echo "ERROR: execute_method patch failed"; exit 1; } && \
    sed -i 's/triton-ascend==3.2.0rc4/triton-ascend==3.2.1/' requirements-npu.txt && \
    grep -q 'triton-ascend==3.2.1' requirements-npu.txt || { echo "ERROR: triton-ascend version patch failed"; exit 1; }

# ============================================
# Install MindSpeed + Megatron-LM + MindSpeed-LLM
# ============================================
RUN ARCH=$(uname -m) && \
    DEVLIB_DIR=$(find /usr/local/Ascend -type d -path "*/devlib/linux/${ARCH}" | head -n 1) && \
    if [ -n "$DEVLIB_DIR" ]; then \
        export LD_LIBRARY_PATH="${DEVLIB_DIR}:${LD_LIBRARY_PATH}"; \
    fi && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then \
        source /usr/local/Ascend/nnal/atb/set_env.sh; \
    fi && \
    PIP_EXTRA="--extra-index-url https://mirrors.aliyun.com/pypi/simple/" && \
    if [ "$ARCH" = "x86_64" ]; then \
        PIP_EXTRA="--extra-index-url https://download.pytorch.org/whl/cpu/"; \
    fi && \
    pip install ${PIP_EXTRA} torch==${TORCH_VERSION} torch_npu==${TORCH_NPU_VERSION} && \
    pip install ${PIP_EXTRA} -r /home/ci_deps/MindSpeed/requirements.txt && \
    pip install ${PIP_EXTRA} -e /home/ci_deps/MindSpeed && \
    pip install ${PIP_EXTRA} --no-deps -e /home/ci_deps/Megatron-LM && \
    pip install ${PIP_EXTRA} -e /home/ci_deps/MindSpeed-LLM && \
    pip install ${PIP_EXTRA} transformers==4.57.1 && \
    pip install ${PIP_EXTRA} protobuf==6.31.1 && \
    pip install ${PIP_EXTRA} pytest-timeout && \
    pip install ${PIP_EXTRA} nltk pyarrow && \
    pip install ${PIP_EXTRA} coverage && \
    rm -rf /tmp/* /var/tmp/* && \
    pip cache purge

# ============================================
# Install vLLM + vLLM-ascend (deps for verl)
# ============================================
RUN ARCH=$(uname -m) && \
    DEVLIB_DIR=$(find /usr/local/Ascend -type d -path "*/devlib/linux/${ARCH}" | head -n 1) && \
    if [ -n "$DEVLIB_DIR" ]; then \
        export LD_LIBRARY_PATH="${DEVLIB_DIR}:${LD_LIBRARY_PATH}"; \
    fi && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then \
        source /usr/local/Ascend/nnal/atb/set_env.sh; \
    fi && \
    PIP_EXTRA="--extra-index-url https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/" && \
    if [ "$ARCH" = "x86_64" ]; then \
        PIP_EXTRA="--extra-index-url https://download.pytorch.org/whl/cpu/"; \
    fi && \
    cd /home/ci_deps/vllm && \
    pip install ${PIP_EXTRA} setuptools-scm wheel packaging build jinja2 && \
    pip install ${PIP_EXTRA} -r requirements/build.txt && \
    VLLM_TARGET_DEVICE=empty pip install ${PIP_EXTRA} -v -e . && cd .. && \
    cd /home/ci_deps/vllm-ascend && \
    pip install ${PIP_EXTRA} -r requirements.txt && \
    export SOC_VERSION=${SOC_VERSION} && \
    export COMPILE_CUSTOM_KERNELS=1 && \
    pip install ${PIP_EXTRA} --no-build-isolation --no-deps -v -e . && cd .. && \
    rm -rf /tmp/* /var/tmp/* && \
    pip cache purge

# ============================================
# Install verl
# ============================================
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then \
        source /usr/local/Ascend/nnal/atb/set_env.sh; \
    fi && \
    PIP_EXTRA="--extra-index-url https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/" && \
    cd /home/ci_deps/verl && \
    pip install ${PIP_EXTRA} -r requirements-npu.txt && \
    pip install ${PIP_EXTRA} -v -e . && \
    cd .. && \
    rm -rf /tmp/* /var/tmp/* && \
    pip cache purge

# ============================================
# Install mbridge (verl megatron bridge dependency)
# ============================================
RUN pip install mbridge --extra-index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip cache purge

ENV ENABLE_ATB=1
ENV PYTHONPATH="/home/ci_deps/Megatron-LM:/home/ci_deps/MindSpeed-LLM${PYTHONPATH:+:${PYTHONPATH}}"

WORKDIR /home/ci_deps/MindSpeed

RUN pip install transformers==4.57.1 --extra-index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip cache purge

RUN pip list

CMD ["/bin/bash"]
