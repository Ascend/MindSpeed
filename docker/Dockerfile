# ============================================
# MindSpeed Core Docker Image
# NPU Type: Configurable (910b, a3, etc.)
# Supports: x86_64 and aarch64
# Supports: Ubuntu and openEuler CANN base images
# ============================================

ARG OS=openeuler24.03
ARG BASE_IMAGE_VERSION=9.0.0-beta.2
ARG NPU_TYPE=910b
ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE=""

FROM ${BASE_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${BASE_IMAGE_VERSION}-${NPU_TYPE}-${OS}-py${PYTHON_VERSION}}

USER root
SHELL ["/bin/bash", "-c"]

ARG OS_FAMILY=ubuntu
ARG TORCH_VERSION=2.7.1
ARG TORCH_NPU_VERSION=2.7.1
ARG MINDSPEED_BRANCH=master
ARG MEGATRON_BRANCH=core_v0.12.1

# Prepare required system dependencies
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

# Prepare MindSpeed source and Megatron-LM
RUN git clone https://gitcode.com/Ascend/MindSpeed.git && \
    cd MindSpeed && git checkout ${MINDSPEED_BRANCH} && cd .. && \
    git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && git checkout ${MEGATRON_BRANCH} && cd ..

# Install MindSpeed and Megatron-LM
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"; \
    fi && \
    DEVLIB_DIR=$(find /usr/local/Ascend -type d -path "*/devlib/linux/${ARCH}" | head -n 1) && \
    if [ -n "$DEVLIB_DIR" ]; then \
        export LD_LIBRARY_PATH="${DEVLIB_DIR}:${LD_LIBRARY_PATH}"; \
    fi && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then \
        source /usr/local/Ascend/nnal/atb/set_env.sh; \
    fi && \
    pip install torch==${TORCH_VERSION} torch_npu==${TORCH_NPU_VERSION} && \
    pip install -r /MindSpeed/requirements.txt && \
    pip install -e /MindSpeed && \
    pip install -e /Megatron-LM && \
    pip install transformers==4.57.1 && \
    rm -rf /tmp/* /var/tmp/* && \
    pip cache purge

ENV PYTHONPATH="/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

WORKDIR /MindSpeed

# Show install results
RUN pip list

# Setting Default Commands
CMD ["/bin/bash"]
