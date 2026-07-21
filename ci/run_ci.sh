#!/bin/bash
# ============================================
# MindSpeed CI Entry Script (UT & ST Gate)
#
# Architecture:
#   CI automation creates ${WORKSPACE} and places the PR-merged
#   MindSpeed repo at ${WORKSPACE}/CODE/. This script installs
#   that copy and runs the test suite against it.
#
#   The Docker image (mindspeed-ci) provides only dependencies:
#     /home/ci_deps/Megatron-LM
#     /home/ci_deps/MindSpeed-LLM
#     /home/ci_deps/verl
#     /home/ci_deps/vllm
#     /home/ci_deps/vllm-ascend
#     /home/models
#
#   MindSpeed itself is NOT installed in the image; it comes
#   from ${WORKSPACE}/CODE/ and is pip-installed at runtime.
# ============================================
set -euo pipefail

# --------------------------------------------------
# Distributed communication setup (Ascend NPU)
# --------------------------------------------------
export MASTER_ADDR=localhost
export MASTER_PORT=6001
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp189s0f0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp189s0f0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-enp189s0f0}
export GLOO_SOCKET_FAMILY=AF_INET
export HCCL_SOCKET_FAMILY=AF_INET
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800

# --------------------------------------------------
# Immutable dependency paths (provided by Docker image)
# --------------------------------------------------
MEGATRON_DIR="/home/ci_deps/Megatron-LM"
MINDSPEED_LLM_DIR="/home/ci_deps/MindSpeed-LLM"
VERL_DIR="/home/ci_deps/verl"
MODELS_DIR="/home/models"

# --------------------------------------------------
# Files / directories that trigger CI when changed
# --------------------------------------------------
TRIGGER_PATTERNS=("ci/" "mindspeed/" "tests_extend/" "tests_extend_v2/" "requirements.txt" "setup.py")

# --------------------------------------------------
# Helper: retry git checkout until success
# --------------------------------------------------
try_checkout_branch() {
    local branch_name="$1"
    while true; do
        if git checkout "$branch_name"; then
            echo "Successfully checked out branch: ${branch_name}"
            return 0
        fi
        echo "Failed to check out branch: ${branch_name}. Fetching all and retrying..."
        git fetch --all || true
    done
}

# --------------------------------------------------
# Main UT / ST runner
# --------------------------------------------------
run_ut() {
    local workspace="$1"
    local branch="$2"
    local code_dir="${workspace}/CODE"

    # --- Ascend environment (external scripts may not be -u safe) ---
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    set -u
    export ENABLE_ATB=1

    cd "${code_dir}"

    # --- Determine Python version at runtime (matches Docker image) ---
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "Detected Python version: $PYTHON_VERSION"

    # --- Install MindSpeed from PR-merged code ---
    pip$PYTHON_VERSION install -e .
    pip$PYTHON_VERSION install transformers==4.57.1

    export PYTHONPATH="${PYTHONPATH}:${code_dir}"
    cd "${workspace}"

    # --- Legacy branch support: TransformerEngineNPU + MegatronAdaptor ---
    if [ "$branch" == "core_r0.17.0" ] || [ "$branch" == "core_r0.18.0" ]; then
        git clone https://gitcode.com/Ascend/TransformerEngineNPU.git
        rm -rf TransformerEngineNPU/pyproject.toml
        pip install -e TransformerEngineNPU -v
        git clone https://gitcode.com/Ascend/MegatronAdaptor.git
        cd MegatronAdaptor
        git checkout "$branch"
        pip install -e .
        cd ..
    fi

    # --- Resolve Megatron-LM branch from README.md ---
    checkout_lines=$(grep "git checkout" "${code_dir}/README.md" | awk '{print substr($0, index($0,$3))}' | tail -n 1 || true)
    echo "Resolved Megatron-LM ref: ${checkout_lines}"
    if [ -z "$checkout_lines" ]; then
        echo "ERROR: No valid git checkout line found in README.md"
        exit 1
    fi

    # --- Prepare Megatron-LM ---
    cd "${workspace}"
    cp -rf "${MEGATRON_DIR}" ./
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/Megatron-LM"
    cd Megatron-LM
    try_checkout_branch "$checkout_lines"

    # --- Copy test suite from PR-merged code into Megatron-LM ---
    cp -r "${code_dir}/tests_extend" ./

    # --- Dump installed packages ---
    pip$PYTHON_VERSION list

    # --- Apply required source patches ---
    sed -i '1s/^/import mindspeed.megatron_adaptor\n/' pretrain_gpt.py
    sed -i '1s|^|from __future__ import annotations\n|' megatron/core/dist_checkpointing/strategies/base.py

    # ============================================
    # ST 1: Pretrain base
    # ============================================
    echo "===== Running pretrain_base ST ====="
    bash ./tests_extend/system_tests/pretrain_base.sh
    if [ $? -ne 0 ]; then
        echo "ERROR: pretrain_base ST failed"
        return 1
    fi

    # ============================================
    # ST 2: MindSpeed-LLM (DeepSeek) – skip on legacy branches
    # ============================================
    echo "Branch: $branch"
    local skip_llm_branches=("26.0.0_core_r0.12.1" "core_r0.14.0" "core_r0.15.3" "core_r0.16.0" "core_r0.17.0" "core_r0.18.0")
    local skip_llm=false
    for skip in "${skip_llm_branches[@]}"; do
        if [[ "$checkout_lines" == *"$skip"* ]] || [[ "$branch" == *"$skip"* ]]; then
            skip_llm=true
            break
        fi
    done

    if [ "$skip_llm" = false ]; then
        cp -rf "${MINDSPEED_LLM_DIR}/mindspeed_llm" ./
        cp -rf "${MINDSPEED_LLM_DIR}/pretrain_gpt.py" pretrain_gpt_llm.py

        echo "MindSpeed-LLM version info:"
        cd "${MINDSPEED_LLM_DIR}"
        git log -1
        git branch
        cd "${workspace}/Megatron-LM"

        local commitId
        commitId=$(git rev-parse --short=9 HEAD)
        local megatron_id='1d462bd37'
        echo "Megatron-LM commitId: $commitId"
        if [ "$commitId" != "${megatron_id}" ]; then
            echo "===== Running DeepSeek V3 pretrain ST ====="
            bash ./tests_extend/system_tests/deepseek/pretrain_deepseek_v3_ptd_dualpipev.sh
            if [ $? -ne 0 ]; then
                echo "ERROR: DeepSeek V3 pretrain ST failed"
                return 1
            fi
        fi
    else
        echo "Skipping MindSpeed-LLM ST for legacy branch"
    fi

    # ============================================
    # ST 3: verl – skip on branches without args_utils.py or legacy
    # ============================================
    if [ -f "${code_dir}/mindspeed/args_utils.py" ] && \
       [[ "$branch" != "core_r0.15.3" ]] && \
       [[ "$branch" != "core_r0.17.0" ]] && \
       [[ "$branch" != "core_r0.18.0" ]]; then
        cd "${workspace}"
        cp -rf "${VERL_DIR}" ./
        cd verl
        echo "===== Running verl ST ====="
        echo "USE_DIST_CKPT=False MODEL_ID=${MODELS_DIR}/Qwen2.5-0.5B MODEL_PATH=${MODELS_DIR}/Qwen2.5-0.5B DIST_CKPT_PATH=${MODELS_DIR}/Qwen2.5-0.5B-dist/ HOME=${MODELS_DIR} bash tests/special_npu/run_qwen2_5_05b_grpo_mindspeed.sh"
        USE_DIST_CKPT=False \
            MODEL_ID="${MODELS_DIR}/Qwen2.5-0.5B" \
            MODEL_PATH="${MODELS_DIR}/Qwen2.5-0.5B" \
            DIST_CKPT_PATH="${MODELS_DIR}/Qwen2.5-0.5B-dist/" \
            HOME="${MODELS_DIR}" \
            bash tests/special_npu/run_qwen2_5_05b_grpo_mindspeed.sh
        if [ $? -ne 0 ]; then
            echo "ERROR: verl ST failed"
            return 1
        fi
    else
        echo "Skipping verl ST (args_utils.py absent or legacy branch)"
    fi

    # ============================================
    # UT: pytest
    # ============================================
    cd "${workspace}/Megatron-LM"
    echo "===== Running unit tests ====="
    python$PYTHON_VERSION -m pytest --color=no --timeout=1800 -k "not allocator" -x ./tests_extend/unit_tests/
}

# ============================================
# Main: diff-driven gate
# ============================================
WORKSPACE="$1"
pr_id="$2"
branch="$3"
echo "branch=${branch}"

CODE_DIR="${WORKSPACE}/CODE"

cd "${CODE_DIR}"
git diff-tree -r --name-only --no-commit-id "origin/${branch}" HEAD > "${WORKSPACE}/modify.txt"
cat "${WORKSPACE}/modify.txt"

for pattern in "${TRIGGER_PATTERNS[@]}"; do
    if grep -q "${pattern}" "${WORKSPACE}/modify.txt"; then
        echo "CI triggered by change in: ${pattern}"
        run_ut "$WORKSPACE" "$branch"
        exit $?
    fi
done

echo "No CI-trigger path changed. Skipping UT/ST."
exit 0
