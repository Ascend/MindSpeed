#!/bin/bash
# ============================================
# MindSpeed CI Entry Script (UT Gate)
#
# Architecture:
#   CI automation creates ${WORKSPACE} and places the PR-merged
#   MindSpeed repo at ${WORKSPACE}/CODE/. This script installs
#   that copy and runs unit tests (pytest) only.
#
#   The Docker image (mindspeed-ci) provides only dependencies:
#     /mindspeed_ci_deps/Megatron-LM
#     /home/models
#
#   MindSpeed itself is NOT installed in the image; it comes
#   from ${WORKSPACE}/CODE/ and is pip-installed at runtime.
# ============================================
set -e

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
MEGATRON_DIR="/mindspeed_ci_deps/Megatron-LM"
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
# Main UT runner
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
    # UT: pytest
    # ============================================
    echo "===== Running unit tests ====="
    python$PYTHON_VERSION -m pytest --color=no --timeout=1800 -k "not allocator" -x ./tests_extend/unit_tests/
    if [ $? -ne 0 ]; then
        echo "ERROR: Unit tests failed"
        return 1
    fi
    echo "===== Unit tests passed ====="
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

echo "No CI-trigger path changed. Skipping UT."
exit 0
