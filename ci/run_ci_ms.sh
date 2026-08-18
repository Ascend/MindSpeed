#!/bin/bash
# ============================================
# MindSpeed CI Entry Script (MindSpore UT Gate)
#
# Architecture:
#   CI automation creates ${WORKSPACE} and places the PR-merged
#   MindSpeed repo at ${WORKSPACE}/CODE/. This script assembles
#   the MindSpore ecosystem (MindSpeed-Core-MS, MindSpeed-LLM,
#   Megatron-LM, msadapter) and runs MindSpore unit tests.
#
#   Dependencies not in the Docker image are cloned at runtime:
#     - MindSpeed-Core-MS (gitcode)
#     - MindSpeed-LLM      (gitcode, pinned commit)
#     - Megatron-LM        (gitee, core_v0.12.1)
#     - msadapter          (/home/msadapter, pinned commit)
#     - MindSpeed          (from ${WORKSPACE}/CODE/)
# ============================================
set -e

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
# Main MindSpore UT runner
# --------------------------------------------------
run_ms_ut() {
    local workspace="$1"
    local password="$2"
    local code_dir="${workspace}/CODE"

    # --- Ascend environment (external scripts may not be -u safe) ---
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    set -u
    export ENABLE_ATB=1

    cd "${workspace}"

    # ============================================
    # 1. Clone MindSpeed-Core-MS
    # ============================================
    echo "===== Cloning MindSpeed-Core-MS ====="
    rm -rf MindSpeed-Core-MS/
    git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master
    if [ $? -ne 0 ]; then
        echo "Error: git clone MindSpeed-Core-MS"
        exit 1
    fi
    cd MindSpeed-Core-MS
    MindSpeed_Core_MS_PATH=$(pwd)
    cd "${workspace}"
    echo "...............................................done MindSpeed-Core-MS"

    # ============================================
    # 2. Copy MindSpeed from PR-merged code
    # ============================================
    echo "===== Copying MindSpeed from CODE ====="
    cd "${MindSpeed_Core_MS_PATH}"
    rm -rf MindSpeed/
    cp -r "${code_dir}" ./MindSpeed
    if [ $? -ne 0 ]; then
        echo "Error: cp MindSpeed"
        exit 1
    fi
    echo "...............................................done MindSpeed"

    # ============================================
    # 3. Clone MindSpeed-LLM
    # ============================================
    echo "===== Cloning MindSpeed-LLM ====="
    cd "${MindSpeed_Core_MS_PATH}"
    rm -rf MindSpeed-LLM/
    git clone https://gitcode.com/Ascend/MindSpeed-LLM.git -b master
    if [ $? -ne 0 ]; then
        echo "Error: git clone MindSpeed-LLM"
        exit 1
    fi
    cd MindSpeed-LLM
    git checkout 15fdb9b71b0c23ea2dc684e1cbb5d302500bfa39
    cd ..
    rm -rf MindSpeed-LLM/tests
    echo "...............................................done MindSpeed-LLM"

    # ============================================
    # 4. Clone Megatron-LM
    # ============================================
    echo "===== Cloning Megatron-LM ====="
    cd "${MindSpeed_Core_MS_PATH}"
    rm -rf Megatron-LM/
    git clone https://gitee.com/mirrors/Megatron-LM.git -b core_v0.12.1
    if [ $? -ne 0 ]; then
        echo "Error: git clone Megatron-LM"
        exit 1
    fi
    rm -rf Megatron-LM/tests
    echo "...............................................done Megatron-LM"

    # ============================================
    # 5. Prepare msadapter
    # ============================================
    echo "===== Preparing msadapter ====="
    cd "${MindSpeed_Core_MS_PATH}"
    rm -rf msadapter/
    cp -rf /home/msadapter/ ./msadapter
    if [ $? -ne 0 ]; then
        echo "Error: cp msadapter"
        exit 1
    fi
    cd msadapter
    git checkout 951a8218d4c29785e48f304e720212b57056573e
    cd ..
    rm -rf msadapter/tests
    echo "...............................................done msadapter"

    # ============================================
    # 6. Install Python dependencies
    # ============================================
    echo "===== Installing Python dependencies ====="
    pip install mindspore==2.8.0 -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple
    pip install ml_dtypes -i https://repo.huaweicloud.com/repository/pypi/simple --no-deps
    pip install transformers==4.51.0

    # ============================================
    # 7. Set PYTHONPATH
    # ============================================
    echo "===== Setting PYTHONPATH ====="
    export PYTHONPATH="${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/Megatron-LM:$PYTHONPATH"
    export PYTHONPATH="${MindSpeed_Core_MS_PATH}/msadapter/:${MindSpeed_Core_MS_PATH}/msadapter/msa_thirdparty/:$PYTHONPATH"
    echo "PYTHONPATH=${PYTHONPATH}"
    echo "...............................................done PYTHONPATH"

    # ============================================
    # 8. Run MindSpore unit tests
    # ============================================
    echo "===== Running MindSpore unit tests ====="
    cd "${MindSpeed_Core_MS_PATH}/Megatron-LM"
    pytest ../MindSpeed/tests_extend/mindspore/unit_tests/ -s -v -x --ai-framework mindspore
    if [ $? -ne 0 ]; then
        echo "ERROR: MindSpore unit tests failed"
        return 1
    fi
    echo "===== MindSpore unit tests passed ====="
}

# ============================================
# Main entry point
# ============================================
WORKSPACE="$1"
password="$2"

echo "Workspace: ${WORKSPACE}"

if [ -z "${WORKSPACE}" ]; then
    echo "Usage: $0 <workspace> [password]"
    exit 1
fi

run_ms_ut "${WORKSPACE}" "${password}"
exit $?
