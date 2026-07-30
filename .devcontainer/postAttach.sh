#!/bin/bash
# 每次新建终端 / attach 时执行（幂等）

# ── 1. CANN 环境 source 注入到 .bashrc ──────────────────────────────────────
grep -q "ascend-toolkit/set_env.sh" ~/.bashrc 2>/dev/null || \
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
grep -q "nnal/atb/set_env.sh" ~/.bashrc 2>/dev/null || \
    echo "source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null" >> ~/.bashrc

# ── 2. 写入快捷 cd alias（若 MindSpeed 不在 /workspace 根目录） ──────────
if [ -f /tmp/.mindspeed_root ]; then
    MINDSPEED_ROOT=$(cat /tmp/.mindspeed_root)
    # 校验路径不包含单引号等 shell 元字符，防止注入
    if [[ "${MINDSPEED_ROOT}" =~ [\047] ]]; then
        echo "ERROR: /tmp/.mindspeed_root contains single quote (') — refusing to write alias" >&2
    elif [ "${MINDSPEED_ROOT}" != "/workspace" ]; then
        grep -q "alias cdm=" ~/.bashrc 2>/dev/null || \
            echo "alias cdm='cd ${MINDSPEED_ROOT}'  # 快速跳转到 MindSpeed 根目录" >> ~/.bashrc
    fi
fi
