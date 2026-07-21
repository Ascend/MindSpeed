#!/bin/bash
set -e

echo "=========================================="
echo "Configuring repository..."
echo "=========================================="

if [ -f /etc/os-release ]; then
    source /etc/os-release
else
    echo "ERROR: Cannot find /etc/os-release"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$ID" = "ubuntu" ]; then
    echo "[LOG INFO] Detected Ubuntu OS, using apt repository config..."
    bash "${SCRIPT_DIR}/configure_apt_repo.sh"
elif [ "${ID}" = "openEuler" ] || [[ "${ID_LIKE}" =~ "openEuler" ]]; then
    echo "[LOG INFO] Detected openEuler OS, using yum repository config..."
    bash "${SCRIPT_DIR}/configure_yum_repo.sh"
else
    echo "[LOG WARNING] Unknown OS: ${ID}, skipping repo configuration..."
fi

echo "=========================================="
echo "Repository configuration done!"
echo "=========================================="
