#!/bin/bash
set -e

echo "=========================================="
echo "Configuring apt repository..."
echo "=========================================="

if [ -f /etc/os-release ]; then
    source /etc/os-release
else
    echo "ERROR: Cannot find /etc/os-release"
    exit 1
fi

if [ "$ID" != "ubuntu" ]; then
    echo "ERROR: This script is for Ubuntu only"
    exit 1
fi

UBUNTU_CODENAME=$(lsb_release -cs 2>/dev/null || echo "$VERSION_CODENAME")
if [ -z "$UBUNTU_CODENAME" ]; then
    echo "ERROR: Cannot determine Ubuntu version codename"
    exit 1
fi

ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    HUAWEICLOUD_PATH="/ubuntu-ports/"
else
    HUAWEICLOUD_PATH="/ubuntu/"
fi

mkdir -p /etc/apt/sources.list.d/backup
cp /etc/apt/sources.list /etc/apt/sources.list.d/backup/ 2>/dev/null || true

cat > /etc/apt/sources.list <<EOF
deb http://repo.huaweicloud.com${HUAWEICLOUD_PATH} $UBUNTU_CODENAME main restricted universe multiverse
deb-src http://repo.huaweicloud.com${HUAWEICLOUD_PATH} $UBUNTU_CODENAME main restricted universe multiverse

deb http://repo.huaweicloud.com${HUAWEICLOUD_PATH} $UBUNTU_CODENAME-security main restricted universe multiverse
deb-src http://repo.huaweicloud.com${HUAWEICLOUD_PATH} $UBUNTU_CODENAME-security main restricted universe multiverse

deb http://repo.huaweicloud.com${HUAWEICLOUD_PATH} $UBUNTU_CODENAME-updates main restricted universe multiverse
deb-src http://repo.huaweicloud.com${HUAWEICLOUD_PATH} $UBUNTU_CODENAME-updates main restricted universe multiverse
deb http://repo.huaweicloud.com${HUAWEICLOUD_PATH} $UBUNTU_CODENAME-backports main restricted universe multiverse
deb-src http://repo.huaweicloud.com${HUAWEICLOUD_PATH} $UBUNTU_CODENAME-backports main restricted universe multiverse
EOF

MAX_RETRIES=3
for i in $(seq 1 $MAX_RETRIES); do
    echo ">>> apt-get update attempt $i of $MAX_RETRIES"
    if apt-get update; then
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo "WARNING: apt-get update failed after $MAX_RETRIES attempts"
        exit 0
    fi
    sleep 5
done

apt-get install -y ca-certificates
apt-get clean

echo "=========================================="
echo "Apt repository configured successfully!"
echo "=========================================="
