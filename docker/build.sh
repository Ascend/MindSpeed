#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

NPU_TYPE="910b"
OS="openeuler24.03"
BASE_IMAGE_VERSION="9.0.0-beta.2"
BASE_IMAGE=""
PYTHON_VERSION="3.11"
TORCH_VERSION="2.7.1"
TORCH_NPU_VERSION="2.7.1"
MINDSPEED_BRANCH="master"
MEGATRON_BRANCH="core_v0.12.1"
IMAGE_NAME=""
NO_CACHE=""
CLEANUP_ON_FAIL=false
NPU_TYPE_EXPLICIT=false
OS_EXPLICIT=false

cleanup_dangling() {
    echo ">>> Cleaning up dangling images and corresponding containers..."
    local dangling_images
    dangling_images=$(docker images -f "dangling=true" -q 2>/dev/null || true)
    if [ -n "$dangling_images" ]; then
        for img_id in $dangling_images; do
            local containers
            containers=$(docker ps -a -q --filter "ancestor=$img_id" 2>/dev/null || true)
            if [ -n "$containers" ]; then
                docker rm -f $containers 2>/dev/null || true
            fi
        done
        docker rmi $dangling_images 2>/dev/null || true
    fi
}

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build MindSpeed Core Docker Image

Options:
    -t, --npu-type TYPE       NPU type: a3 or 910b (default: 910b)
    -o, --os OS               OS: openeuler24.03 or ubuntu22.04 (default: openeuler24.03, auto-detected from --base-image if not specified)
    -i, --image-name NAME     Image name (default: mindspeed-core:{branch}-{chip}-{os}-py{py_ver}-{arch})
    -n, --no-cache            Build without cache
    --base-image-version VER  Base image CANN version (default: 9.0.0-beta.2)
    --base-image IMAGE        Full base image name (higher priority than --base-image-version; passed through unchanged)
    --python-version VER      Python tag in the CANN base image (default: 3.11)
    --torch-version VER       PyTorch version (default: 2.7.1)
    --torch-npu-version VER   torch_npu version (default: 2.7.1)
    --mindspeed-branch REF    MindSpeed branch/tag/ref to clone (default: master)
    --megatron-branch REF     Megatron-LM branch/tag/ref to checkout (default: core_v0.12.1)
    --cleanup-on-fail         Clean dangling images/containers if build fails
    -h, --help                Show help

Examples:
    bash $0
    bash $0 -t a3 -o openeuler24.03 --base-image-version 9.0.0-beta.2
    bash $0 --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.2-910b-openeuler24.03-py3.11

Note:
    CANN base image tags use lowercase chip names, such as a3 and 910b. A full --base-image value is used exactly as provided.
EOF
}

parse_base_image_tag() {
    local image="$1"
    local tag="${image##*:}"
    local tag_lower
    tag_lower=$(echo "$tag" | tr '[:upper:]' '[:lower:]')

    if [[ "$tag_lower" == *"910b"* ]]; then
        DETECTED_NPU_TYPE="910b"
    elif [[ "$tag_lower" == *"-a3-"* ]] || [[ "$tag_lower" == *"-a3-py"* ]]; then
        DETECTED_NPU_TYPE="a3"
    fi

    if [[ "$tag_lower" == *"openeuler24.03"* ]]; then
        DETECTED_OS="openeuler24.03"
    elif [[ "$tag_lower" == *"ubuntu22.04"* ]]; then
        DETECTED_OS="ubuntu22.04"
    fi

    if [[ "$tag_lower" =~ py([0-9]+\.[0-9]+) ]]; then
        DETECTED_PYTHON_VERSION="${BASH_REMATCH[1]}"
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--npu-type)        NPU_TYPE="$2"; NPU_TYPE_EXPLICIT=true; shift 2 ;;
        -o|--os)              OS="$2"; OS_EXPLICIT=true; shift 2 ;;
        -i|--image-name)      IMAGE_NAME="$2"; shift 2 ;;
        -n|--no-cache)        NO_CACHE="--no-cache"; shift ;;
        --base-image-version) BASE_IMAGE_VERSION="$2"; shift 2 ;;
        --base-image)         BASE_IMAGE="$2"; shift 2 ;;
        --python-version)     PYTHON_VERSION="$2"; shift 2 ;;
        --torch-version)      TORCH_VERSION="$2"; shift 2 ;;
        --torch-npu-version)  TORCH_NPU_VERSION="$2"; shift 2 ;;
        --mindspeed-branch)   MINDSPEED_BRANCH="$2"; shift 2 ;;
        --megatron-branch)    MEGATRON_BRANCH="$2"; shift 2 ;;
        --cleanup-on-fail)    CLEANUP_ON_FAIL=true; shift ;;
        -h|--help)            show_help; exit 0 ;;
        *)                    echo "Unknown argument: $1"; show_help; exit 1 ;;
    esac
done

if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found: $DOCKERFILE"
    exit 1
fi

DETECTED_NPU_TYPE=""
DETECTED_OS=""
DETECTED_PYTHON_VERSION=""
if [ -n "$BASE_IMAGE" ]; then
    parse_base_image_tag "$BASE_IMAGE"
    if [ "$NPU_TYPE_EXPLICIT" = false ] && [ -n "$DETECTED_NPU_TYPE" ]; then
        NPU_TYPE="$DETECTED_NPU_TYPE"
    fi
    if [ "$OS_EXPLICIT" = false ] && [ -n "$DETECTED_OS" ]; then
        OS="$DETECTED_OS"
    fi
    if [ -n "$DETECTED_PYTHON_VERSION" ]; then
        PYTHON_VERSION="$DETECTED_PYTHON_VERSION"
    fi
fi

NPU_TYPE_LOWER=$(echo "$NPU_TYPE" | tr '[:upper:]' '[:lower:]')
OS=$(echo "$OS" | tr '[:upper:]' '[:lower:]')

if [ "$NPU_TYPE_LOWER" != "a3" ] && [ "$NPU_TYPE_LOWER" != "910b" ]; then
    echo "Error: NPU type must be a3 or 910b"
    exit 1
fi

if [ "$OS" != "ubuntu22.04" ] && [ "$OS" != "openeuler24.03" ]; then
    echo "Error: OS must be ubuntu22.04 or openeuler24.03"
    exit 1
fi

case "$OS" in
    ubuntu*) OS_FAMILY="ubuntu"; REPO_SCRIPT="configure_apt_repo.sh" ;;
    openeuler*) OS_FAMILY="openeuler"; REPO_SCRIPT="configure_yum_repo.sh" ;;
esac

HOST_ARCH=$(uname -m)
case "$HOST_ARCH" in
    arm64) ARCH_NAME="aarch64" ;;
    *)     ARCH_NAME="$HOST_ARCH" ;;
esac
if [ -z "$IMAGE_NAME" ]; then
    TAG_REF=$(echo "$MINDSPEED_BRANCH" | tr '/:' '--')
    IMAGE_NAME="mindspeed-core:${TAG_REF}-${NPU_TYPE_LOWER}-${OS}-py${PYTHON_VERSION}-${ARCH_NAME}"
fi

cd "$SCRIPT_DIR"
cp "${SCRIPT_DIR}/${REPO_SCRIPT}" configure_repo.sh
trap 'rm -f configure_repo.sh' EXIT

BUILD_ARGS="--build-arg OS=${OS}"
BUILD_ARGS="$BUILD_ARGS --build-arg OS_FAMILY=${OS_FAMILY}"
BUILD_ARGS="$BUILD_ARGS --build-arg NPU_TYPE=${NPU_TYPE_LOWER}"
BUILD_ARGS="$BUILD_ARGS --build-arg PYTHON_VERSION=${PYTHON_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg TORCH_VERSION=${TORCH_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg TORCH_NPU_VERSION=${TORCH_NPU_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg MINDSPEED_BRANCH=${MINDSPEED_BRANCH}"
BUILD_ARGS="$BUILD_ARGS --build-arg MEGATRON_BRANCH=${MEGATRON_BRANCH}"

if [ -n "$BASE_IMAGE" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg BASE_IMAGE=${BASE_IMAGE}"
else
    BUILD_ARGS="$BUILD_ARGS --build-arg BASE_IMAGE_VERSION=${BASE_IMAGE_VERSION}"
fi

echo "=========================================="
echo "Build Configuration"
echo "=========================================="
echo "NPU Type:           ${NPU_TYPE_LOWER}"
echo "OS:                 ${OS}"
echo "OS Family:          ${OS_FAMILY}"
echo "CPU Architecture:   ${ARCH_NAME}"
echo "Dockerfile:         ${DOCKERFILE}"
echo "Image Name:         ${IMAGE_NAME}"
echo "Base Image Version: ${BASE_IMAGE_VERSION}"
if [ -n "$BASE_IMAGE" ]; then
    echo "Base Image:         ${BASE_IMAGE}"
fi
echo "Python Version:     ${PYTHON_VERSION}"
echo "PyTorch Version:    ${TORCH_VERSION}"
echo "torch_npu Version:  ${TORCH_NPU_VERSION}"
echo "MindSpeed Ref:      ${MINDSPEED_BRANCH}"
echo "Megatron-LM Ref:    ${MEGATRON_BRANCH}"
echo "No Cache:           ${NO_CACHE:-No}"
echo "=========================================="

set +e
docker build \
    -t "$IMAGE_NAME" \
    -f "$DOCKERFILE" \
    $BUILD_ARGS \
    $NO_CACHE \
    --network=host \
    .
BUILD_RESULT=$?
set -e

if [ $BUILD_RESULT -eq 0 ]; then
    echo "=========================================="
    echo "Build Complete!"
    echo "Image: ${IMAGE_NAME}"
    echo "=========================================="
    exit 0
fi

echo "=========================================="
echo "Build Failed!"
echo "=========================================="
if [ "$CLEANUP_ON_FAIL" = true ]; then
    cleanup_dangling
fi
exit $BUILD_RESULT
