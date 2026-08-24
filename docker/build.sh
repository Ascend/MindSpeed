#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

NPU_TYPE="910b"
OS="openeuler24.03"
ARCH=""
BASE_IMAGE_VERSION="9.1.0"
BASE_IMAGE=""
PYTHON_VERSION="3.11"
TORCH_VERSION="2.7.1"
TORCH_NPU_VERSION="2.7.1.post8"
NUMPY_VERSION="1.26.0"
MINDSPEED_BRANCH="v26.1.0_core_r0.12.1"
MEGATRON_BRANCH="core_v0.12.1"
IMAGE_VERSION="v26.1.0_core_r0.12.1"
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
    -t, --npu-type TYPE       NPU type: a3, 910b, or 950 (default: 910b)
    -o, --os OS               OS: openeuler24.03 or ubuntu22.04 (default: openeuler24.03, auto-detected from --base-image if not specified)
    -a, --arch ARCH           Architecture: aarch64 or x86_64 (default: current host architecture)
    -i, --image-name NAME     Image name (default: mindspeed-core:{version}-cann{cann_ver}-torch_npu{torch_npu_ver}-{chip}-{os}-py{py_ver}-{arch})
    -n, --no-cache            Build without cache
    --base-image-version VER  Base image CANN version (default: 9.1.0)
    --base-image IMAGE        Full base image name (higher priority than --base-image-version; passed through unchanged)
    --python-version VER      Python tag in the CANN base image (default: 3.11)
    --torch-version VER       PyTorch version (default: 2.7.1)
    --torch-npu-version VER   torch_npu version (default: 2.7.1.post8)
    --numpy-version VER       NumPy version (default: 1.26.0)
    --mindspeed-branch REF    MindSpeed branch/tag/ref to clone (default: v26.1.0_core_r0.12.1)
    --megatron-branch REF     Megatron-LM branch/tag/ref to checkout (default: core_v0.12.1)
    --image-version VERSION   MindSpeed version field used in the default image tag (default: v26.1.0_core_r0.12.1)
    --cleanup-on-fail         Clean dangling images/containers if build fails
    -h, --help                Show help

Examples:
    bash $0
    bash $0 -t a3 -o openeuler24.03 -a aarch64
    bash $0 -t 950 -o ubuntu22.04 -a x86_64
    bash $0 --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.11

Note:
    CANN base image tags use lowercase chip names: a3, 910b, and 950. A full --base-image value is used exactly as provided.
    Proxy variables from the host environment are forwarded to the build when set: http_proxy, https_proxy, HTTP_PROXY, HTTPS_PROXY, NO_PROXY, and no_proxy.
EOF
}

parse_base_image_tag() {
    local image="$1"
    local tag="${image##*:}"
    local tag_lower
    tag_lower=$(echo "$tag" | tr '[:upper:]' '[:lower:]')

    if [[ "$tag_lower" =~ ^(.+)-(a3|910b|950)-(openeuler24[.]03|ubuntu22[.]04)-py[0-9]+[.][0-9]+$ ]]; then
        DETECTED_BASE_IMAGE_VERSION="${BASH_REMATCH[1]}"
    fi

    if [[ "$tag_lower" == *"910b"* ]]; then
        DETECTED_NPU_TYPE="910b"
    elif [[ "$tag_lower" == *"-950-"* ]] || [[ "$tag_lower" == *"-950-py"* ]]; then
        DETECTED_NPU_TYPE="950"
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
        -a|--arch)            ARCH="$2"; shift 2 ;;
        -i|--image-name)      IMAGE_NAME="$2"; shift 2 ;;
        -n|--no-cache)        NO_CACHE="--no-cache"; shift ;;
        --base-image-version) BASE_IMAGE_VERSION="$2"; shift 2 ;;
        --base-image)         BASE_IMAGE="$2"; shift 2 ;;
        --python-version)     PYTHON_VERSION="$2"; shift 2 ;;
        --torch-version)      TORCH_VERSION="$2"; shift 2 ;;
        --torch-npu-version)  TORCH_NPU_VERSION="$2"; shift 2 ;;
        --numpy-version)      NUMPY_VERSION="$2"; shift 2 ;;
        --mindspeed-branch)   MINDSPEED_BRANCH="$2"; shift 2 ;;
        --megatron-branch)    MEGATRON_BRANCH="$2"; shift 2 ;;
        --image-version)      IMAGE_VERSION="$2"; shift 2 ;;
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
DETECTED_BASE_IMAGE_VERSION=""
if [ -n "$BASE_IMAGE" ]; then
    parse_base_image_tag "$BASE_IMAGE"
    if [ -n "$DETECTED_BASE_IMAGE_VERSION" ]; then
        BASE_IMAGE_VERSION="$DETECTED_BASE_IMAGE_VERSION"
    fi
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

if [ "$NPU_TYPE_LOWER" != "a3" ] && [ "$NPU_TYPE_LOWER" != "910b" ] && [ "$NPU_TYPE_LOWER" != "950" ]; then
    echo "Error: NPU type must be a3, 910b, or 950"
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

if [ -z "$ARCH" ]; then
    ARCH=$(uname -m)
fi
ARCH=$(echo "$ARCH" | tr '[:upper:]' '[:lower:]')
case "$ARCH" in
    arm64|aarch64) ARCH_NAME="aarch64"; TARGET_PLATFORM="linux/arm64" ;;
    amd64|x86_64) ARCH_NAME="x86_64"; TARGET_PLATFORM="linux/amd64" ;;
    *)
        echo "Error: architecture must be aarch64 or x86_64"
        exit 1
        ;;
esac
if [ -z "$IMAGE_NAME" ]; then
    TAG_VERSION=$(echo "$IMAGE_VERSION" | tr '/:' '--')
    IMAGE_NAME="mindspeed-core:${TAG_VERSION}-cann${BASE_IMAGE_VERSION}-torch_npu${TORCH_NPU_VERSION}-${NPU_TYPE_LOWER}-${OS}-py${PYTHON_VERSION}-${ARCH_NAME}"
fi

cd "$SCRIPT_DIR"
cp "${SCRIPT_DIR}/${REPO_SCRIPT}" configure_repo.sh
trap 'rm -f configure_repo.sh' EXIT

BUILD_ARGS=(
    --build-arg "OS=${OS}"
    --build-arg "OS_FAMILY=${OS_FAMILY}"
    --build-arg "NPU_TYPE=${NPU_TYPE_LOWER}"
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
    --build-arg "TORCH_VERSION=${TORCH_VERSION}"
    --build-arg "TORCH_NPU_VERSION=${TORCH_NPU_VERSION}"
    --build-arg "NUMPY_VERSION=${NUMPY_VERSION}"
    --build-arg "MINDSPEED_BRANCH=${MINDSPEED_BRANCH}"
    --build-arg "MEGATRON_BRANCH=${MEGATRON_BRANCH}"
)

if [ -n "$BASE_IMAGE" ]; then
    BUILD_ARGS+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
else
    BUILD_ARGS+=(--build-arg "BASE_IMAGE_VERSION=${BASE_IMAGE_VERSION}")
fi

# Forward host proxy variables without expanding their values onto the command line.
# Docker treats these as predefined build arguments and excludes them from the
# image history by default.
FORWARDED_PROXY_VARS=()
for PROXY_VAR in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY no_proxy; do
    if [ -n "${!PROXY_VAR:-}" ]; then
        BUILD_ARGS+=(--build-arg "$PROXY_VAR")
        FORWARDED_PROXY_VARS+=("$PROXY_VAR")
    fi
done

echo "=========================================="
echo "Build Configuration"
echo "=========================================="
echo "NPU Type:           ${NPU_TYPE_LOWER}"
echo "OS:                 ${OS}"
echo "OS Family:          ${OS_FAMILY}"
echo "CPU Architecture:   ${ARCH_NAME}"
echo "Target Platform:     ${TARGET_PLATFORM}"
echo "Dockerfile:         ${DOCKERFILE}"
echo "Image Name:         ${IMAGE_NAME}"
echo "Base Image Version: ${BASE_IMAGE_VERSION}"
if [ -n "$BASE_IMAGE" ]; then
    echo "Base Image:         ${BASE_IMAGE}"
fi
echo "Python Version:     ${PYTHON_VERSION}"
echo "PyTorch Version:    ${TORCH_VERSION}"
echo "torch_npu Version:  ${TORCH_NPU_VERSION}"
echo "NumPy Version:      ${NUMPY_VERSION}"
echo "MindSpeed Ref:      ${MINDSPEED_BRANCH}"
echo "Megatron-LM Ref:    ${MEGATRON_BRANCH}"
echo "No Cache:           ${NO_CACHE:-No}"
if [ ${#FORWARDED_PROXY_VARS[@]} -gt 0 ]; then
    echo "Proxy Variables:    ${FORWARDED_PROXY_VARS[*]}"
else
    echo "Proxy Variables:    None"
fi
echo "=========================================="

set +e
docker build \
    --platform "$TARGET_PLATFORM" \
    -t "$IMAGE_NAME" \
    -f "$DOCKERFILE" \
    "${BUILD_ARGS[@]}" \
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
