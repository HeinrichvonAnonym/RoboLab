#!/usr/bin/env bash
# roboLab — bootstrap build/runtime deps on Ubuntu 22.04 x86_64 / ROS 2 Humble.
# Run from anywhere:  bash scripts/install_x86.sh
set -euo pipefail

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This script targets x86_64 Linux; detected: $(uname -m)" >&2
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found; install equivalent packages manually (see README.md)." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${INSTALL_SYSTEM:-1}" == "1" ]]; then
  source /etc/os-release
  if [[ "$ID" != ubuntu || "$VERSION_ID" != 22.04 ]]; then
    echo "System dependency installation requires Ubuntu 22.04." >&2
    exit 1
  fi
  echo "==> apt packages (ROS 2 Humble must already be installed)"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ccache \
    git \
    pkg-config \
    curl \
    libprotobuf-dev \
    protobuf-compiler \
    libusb-1.0-0-dev \
    libeigen3-dev \
    libpoco-dev \
    libssl-dev \
    libudev-dev \
    libyaml-cpp-dev \
    libhdf5-dev \
    libturbojpeg0-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libgtest-dev \
    ros-humble-rclcpp \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs

  git -C "$ROOT" submodule update --init --depth 1 third_party/serial

  if [[ "${INSTALL_CUDA:-1}" == "1" ]]; then
    echo "==> CUDA 11.8 compiler and runtime development files (keep the installed driver/kernel)"
    if ! dpkg-query -W -f='${Status}' cuda-keyring 2>/dev/null | grep -q 'install ok installed'; then
      keyring_file="$(mktemp --suffix=.deb)"
      trap 'rm -f "$keyring_file"' EXIT
      curl --proto '=https' --tlsv1.2 -fL --retry 3 \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
        -o "$keyring_file"
      sudo dpkg -i "$keyring_file"
      rm -f "$keyring_file"
      trap - EXIT
    fi
    sudo apt-get update
    # Kinect only uses nvcc and cudart. The full toolkit pulls in multi-GB math
    # libraries, profilers and IDEs that neither Kinect nor this default build uses.
    sudo apt-get install -y --no-install-recommends cuda-nvcc-11-8 cuda-cudart-dev-11-8
    # libfreenect2 includes helper_math.h from this already-pinned submodule.
    git -C "$ROOT" submodule update --init --depth 1 third_party/cuda-samples
  fi
fi

if [[ "${INSTALL_RUST:-1}" == "1" ]]; then
  echo "==> Rust 1.93.0 / Cargo for zenoh-c 1.8.0"
  if [[ -f "${HOME}/.cargo/env" ]]; then source "${HOME}/.cargo/env"; fi
  if ! command -v rustup >/dev/null 2>&1; then
    rustup_script="$(mktemp)"
    trap 'rm -f "$rustup_script"' EXIT
    curl --proto '=https' --tlsv1.2 -fsSL --retry 3 https://sh.rustup.rs -o "$rustup_script"
    sh "$rustup_script" -y --profile minimal --default-toolchain 1.93.0 --no-modify-path
    rm -f "$rustup_script"
    trap - EXIT
    source "${HOME}/.cargo/env"
  fi
  rustup toolchain install 1.93.0 --profile minimal
  cargo +1.93.0 --version
fi

# Peer communication does not require a router.
if [[ "${INSTALL_ZENOHD:-0}" == "1" ]]; then
  cargo +1.93.0 install zenohd --version 1.8.0 --locked
fi

echo ""
echo "Done. Next:"
echo "  1. source /opt/ros/humble/setup.bash"
echo "  2. From repo root:  bash build.sh"
echo "  3. Run:             ./start.sh"
echo ""
