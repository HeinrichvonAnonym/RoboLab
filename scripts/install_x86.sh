#!/usr/bin/env bash
# roboLab — bootstrap build/runtime deps on x86_64 Debian/Ubuntu.
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

echo "==> apt packages (toolchain, CMake, Protobuf, Franka/freenect2 build deps)"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  curl \
  libprotobuf-dev \
  protobuf-compiler \
  libusb-1.0-0-dev \
  libeigen3-dev \
  libpoco-dev \
  libssl-dev \
  libudev-dev

echo "==> Rust / Cargo (required to build bundled zenoh-c unless you use a preinstalled zenohc)"
if ! command -v cargo >/dev/null 2>&1; then
  echo "    Installing rustup (stable) — see https://rustup.rs/"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi
rustup default stable
command -v cargo && cargo --version

echo "==> Optional: Zenoh router (zenohd) — set INSTALL_ZENOHD=1 to: cargo install zenohd --locked"
if [[ "${INSTALL_ZENOHD:-0}" == "1" ]]; then
  cargo install zenohd --locked || echo "zenohd install failed; peer mode works without a router."
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo ""
echo "Done. Next:"
echo "  1. Ensure zenoh-c is available (clone into ${ROOT}/third_party/zenoh-c or set ROBOLAB_ZENOHC_ROOT / CMAKE_PREFIX_PATH)."
echo "  2. From repo root:  bash build.sh"
echo "  3. Run:             ./start.sh"
echo ""
