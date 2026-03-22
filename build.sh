#!/usr/bin/env bash
# Configure and build from repository root (run after cmake deps + zenoh-c are available).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ ! -d build ]]; then
  mkdir build
fi
cd build
cmake ..
make -j"$(nproc 2>/dev/null || echo 4)"
cd ..
