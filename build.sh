#!/usr/bin/env bash
# Configure and build incrementally. Arguments are forwarded to CMake configuration.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ -f "${HOME}/.cargo/env" ]]; then source "${HOME}/.cargo/env"; fi
cpus="$(nproc 2>/dev/null || echo 4)"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$((cpus < 8 ? cpus : 8))}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$((cpus < 4 ? cpus : 4))}"
cmake -S "$ROOT" -B "$ROOT/build" "$@"
cmake --build "$ROOT/build"
