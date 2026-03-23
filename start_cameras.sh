#!/usr/bin/env bash
# Launch robo_lab_main from the repository root so bringup YAML paths resolve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec ./build/apps/robo_lab_main config/bringup_cameras.yaml "$@"
