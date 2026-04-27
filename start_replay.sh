#!/usr/bin/env bash
# Launch the replay bringup (franka_plugin + franka_replayer_plugin) from the
# repository root so YAML paths resolve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec ./build/apps/robo_lab_main ./config/start_replay.yaml "$@"
