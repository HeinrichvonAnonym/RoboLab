#!/usr/bin/env bash
# Publish GELLO commands using the existing calibrated Panda implementation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
python_command="${ROBOLAB_PYTHON:-$ROOT/.venv-teleop/bin/python}"
if ! env -u PYTHONPATH "$python_command" -c 'import gello, tyro, zenoh, dynamixel_sdk' >/dev/null 2>&1; then
  echo "start_gello.sh: run bash scripts/setup_teleop.sh first (or set ROBOLAB_PYTHON)" >&2
  exit 1
fi
exec env -u PYTHONPATH "$python_command" -u \
  "$ROOT/python/third_party/gello_software/experiments/gello_publish_zenoh.py" "$@"
