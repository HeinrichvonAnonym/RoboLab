#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
cd "$(dirname "$0")/.."
if [[ -n "${INFERENCE_PYTHON:-}" ]]; then
  interpreter="$INFERENCE_PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  interpreter="$(command -v python3.12)"
elif [[ -x .venv-lerobot/bin/python ]]; then
  # Reuse only the interpreter to create a separate environment; no packages
  # are installed in, or inherited from, the LeRobot environment.
  interpreter="$(readlink -f .venv-lerobot/bin/python)"
else
  interpreter=python3
fi
"$interpreter" -c 'import sys; assert sys.version_info >= (3, 10), "Python >=3.10 required; set INFERENCE_PYTHON"'
if [[ ! -x .venv-inference/bin/python ]]; then
  "$interpreter" -m venv .venv-inference
fi
.venv-inference/bin/python -m pip install -r scripts/requirements-inference.lock
printf '%s\n' 'Ready: .venv-inference/bin/python scripts/run_python_inference.py --help'
