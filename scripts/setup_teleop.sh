#!/usr/bin/env bash
# Lightweight environment for GELLO -> Zenoh -> Panda and its MuJoCo viewer.
set -euo pipefail
unset PYTHONPATH
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
interpreter="${TELEOP_PYTHON:-/usr/bin/python3}"
"$interpreter" -c 'import sys; assert sys.version_info >= (3, 10), "Python >=3.10 required"'
if [[ ! -x .venv-teleop/bin/python ]]; then
  "$interpreter" -m venv .venv-teleop
fi
git submodule update --init --depth 1 python/third_party/gello_software/third_party/DynamixelSDK
.venv-teleop/bin/python -m pip install -r scripts/requirements-teleop.txt
.venv-teleop/bin/python -m pip install --no-deps \
  -e python/third_party/gello_software \
  -e python/third_party/gello_software/third_party/DynamixelSDK/python
# Both legacy tools import scripts/gen; use the matching system protoc.
mkdir -p scripts/gen
"${ROBOLAB_PROTOC_EXECUTABLE:-/usr/bin/protoc}" -I proto --python_out=scripts/gen proto/franka.proto
env -u PYTHONPATH .venv-teleop/bin/python -c 'import sys; sys.path.insert(0, "scripts/gen"); import franka_pb2, mujoco, zenoh, dynamixel_sdk, gello'
echo "Ready: ./start_gello.sh --help"
