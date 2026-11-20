#!/usr/bin/env bash
# Launch robo_lab_main from the repository root so bringup YAML paths resolve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/build/apps/robo_lab_main" ]]; then
  echo "start.sh: executable missing; run bash build.sh first" >&2
  exit 1
fi

launch_mujoco_viewer() {
  local viewer_cmd terminal_cmd
  local -a python_command
  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "start.sh: no graphical display; skipping optional MuJoCo viewer" >&2
    return
  fi
  # An explicitly selected legacy Conda environment is still supported.
  # The default no longer assumes that an environment named roboLab exists.
  if [[ -n "${ROBOLAB_MUJOCO_CONDA_ENV:-}" ]]; then
    local conda_command="${CONDA_EXE:-}"
    if [[ -z "$conda_command" ]]; then conda_command="$(command -v conda || true)"; fi
    if [[ ! -x "$conda_command" ]]; then
      echo "start.sh: selected Conda environment requires conda; skipping optional viewer" >&2
      return
    fi
    python_command=("$conda_command" run --no-capture-output -n "$ROBOLAB_MUJOCO_CONDA_ENV" python3)
  elif [[ -n "${ROBOLAB_PYTHON:-}" ]]; then
    python_command=("$ROBOLAB_PYTHON")
  elif [[ -x "$ROOT/.venv-teleop/bin/python" ]]; then
    python_command=("$ROOT/.venv-teleop/bin/python")
  else
    python_command=(python3)
  fi
  if ! env -u PYTHONPATH "${python_command[@]}" -c 'import mujoco, zenoh, google.protobuf, yaml' >/dev/null 2>&1; then
    echo "start.sh: viewer Python/dependencies unavailable (${python_command[*]}). Run bash scripts/setup_teleop.sh; continuing without the optional viewer." >&2
    return
  fi
  printf -v viewer_cmd '%q ' env -u PYTHONPATH "${python_command[@]}" "$ROOT/scripts/mujoco_franka_state_viewer.py" --bringup "${1:-config/bringup_franka_config.yaml}"
  viewer_cmd="exec $viewer_cmd"

  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal -- bash -c "$viewer_cmd" &
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -e bash -c "$viewer_cmd" &
  elif command -v xfce4-terminal >/dev/null 2>&1; then
    printf -v terminal_cmd 'bash -c %q' "$viewer_cmd"
    xfce4-terminal --command "$terminal_cmd" &
  elif command -v konsole >/dev/null 2>&1; then
    konsole -e bash -c "$viewer_cmd" &
  elif command -v xterm >/dev/null 2>&1; then
    xterm -e bash -c "$viewer_cmd" &
  else
    echo "start.sh: no supported terminal found; MuJoCo viewer not launched" >&2
  fi
}

if [[ "${ROBOLAB_SKIP_MUJOCO_VIEWER:-0}" != "1" ]]; then
  launch_mujoco_viewer "$@"
fi

exec ./build/apps/robo_lab_main "$@"
