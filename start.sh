#!/usr/bin/env bash
# Launch robo_lab_main from the repository root so bringup YAML paths resolve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

launch_mujoco_viewer() {
  local viewer_cmd
  local conda_env="${ROBOLAB_MUJOCO_CONDA_ENV:-roboLab}"
  printf -v viewer_cmd 'cd %q && conda run --no-capture-output -n %q python3 scripts/mujoco_franka_state_viewer.py; exec bash' "$ROOT" "$conda_env"

  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal -- bash -lc "$viewer_cmd" &
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -e bash -lc "$viewer_cmd" &
  elif command -v xfce4-terminal >/dev/null 2>&1; then
    xfce4-terminal --command "bash -lc '$viewer_cmd'" &
  elif command -v konsole >/dev/null 2>&1; then
    konsole -e bash -lc "$viewer_cmd" &
  elif command -v xterm >/dev/null 2>&1; then
    xterm -e bash -lc "$viewer_cmd" &
  else
    echo "start.sh: no supported terminal found; MuJoCo viewer not launched" >&2
  fi
}

if [[ "${ROBOLAB_SKIP_MUJOCO_VIEWER:-0}" != "1" ]]; then
  launch_mujoco_viewer
fi

exec ./build/apps/robo_lab_main "$@"
