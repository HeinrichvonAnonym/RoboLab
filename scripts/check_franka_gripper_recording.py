#!/usr/bin/env python3
"""Replay recorded gripper targets in real time against the hardware-free C++ worker.

Build first:
  cmake -S . -B build -DROBOLAB_BUILD_GRIPPER_TESTS=ON
  cmake --build build --target gripper_worker_test
Then run with the roboLab Python environment (requires h5py and numpy).
No Zenoh sessions, robot devices, or recording files are modified.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", nargs="?", type=Path, help="Default: latest completed data/*.h5")
    parser.add_argument("--binary", type=Path, default=root / "build/src/plugins/gripper_worker_test")
    args = parser.parse_args()
    recording = args.recording
    if recording is None:
        candidates = list((root / "data").glob("*.h5"))
        if not candidates:
            parser.error("No completed recordings in data/")
        recording = max(candidates, key=lambda p: p.stat().st_mtime)
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"Test binary missing: {binary}; build gripper_worker_test first")
    with h5py.File(recording, "r") as data:
        command = data["franka_command/joints_position"]
        if command.ndim != 2 or command.shape[1] != 8:
            parser.error("Recording must contain eight-dimensional Franka commands")
        timestamps = data["franka_command/timestamps_ns"][:]
        gripper = command[:, 7]
    if (len(timestamps) == 0 or len(timestamps) != len(gripper)
            or np.any(np.diff(timestamps) < 0) or not np.isfinite(gripper).all()):
        parser.error("Recording has empty, mismatched, non-monotonic or non-finite commands")
    seconds = (timestamps - timestamps[0]).astype(np.float64) / 1e9
    transitions = np.count_nonzero(np.diff(gripper >= 0.5))
    print(f"Recording: {recording}\nCommands: {len(gripper)}, threshold transitions: {transitions}, "
          f"duration: {seconds[-1]:.3f}s (real-time replay)", flush=True)
    with tempfile.TemporaryDirectory(prefix="robolab-gripper-replay-") as temporary:
        trace = Path(temporary) / "targets.txt"
        np.savetxt(trace, np.column_stack((seconds, gripper)), fmt="%.17g")
        subprocess.run([str(binary), "--replay", str(trace)], check=True)


if __name__ == "__main__":
    main()
