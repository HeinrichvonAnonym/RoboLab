#!/usr/bin/env python3
"""Replay Franka joint trajectory from recorder HDF5 into franka_plugin command topic.

Reads only `franka_state/joints_position` from a recorded .h5 file and publishes
each row as a `franka.RobotCommand` to Zenoh.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _ensure_franka_pb2(repo_root: Path) -> None:
    gen = repo_root / "scripts" / "gen"
    pb2 = gen / "franka_pb2.py"
    proto = repo_root / "proto" / "franka.proto"
    if pb2.is_file():
        return
    gen.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["protoc", "-I", str(repo_root / "proto"), f"--python_out={gen}", str(proto)],
        check=True,
    )


def _load_trajectory(h5_path: Path):
    try:
        import h5py
        import numpy as np
    except ImportError:
        print("Missing dependency: pip install h5py numpy", file=sys.stderr)
        raise SystemExit(1)

    if not h5_path.is_file():
        print(f"HDF5 file not found: {h5_path}", file=sys.stderr)
        raise SystemExit(1)

    with h5py.File(h5_path, "r") as f:
        if "franka_state" not in f:
            print("Missing group 'franka_state' in HDF5", file=sys.stderr)
            raise SystemExit(1)
        g = f["franka_state"]
        if "joints_position" not in g:
            print("Missing dataset 'franka_state/joints_position'", file=sys.stderr)
            raise SystemExit(1)

        q = np.asarray(g["joints_position"])
        if q.ndim != 2 or q.shape[1] != 7:
            print(
                f"Expected joints_position shape (N,7), got {q.shape}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        ts = None
        if "timestamps_ns" in g:
            ts = np.asarray(g["timestamps_ns"])
            if ts.shape[0] != q.shape[0]:
                ts = None
        return q, ts


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay franka_state joint positions from HDF5.")
    parser.add_argument(
        "h5",
        help="Path to recorder HDF5 file (e.g. data/recording_xxx.h5)",
    )
    parser.add_argument(
        "--key",
        default="franka/command",
        help="Zenoh publish key (must match franka plugin cmd_topic)",
    )
    parser.add_argument(
        "--mode",
        default="position",
        help="RobotCommand.mode",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed factor for timestamp-based replay (2.0 = 2x faster)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=0.0,
        help="If >0, ignore timestamps and publish at fixed rate",
    )
    parser.add_argument(
        "--connect",
        "-e",
        action="append",
        default=None,
        metavar="ENDPOINT",
        help="Zenoh endpoint (repeatable), e.g. tcp/127.0.0.1:7447",
    )
    parser.add_argument(
        "--cmd-filter-alpha",
        type=float,
        default=1.0,
        help="EMA filter on joint command (0<alpha<=1). 1.0 disables filtering.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_franka_pb2(repo_root)
    sys.path.insert(0, str(repo_root / "scripts" / "gen"))

    try:
        import franka_pb2
        import zenoh
        import numpy as np
    except ImportError as e:
        print("Missing dependency: pip install eclipse-zenoh protobuf numpy", file=sys.stderr)
        raise SystemExit(1) from e

    q, ts = _load_trajectory(Path(args.h5))
    n = int(q.shape[0])
    if n == 0:
        print("No trajectory samples in HDF5", file=sys.stderr)
        return 1

    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", json.dumps(list(args.connect)))

    fixed_dt = (1.0 / args.hz) if args.hz > 0 else None
    speed = max(1e-6, float(args.speed))
    alpha = float(args.cmd_filter_alpha)
    if not (0.0 < alpha <= 1.0):
        print("--cmd-filter-alpha must be in (0, 1]", file=sys.stderr)
        return 1

    print(
        f"Replay samples={n}, key='{args.key}', mode='{args.mode}', "
        f"speed={speed}, hz={args.hz}, cmd_filter_alpha={alpha}"
    )

    with zenoh.open(conf) as session:
        pub = session.declare_publisher(args.key)
        t_start = time.perf_counter()
        filtered_row = None

        for i in range(n):
            cmd = franka_pb2.RobotCommand()
            cmd.type = 2  # TYPE_JOINT_TARGET
            cmd.sequence = i + 1
            cmd.mode = args.mode
            cmd.note = "franka_arm_traj_replay"
            # Keep compatibility with older generated franka_pb2 that may not
            # include sys_time yet.
            if hasattr(cmd, "sys_time"):
                cmd.sys_time = float(i)

            cmd.ClearField("joints")
            row = q[i]
            if filtered_row is None:
                filtered_row = row.astype(float, copy=True)
            else:
                # Exponential moving average to smooth command trajectory.
                filtered_row = alpha * row + (1.0 - alpha) * filtered_row
            for j in range(7):
                jc = cmd.joints.add()
                jc.position = float(filtered_row[j])
                jc.velocity = 0.0
                jc.effort = 0.0

            pub.put(cmd.SerializeToString())

            if i + 1 < n:
                if fixed_dt is not None:
                    time.sleep(max(0.0, fixed_dt))
                elif ts is not None:
                    dt_ns = int(ts[i + 1]) - int(ts[i])
                    if dt_ns > 0:
                        time.sleep((dt_ns * 1e-9) / speed)

            if i % 50 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"Published {i+1}/{n} samples (elapsed {elapsed:.2f}s)")

    print("Replay finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
