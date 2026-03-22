#!/usr/bin/env python3
"""Publish a demo RobotCommand (protobuf) on Zenoh for robo_lab Franka plugin.

Requires:
  pip install eclipse-zenoh protobuf

Regenerate Python stubs after changing proto/ (or if missing):
  rm -f scripts/gen/franka_pb2.py && bash scripts/regen_proto_py.sh

Run robo_lab_main with bringup first, then from the repo root:

  python3 scripts/publish_demo_command.py
  python3 scripts/publish_demo_command.py -n 5 --interval 0.5
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
    try:
        subprocess.run(
            ["protoc", "-I", str(repo_root / "proto"), f"--python_out={gen}", str(proto)],
            check=True,
        )
    except FileNotFoundError as e:
        print(
            "protoc not found. Install protobuf-compiler or run: bash scripts/regen_proto_py.sh",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    except subprocess.CalledProcessError as e:
        print("protoc failed; try: bash scripts/regen_proto_py.sh", file=sys.stderr)
        raise SystemExit(1) from e


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--key",
        default="robot/command",
        help="Zenoh key expression (must match topic in config/plugins/franka_plugin_config.yaml)",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=1,
        help="Number of publishes (default 1)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        help="Seconds between publishes when count > 1",
    )
    parser.add_argument(
        "--connect",
        "-e",
        action="append",
        default=None,
        metavar="ENDPOINT",
        help="Zenoh connect endpoint (repeatable), e.g. tcp/127.0.0.1:7447",
    )
    parser.add_argument(
        "--mode",
        default="position",
        help="RobotCommand.mode string",
    )
    parser.add_argument(
        "--note",
        default="publish_demo_command.py",
        help="RobotCommand.note string",
    )
    parser.add_argument(
        "--joint-pos",
        nargs="*",
        type=float,
        default=None,
        metavar="P",
        help="Optional joint position targets (one JointCommand per value). Default: seven zeros (7-DOF).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_franka_pb2(repo_root)
    sys.path.insert(0, str(repo_root / "scripts" / "gen"))

    try:
        import franka_pb2
    except ImportError as e:
        print("Could not import franka_pb2 after generation.", file=sys.stderr)
        raise SystemExit(1) from e

    try:
        import zenoh
    except ImportError:
        print("Missing dependency: pip install eclipse-zenoh protobuf", file=sys.stderr)
        return 1

    conf = zenoh.Config()
    if args.connect:
        eps = json.dumps(list(args.connect))
        conf.insert_json5("connect/endpoints", eps)

    try:
        with zenoh.open(conf) as session:
            pub = session.declare_publisher(args.key)
            for k in range(args.count):
                cmd = franka_pb2.RobotCommand()
                cmd.type = 1  # RobotCommand.Type.TYPE_DEMO
                cmd.sequence = k + 1
                cmd.mode = args.mode
                cmd.note = args.note
                positions = args.joint_pos if args.joint_pos else [0.0] * 7
                cmd.ClearField("joints")
                for p in positions:
                    jc = cmd.joints.add()
                    jc.position = p
                    jc.velocity = 0.0
                    jc.effort = 0.0
                payload = cmd.SerializeToString()
                pub.put(payload)
                print(
                    f"put '{args.key}' RobotCommand seq={cmd.sequence} joints={len(positions)} bytes={len(payload)}"
                )
                if k + 1 < args.count and args.interval > 0:
                    time.sleep(args.interval)
    except Exception as ex:
        print(f"zenoh error: {ex}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
