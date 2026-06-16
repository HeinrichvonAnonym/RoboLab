#!/usr/bin/env python3
"""Publish GELLO joint targets as franka.RobotCommand over Zenoh.

This script only turns the GELLO device into a command publisher. It does not
subscribe to robot state, instantiate RobotEnv, home the robot, or make control
decisions. The existing roboLab control stack remains responsible for consuming
the command topic and enforcing robot-side safety.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.agents.gello_agent import GelloAgent


def _repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "proto" / "franka.proto").is_file():
            return path
    raise RuntimeError("Could not find repo root containing proto/franka.proto")


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
    except FileNotFoundError as exc:
        raise RuntimeError(
            "protoc not found. Install protobuf-compiler or run "
            "bash scripts/regen_proto_py.sh from the repo root."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("protoc failed; try bash scripts/regen_proto_py.sh") from exc


@dataclass
class Args:
    gello_port: str
    cmd_topic: str = "franka/command"
    mode: str = "position"
    hz: float = 100.0
    connect: Optional[Tuple[str, ...]] = None
    start_joints: Optional[Tuple[float, ...]] = None
    note: str = "gello_publish_zenoh.py"
    dry_run: bool = False

    def __post_init__(self):
        if self.start_joints is not None:
            self.start_joints = np.array(self.start_joints)


def main(args: Args) -> int:
    repo_root = _repo_root()
    _ensure_franka_pb2(repo_root)
    sys.path.insert(0, str(repo_root / "scripts" / "gen"))

    try:
        import franka_pb2
    except ImportError as exc:
        raise RuntimeError("Could not import generated franka_pb2") from exc

    try:
        import zenoh
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: pip install 'eclipse-zenoh<1.8' protobuf"
        ) from exc

    if args.hz <= 0:
        raise ValueError("--hz must be positive")

    agent = GelloAgent(port=args.gello_port, start_joints=args.start_joints)
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", json.dumps(list(args.connect)))

    period = 1.0 / args.hz
    sequence = 0
    print(
        f"Publishing GELLO arm joints to '{args.cmd_topic}' at {args.hz:g} Hz "
        f"(dry_run={args.dry_run})"
    )

    with zenoh.open(conf) as session:
        publisher = None if args.dry_run else session.declare_publisher(args.cmd_topic)
        next_time = time.monotonic()
        while True:
            command = np.asarray(agent.act({}), dtype=float)
            if command.shape != (7,) and command.shape != (8,):
                raise ValueError(
                    f"GELLO action must contain 7 arm joints or 8 arm+gripper values, "
                    f"got shape {command.shape}"
                )
            arm = command[:7]
            gripper = float(command[7]) if command.shape[0] == 8 else None

            cmd = franka_pb2.RobotCommand()
            cmd.type = franka_pb2.RobotCommand.TYPE_JOINT_TARGET
            sequence += 1
            cmd.sequence = sequence
            cmd.mode = args.mode
            cmd.note = args.note
            if hasattr(cmd, "sys_time"):
                cmd.sys_time = float(time.time())
            cmd.ClearField("joints")
            for position in command:
                joint = cmd.joints.add()
                joint.position = float(position)
                joint.velocity = 0.0
                joint.effort = 0.0

            if publisher is not None:
                publisher.put(cmd.SerializeToString())
            if sequence % max(1, int(args.hz)) == 0:
                if gripper is None:
                    print(f"seq={sequence} q={np.array2string(arm, precision=3)}")
                else:
                    print(
                        f"seq={sequence} q={np.array2string(arm, precision=3)} "
                        f"gripper={gripper:.3f}"
                    )

            next_time += period
            time.sleep(max(0.0, next_time - time.monotonic()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
