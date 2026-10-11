#!/usr/bin/env python3
"""Publish GELLO joint targets as franka.RobotCommand over Zenoh.

Arm positions are unwrapped continuously and clamped to the Panda joint limits
before publication (including in dry-run mode). Positions remain in radians;
the optional eighth value retains its normalized gripper meaning.

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


# Franka Panda nominal position limits (rad), matching
# config/cartesian_control/cartesian_control_config.yaml.
PANDA_JOINT_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
)
PANDA_JOINT_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
)


class PandaJointPositionNormalizer:
    """Remove turns, preserve continuity, then clamp only the seven arm joints.

    Consecutive valid samples must differ physically by less than pi radians.
    Keep the unclamped angle as state so returning from a joint limit does not
    introduce an offset. Invalid samples never change this state.
    """

    def __init__(self) -> None:
        self._previous_phase: Optional[np.ndarray] = None
        self._continuous: Optional[np.ndarray] = None

    def normalize(self, action) -> Tuple[np.ndarray, Tuple[int, ...]]:
        command = np.asarray(action, dtype=float)
        if command.shape not in ((7,), (8,)):
            raise ValueError(
                "GELLO action must contain 7 arm joints or 8 arm+gripper values, "
                f"got shape {command.shape}"
            )
        if not np.all(np.isfinite(command)):
            raise ValueError("GELLO action contains NaN or Inf")

        arm = command[:7]
        turn = 2.0 * np.pi
        # Reduce before subtracting to avoid overflow for large finite inputs.
        phase = np.remainder(arm, turn)
        if self._previous_phase is None:
            center = (PANDA_JOINT_LOWER + PANDA_JOINT_UPPER) / 2.0
            # The equivalent angle nearest the interval center is also nearest
            # the interval. A tie uses the lower branch, like [-pi, pi).
            continuous = center + np.remainder(phase - center + np.pi, turn) - np.pi
            # Preserve already legal first-frame values exactly, including
            # endpoints and joint 6 values above pi.
            legal = (arm >= PANDA_JOINT_LOWER) & (arm <= PANDA_JOINT_UPPER)
            continuous = np.where(legal, arm, continuous)
        else:
            delta = np.remainder(phase - self._previous_phase + np.pi, turn) - np.pi
            continuous = self._continuous + delta

        result = command.copy()
        result[:7] = np.clip(continuous, PANDA_JOINT_LOWER, PANDA_JOINT_UPPER)
        clipped = tuple(int(index) + 1 for index in np.flatnonzero(result[:7] != continuous))
        self._previous_phase = phase
        self._continuous = continuous
        return result, clipped


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

    if not np.isfinite(args.hz) or args.hz <= 0:
        raise ValueError("--hz must be finite and positive")

    agent = GelloAgent(port=args.gello_port, start_joints=args.start_joints)
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", json.dumps(list(args.connect)))

    period = 1.0 / args.hz
    sequence = 0
    normalizer = PandaJointPositionNormalizer()
    last_status = time.monotonic()
    last_error = float("-inf")
    invalid_frames = 0
    clipped_since_log = set()
    print(
        f"Publishing GELLO arm joints to '{args.cmd_topic}' at {args.hz:g} Hz "
        f"(dry_run={args.dry_run}, joint_pos_norm=True, joint_limits=Panda, units=rad)"
    )

    with zenoh.open(conf) as session:
        publisher = None if args.dry_run else session.declare_publisher(args.cmd_topic)
        next_time = time.monotonic()
        while True:
            action = agent.act({})
            try:
                command, clipped = normalizer.normalize(action)
            except (ValueError, TypeError, OverflowError) as exc:
                invalid_frames += 1
                now = time.monotonic()
                if now - last_error >= 1.0:
                    print(
                        f"Dropping invalid GELLO frame (total={invalid_frames}): {exc}",
                        file=sys.stderr,
                    )
                    last_error = now
                next_time += period
                time.sleep(max(0.0, next_time - time.monotonic()))
                continue
            raw_arm = np.asarray(action, dtype=float)[:7]
            arm = command[:7]
            gripper = float(command[7]) if command.shape[0] == 8 else None
            clipped_since_log.update(clipped)

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
            now = time.monotonic()
            if now - last_status >= 1.0:
                gripper_text = "" if gripper is None else f" gripper={gripper:.3f}"
                print(
                    f"seq={sequence} raw_q={np.array2string(raw_arm, precision=3)} "
                    f"q={np.array2string(arm, precision=3)} "
                    f"clipped={list(clipped)} "
                    f"clipped_since_log={sorted(clipped_since_log)}{gripper_text}"
                )
                last_status = now
                clipped_since_log.clear()

            next_time += period
            time.sleep(max(0.0, next_time - time.monotonic()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
