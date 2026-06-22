#!/usr/bin/env python3
"""Visualize actual and commanded Franka joint states in MuJoCo.

The viewer creates two Franka Panda Gripper instances at the same root pose:
one uses ``state_topic`` and keeps the original visual materials, the other
uses ``cmd_topic`` and is rendered blue and transparent. The script also
subscribes to ``state_machine_topic`` and shows the latest state in the viewer
overlay when the installed MuJoCo viewer exposes overlay support.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Iterable


ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
BLUE_RGBA = "0.1 0.35 1.0 0.35"


@dataclass
class TopicConfig:
    cmd_topic: str = "franka/command"
    state_topic: str = "franka/state"
    state_machine_topic: str = "franka/state_machine"
    arm_home: list[float] = field(
        default_factory=lambda: [-0.32, -0.9, 0.13, -2.75, 0.18, 1.95, 0.49]
    )
    gripper_max_width: float = 0.08
    gripper_closed_width: float = 0.0


@dataclass
class LatestRobotData:
    q: list[float]
    gripper_width: float
    stamp: float = 0.0
    sequence: int = 0
    label: str = "waiting"


@dataclass
class LatestState:
    state: LatestRobotData
    command: LatestRobotData
    state_machine: str = "unknown"
    state_machine_stamp: float = 0.0


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
        print(
            "protoc not found. Install protobuf-compiler or run scripts/regen_proto_py.sh",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _parse_scalar(value: str):
    text = value.strip()
    if not text:
        return ""
    try:
        return ast.literal_eval(text)
    except Exception:
        return text.strip("\"'")


def load_topic_config(path: Path) -> TopicConfig:
    cfg = TopicConfig()
    if not path.is_file():
        raise FileNotFoundError(path)

    values: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_inline_comment(raw_line).strip()
        if not line or ":" not in line or line.startswith("-"):
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {
            "cmd_topic",
            "state_topic",
            "state_machine_topic",
            "arm_home",
            "gripper_max_width",
            "gripper_closed_width",
        }:
            values[key] = _parse_scalar(value)

    cfg.cmd_topic = str(values.get("cmd_topic", cfg.cmd_topic))
    cfg.state_topic = str(values.get("state_topic", cfg.state_topic))
    cfg.state_machine_topic = str(values.get("state_machine_topic", cfg.state_machine_topic))
    if isinstance(values.get("arm_home"), list) and len(values["arm_home"]) >= 7:
        cfg.arm_home = [float(v) for v in values["arm_home"][:7]]
    if "gripper_max_width" in values:
        cfg.gripper_max_width = float(values["gripper_max_width"])
    if "gripper_closed_width" in values:
        cfg.gripper_closed_width = float(values["gripper_closed_width"])
    return cfg


def _face_vertex_index(token: str) -> str:
    return token.split("/", 1)[0]


def _safe_mesh_part_name(value: str) -> str:
    value = value.strip() or "mesh"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


def _resolve_obj_index(token: str, vertex_count: int) -> int:
    raw = int(_face_vertex_index(token))
    if raw < 0:
        return vertex_count + raw + 1
    return raw


def _transform_visual_obj_axes_split(src: Path) -> list[Path]:
    cache_key = f"axis_v3_split_objects:{src}:{src.stat().st_mtime_ns}"
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
    dst_dir = Path(tempfile.gettempdir()) / "robolab_mujoco_franka_meshes"
    dst_dir.mkdir(parents=True, exist_ok=True)

    vertices: list[tuple[float, float, float]] = []
    objects: list[tuple[str, list[list[int]]]] = []
    current_name = "mesh"
    current_faces: list[list[int]] = []

    def finish_current() -> None:
        nonlocal current_faces
        if current_faces:
            objects.append((current_name, current_faces))
            current_faces = []

    with src.open("r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("o "):
                finish_current()
                current_name = line[2:].strip() or "mesh"
            elif line.startswith("f "):
                indices = [_resolve_obj_index(token, len(vertices)) for token in line.split()[1:]]
                if len(indices) >= 3:
                    # Fan-triangulate defensively. Franka meshes are already
                    # triangles, but plain f-lines avoid OBJ parser edge cases.
                    for i in range(1, len(indices) - 1):
                        current_faces.append([indices[0], indices[i], indices[i + 1]])
    finish_current()

    if not objects:
        return [src]

    paths: list[Path] = []
    for object_index, (object_name, faces) in enumerate(objects):
        safe_name = _safe_mesh_part_name(object_name)
        dst = dst_dir / f"{src.stem}_{object_index:02d}_{safe_name}_{digest}.obj"
        paths.append(dst)
        if dst.is_file():
            continue

        used_indices: dict[int, int] = {}
        ordered_indices: list[int] = []
        for face in faces:
            for index in face:
                if index not in used_indices:
                    used_indices[index] = len(ordered_indices) + 1
                    ordered_indices.append(index)

        with dst.open("w", encoding="utf-8") as fout:
            fout.write(f"# Generated from {src} object {object_name} for MuJoCo visual loading.\n")
            for old_index in ordered_indices:
                x, y, z = vertices[old_index - 1]
                fout.write(f"v {x:.9g} {-z:.9g} {y:.9g}\n")
            for face in faces:
                a, b, c = (used_indices[face[0]], used_indices[face[1]], used_indices[face[2]])
                fout.write(f"f {a} {b} {c}\n")
    return paths


def _rewrite_mesh_filenames(filename: str, description_root: Path) -> list[str]:
    prefix = "package://franka_description/"
    if filename.startswith(prefix):
        path = (description_root / filename[len(prefix) :]).resolve()
        parts = path.parts
        if path.suffix.lower() == ".dae":
            obj_path = path.with_suffix(".obj")
            if obj_path.is_file():
                path = obj_path
        if "visual" in parts and path.suffix.lower() == ".obj":
            return [str(path) for path in _transform_visual_obj_axes_split(path)]
        return [str(path)]
    return [filename]


def _child_link_names(root: ET.Element) -> set[str]:
    names: set[str] = set()
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link"):
            names.add(child.get("link", ""))
    return names


def _root_link_name(root: ET.Element) -> str:
    links = [link.get("name", "") for link in root.findall("link")]
    children = _child_link_names(root)
    for name in links:
        if name and name not in children:
            return name
    raise ValueError("Could not find URDF root link")


def _ensure_inertial(link: ET.Element) -> None:
    if link.find("inertial") is not None:
        return
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.01"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "0.0001",
            "ixy": "0",
            "ixz": "0",
            "iyy": "0.0001",
            "iyz": "0",
            "izz": "0.0001",
        },
    )


def _expand_visual_meshes(link: ET.Element, description_root: Path) -> None:
    for visual in list(link.findall("visual")):
        mesh = visual.find("./geometry/mesh")
        if mesh is None or not mesh.get("filename"):
            continue

        mesh_files = _rewrite_mesh_filenames(mesh.get("filename", ""), description_root)
        if len(mesh_files) == 1:
            mesh.set("filename", mesh_files[0])
            continue

        insert_at = list(link).index(visual)
        link.remove(visual)
        for mesh_file in mesh_files:
            visual_copy = copy.deepcopy(visual)
            mesh_copy = visual_copy.find("./geometry/mesh")
            if mesh_copy is not None:
                mesh_copy.set("filename", mesh_file)
            link.insert(insert_at, visual_copy)
            insert_at += 1


def _prefix_robot_copy(
    source: ET.Element,
    *,
    prefix: str,
    description_root: Path,
    blue: bool,
) -> list[ET.Element]:
    material_map: dict[str, str] = {}
    elements = [copy.deepcopy(child) for child in list(source)]
    blue_material = f"{prefix}blue_transparent"

    if blue:
        material = ET.Element("material", {"name": blue_material})
        ET.SubElement(material, "color", {"rgba": BLUE_RGBA})
        elements.insert(0, material)

    for elem in elements:
        for node in elem.iter():
            tag = node.tag
            if tag in {"link", "joint", "material"} and node.get("name"):
                old_name = node.get("name", "")
                new_name = prefix + old_name
                if tag == "material":
                    material_map[old_name] = new_name
                node.set("name", new_name)

            if tag in {"parent", "child"} and node.get("link"):
                node.set("link", prefix + node.get("link", ""))
            if tag == "mimic" and node.get("joint"):
                node.set("joint", prefix + node.get("joint", ""))

        if elem.tag == "link":
            _ensure_inertial(elem)
            if elem.find("visual") is None:
                for collision in elem.findall("collision"):
                    visual = copy.deepcopy(collision)
                    visual.tag = "visual"
                    elem.append(visual)
            _expand_visual_meshes(elem, description_root)
            for collision in list(elem.findall("collision")):
                elem.remove(collision)
            if blue:
                for visual in elem.findall("visual"):
                    material = visual.find("material")
                    if material is None:
                        material = ET.SubElement(visual, "material")
                    material.set("name", blue_material)
                    color = material.find("color")
                    if color is None:
                        color = ET.SubElement(material, "color")
                    color.set("rgba", BLUE_RGBA)

    if not blue:
        for elem in elements:
            for material_ref in elem.iter("material"):
                name = material_ref.get("name")
                if name in material_map:
                    material_ref.set("name", material_map[name])

    return elements


def build_dual_franka_urdf(urdf_path: Path, root_xyz: Iterable[float], root_rpy: Iterable[float]) -> str:
    source = ET.parse(urdf_path).getroot()
    description_root = urdf_path.parents[1]
    original_root_link = _root_link_name(source)

    robot = ET.Element("robot", {"name": "dual_franka_overlay"})
    mujoco_ext = ET.SubElement(robot, "mujoco")
    ET.SubElement(mujoco_ext, "compiler", {"discardvisual": "false"})
    ET.SubElement(robot, "link", {"name": "world"})

    xyz = " ".join(str(float(v)) for v in root_xyz)
    rpy = " ".join(str(float(v)) for v in root_rpy)
    for prefix, blue in (("state_", False), ("cmd_", True)):
        for elem in _prefix_robot_copy(
            source,
            prefix=prefix,
            description_root=description_root,
            blue=blue,
        ):
            robot.append(elem)
        joint = ET.SubElement(robot, "joint", {"name": f"{prefix}root_joint", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": "world"})
        ET.SubElement(joint, "child", {"link": prefix + original_root_link})
        ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})

    return ET.tostring(robot, encoding="unicode")


def _payload_bytes(sample) -> bytes:
    payload = sample.payload
    if hasattr(payload, "to_bytes"):
        return payload.to_bytes()
    return bytes(payload)


def _gripper_width_from_joint(value: float, cfg: TopicConfig) -> float:
    value = min(1.0, max(0.0, float(value)))
    return cfg.gripper_max_width + value * (cfg.gripper_closed_width - cfg.gripper_max_width)


def _decode_robot_message(msg, cfg: TopicConfig) -> tuple[list[float], float]:
    q = [float(msg.joints[i].position) for i in range(min(7, len(msg.joints)))]
    if len(q) < 7:
        q.extend(cfg.arm_home[len(q) : 7])
    width = cfg.gripper_max_width
    if len(msg.joints) >= 8:
        width = _gripper_width_from_joint(float(msg.joints[7].position), cfg)
    return q, width


def _set_joint_qpos(model, data, joint_name: str, value: float) -> None:
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return
    qpos_addr = int(model.jnt_qposadr[jid])
    data.qpos[qpos_addr] = value


def _apply_robot_pose(model, data, prefix: str, robot_data: LatestRobotData) -> None:
    for joint_name, value in zip(ARM_JOINTS, robot_data.q):
        _set_joint_qpos(model, data, prefix + joint_name, value)
    finger_opening = max(0.0, min(robot_data.gripper_width / 2.0, 0.04))
    for joint_name in FINGER_JOINTS:
        _set_joint_qpos(model, data, prefix + joint_name, finger_opening)


def _freshness(stamp: float) -> str:
    if stamp <= 0.0:
        return "never"
    age = time.time() - stamp
    return f"{age:.2f}s ago"


def _overlay_text(cfg: TopicConfig, latest: LatestState) -> tuple[str, str]:
    left = (
        f"state_machine: {latest.state_machine}\n"
        f"state topic: {cfg.state_topic}\n"
        f"cmd topic: {cfg.cmd_topic}\n"
        f"state_machine topic: {cfg.state_machine_topic}"
    )
    right = (
        f"white/state: {latest.state.label} ({_freshness(latest.state.stamp)})\n"
        f"blue/cmd: {latest.command.label} ({_freshness(latest.command.stamp)})\n"
        f"state_machine: {_freshness(latest.state_machine_stamp)}"
    )
    return left, right


def _maybe_add_overlay(viewer, cfg: TopicConfig, latest: LatestState) -> bool:
    if not hasattr(viewer, "add_overlay"):
        return False
    try:
        import mujoco

        left, right = _overlay_text(cfg, latest)
        viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, "Franka State Viewer", left)
        viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Messages", right)
        return True
    except Exception:
        return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default=str(repo_root / "config" / "plugins" / "franka_plugin_config.yaml"),
        help="Franka plugin config path.",
    )
    parser.add_argument(
        "--urdf",
        default=str(
            repo_root
            / "ros_ws"
            / "src"
            / "franka_description"
            / "robots"
            / "franka_panda_gripper.urdf"
        ),
        help="Franka Panda gripper URDF path.",
    )
    parser.add_argument(
        "--connect",
        "-e",
        action="append",
        default=None,
        metavar="ENDPOINT",
        help="Zenoh endpoint (repeatable), e.g. tcp/127.0.0.1:7447.",
    )
    parser.add_argument("--hz", type=float, default=60.0, help="Viewer update rate.")
    parser.add_argument("--root-xyz", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--root-rpy", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open MuJoCo viewer; print subscription status instead.",
    )
    args = parser.parse_args()

    cfg = load_topic_config(Path(args.config))
    latest = LatestState(
        state=LatestRobotData(q=list(cfg.arm_home), gripper_width=cfg.gripper_max_width),
        command=LatestRobotData(q=list(cfg.arm_home), gripper_width=cfg.gripper_max_width),
    )
    latest_lock = Lock()

    _ensure_franka_pb2(repo_root)
    sys.path.insert(0, str(repo_root / "scripts" / "gen"))
    try:
        import franka_pb2
        import mujoco
        import zenoh
    except ImportError as exc:
        print("Missing dependency: pip install eclipse-zenoh protobuf mujoco", file=sys.stderr)
        raise SystemExit(1) from exc

    xml = build_dual_franka_urdf(Path(args.urdf), args.root_xyz, args.root_rpy)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    def on_state(sample) -> None:
        msg = franka_pb2.RobotObservation()
        msg.ParseFromString(_payload_bytes(sample))
        q, width = _decode_robot_message(msg, cfg)
        with latest_lock:
            latest.state = LatestRobotData(
                q=q,
                gripper_width=width,
                stamp=time.time(),
                sequence=int(msg.sequence),
                label=f"seq={int(msg.sequence)}",
            )

    def on_command(sample) -> None:
        msg = franka_pb2.RobotCommand()
        msg.ParseFromString(_payload_bytes(sample))
        q, width = _decode_robot_message(msg, cfg)
        with latest_lock:
            latest.command = LatestRobotData(
                q=q,
                gripper_width=width,
                stamp=time.time(),
                sequence=int(msg.sequence),
                label=f"seq={int(msg.sequence)}",
            )

    def on_state_machine(sample) -> None:
        text = _payload_bytes(sample).decode("utf-8", errors="replace").strip()
        with latest_lock:
            latest.state_machine = text or "unknown"
            latest.state_machine_stamp = time.time()

    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", json.dumps(list(args.connect)))

    period = 1.0 / max(1.0, float(args.hz))
    with zenoh.open(conf) as session:
        sub_state = session.declare_subscriber(cfg.state_topic, on_state)
        sub_cmd = session.declare_subscriber(cfg.cmd_topic, on_command)
        sub_sm = session.declare_subscriber(cfg.state_machine_topic, on_state_machine)
        _ = (sub_state, sub_cmd, sub_sm)
        print(
            "Subscribed:\n"
            f"  state -> white/original: {cfg.state_topic}\n"
            f"  command -> blue/transparent: {cfg.cmd_topic}\n"
            f"  state_machine: {cfg.state_machine_topic}"
        )

        if args.no_gui:
            while True:
                with latest_lock:
                    left, right = _overlay_text(cfg, copy.deepcopy(latest))
                print(f"{left}\n{right}\n")
                time.sleep(1.0)

        import mujoco.viewer

        last_print = 0.0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                with latest_lock:
                    snapshot = copy.deepcopy(latest)
                _apply_robot_pose(model, data, "state_", snapshot.state)
                _apply_robot_pose(model, data, "cmd_", snapshot.command)
                mujoco.mj_forward(model, data)

                overlay_ok = _maybe_add_overlay(viewer, cfg, snapshot)
                if not overlay_ok and time.time() - last_print > 1.0:
                    left, right = _overlay_text(cfg, snapshot)
                    print(f"{left}\n{right}\n")
                    last_print = time.time()
                viewer.sync()
                time.sleep(period)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
