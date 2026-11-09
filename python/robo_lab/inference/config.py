from pathlib import Path
import json
import math
import os

import numpy as np
import yaml


def load_config(path):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load(path.read_text())
    cfg["root"] = root
    for key in ("robot_config", "kinematics_config"):
        p = Path(cfg[key])
        cfg[key] = yaml.safe_load((p if p.is_absolute() else root / p).read_text())
    for key in ("frequency", "max_age_ms", "stale_fault_s", "home_tolerance_rad",
                "position_tolerance_m", "orientation_tolerance_rad", "max_joint_speed",
                "inference_timeout_s"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if type(cfg["chunk_size"]) is not int or not 1 <= cfg["chunk_size"] <= 1000:
        raise ValueError("chunk_size must be an integer in [1,1000]")
    circle = cfg["circle"]
    for key in ("radius_m", "duration_s"):
        if not math.isfinite(circle[key]) or circle[key] <= 0:
            raise ValueError(f"circle.{key} must be finite and positive")
    if type(circle["turns"]) is not int or circle["turns"] < 1:
        raise ValueError("circle.turns must be a positive integer")
    if circle["plane"] not in ("xy", "yz", "zx"):
        raise ValueError("circle.plane must be xy, yz or zx")
    home = np.asarray(cfg["robot_config"]["arm_home"], dtype=float)
    if home.shape != (7,) or not np.isfinite(home).all():
        raise ValueError("arm_home must contain seven finite angles")
    if not isinstance(cfg["prompt"], str) or not cfg["prompt"].strip():
        raise ValueError("prompt must be nonempty text")
    if cfg["policy"] == "lingbot_vla_v2":
        _validate_lingbot(cfg, root)
    elif cfg["policy"] != "fake_circle":
        if not cfg.get("checkpoint_path"):
            raise ValueError("Real policy requires checkpoint_path; currently null")
        raise NotImplementedError("Checkpoint backend is not implemented in v1")
    topics = list(cfg["topics"].values())
    if len(topics) != len(set(topics)) or not all(isinstance(t, str) and t for t in topics):
        raise ValueError("topics must be distinct nonempty strings")
    return cfg


def _repo_path(root, value, key):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require_complete_checkpoint(path):
    if not path.is_dir():
        raise ValueError(f"checkpoint_path is not a directory: {path}")
    incomplete = sorted(item.name for item in path.iterdir() if ".downloading" in item.name)
    if incomplete:
        raise ValueError(f"checkpoint download is incomplete: {incomplete}")
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        raise ValueError(f"missing {index.name} in {path}")
    weight_map = json.loads(index.read_text()).get("weight_map") or {}
    shards = sorted(set(weight_map.values()))
    missing = [name for name in shards if not (path / name).is_file()]
    if not shards or missing:
        raise ValueError(f"checkpoint shards missing: {missing or 'weight_map'}")


def _validate_lingbot(cfg, root):
    for key in ("lingbot_cli", "lingbot_robot_config", "norm_stats", "lingbot_root"):
        if key not in cfg:
            raise ValueError(f"lingbot_vla_v2 requires {key}")
    checkpoint = _repo_path(root, cfg.get("checkpoint_path") or "", "checkpoint_path")
    cli = _repo_path(root, cfg["lingbot_cli"], "lingbot_cli")
    robot = _repo_path(root, cfg["lingbot_robot_config"], "lingbot_robot_config")
    stats = _repo_path(root, cfg["norm_stats"], "norm_stats")
    lingbot_root = _repo_path(root, cfg["lingbot_root"], "lingbot_root")
    _require_complete_checkpoint(checkpoint)
    for label, path in (("lingbot_cli", cli), ("lingbot_robot_config", robot), ("norm_stats", stats)):
        if not path.is_file():
            raise ValueError(f"{label} is not a file: {path}")
    deploy = lingbot_root / "deploy" / "lingbot_vla_v2_policy.py"
    if not deploy.is_file():
        raise ValueError(f"lingbot_root does not contain deploy/lingbot_vla_v2_policy.py: {lingbot_root}")
    qwen = os.environ.get("QWEN3VL_PATH", "")
    if not qwen or not Path(qwen).is_dir():
        raise ValueError("QWEN3VL_PATH must be the local Qwen3-VL-4B-Instruct directory")
    timeout = cfg.get("policy_init_timeout_s")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("policy_init_timeout_s must be finite and positive")
    robot_yaml = yaml.safe_load(robot.read_text())
    if not isinstance(robot_yaml, dict) or "norm_stats" not in robot_yaml:
        raise ValueError("lingbot robot yaml is missing norm_stats")
    payload = json.loads(stats.read_text())
    needed = {
        "action.arm.position",
        "action.effector.position",
        "observation.state.arm.position",
        "observation.state.effector.position",
    }
    if set(payload.get("norm_stats", {})) != needed:
        raise ValueError("norm_stats is missing Franka arm/effector keys")
    cfg["checkpoint_path"] = str(checkpoint)
    cfg["lingbot_cli"] = str(cli)
    cfg["lingbot_robot_config"] = str(robot)
    cfg["norm_stats"] = str(stats)
    cfg["lingbot_root"] = str(lingbot_root)
    cfg["qwen3vl_path"] = str(Path(qwen).resolve())
