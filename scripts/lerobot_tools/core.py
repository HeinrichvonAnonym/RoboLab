"""Read-only H5 ingestion, synchronization and reproducible edit decisions.

All timestamps used for arithmetic are integer nanoseconds. A recording is
aligned once, before edits; edits only mask that canonical grid. In particular,
neither finite differences nor interpolation ever joins the two sides of a cut.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np

TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_SINCE = "2026-10-13T21:00:00+08:00"
GROUPS = ("franka_command", "franka_state", "kinec2_rgb", "realsense_rgb")
CAMERAS = {"kinec2": "kinec2_rgb", "realsense": "realsense_rgb"}
JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)] + ["gripper_close"]
PATTERN = re.compile(r"recording_(\d{8}_\d{6})_(\d+)\.h5$")


@dataclass(frozen=True)
class Config:
    input_dir: str = "data"
    output: str = "outputs/lerobot/franka_20261013_210000"
    since: str = DEFAULT_SINCE
    fps: int = 15
    task: str = "franka manipulation"
    max_gap_ms: float = 200.0
    repo_id: str = "local/franka_manipulation"

    def validate(self):
        if not isinstance(self.fps, int) or isinstance(self.fps, bool) or not 1 <= self.fps <= 240:
            raise ValueError("fps must be an integer between 1 and 240")
        if not math.isfinite(self.max_gap_ms) or self.max_gap_ms <= 0:
            raise ValueError("max_gap_ms must be positive and finite")
        if not self.task.strip():
            raise ValueError("task must not be empty")
        if len(self.repo_id.split("/")) != 2 or not all(self.repo_id.split("/")):
            raise ValueError("repo_id must have the form owner/dataset")
        parse_since(self.since)


def parse_since(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=TZ) if dt.tzinfo is None else dt.astimezone(TZ)


def discover(config: Config) -> list[Path]:
    config.validate()
    root = Path(config.input_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    since = parse_since(config.since)
    return sorted(
        p for p in root.glob("*.h5")
        if (m := PATTERN.fullmatch(p.name))
        and datetime.strptime(m[1], "%Y%m%d_%H%M%S").replace(tzinfo=TZ) >= since
    )


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    # JS Number cannot exactly represent epoch nanoseconds (> 2**53).
    return {"size": stat.st_size, "mtime_ns": str(stat.st_mtime_ns)}


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def empty_plan() -> dict:
    return {"version": 1, "recordings": {}}


def normalize_plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise ValueError("Clip plan must be an object with version: 1")
    recordings = plan.get("recordings", {})
    if not isinstance(recordings, dict):
        raise ValueError("recordings must be an object")
    result = empty_plan()
    for name, edit in recordings.items():
        if not isinstance(name, str) or Path(name).name != name or not PATTERN.fullmatch(name):
            raise ValueError(f"Invalid source filename: {name}")
        if not isinstance(edit, dict) or not isinstance(edit.get("drop", False), bool):
            raise ValueError(f"Invalid edit for {name}")
        intervals = edit.get("exclude", [])
        if not isinstance(intervals, list):
            raise ValueError("exclude must be a list of [start_seconds, end_seconds]")
        checked = []
        for pair in intervals:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Each exclusion needs start and end")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in pair):
                raise ValueError("Clip bounds must be numeric")
            a, b = map(float, pair)
            if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b < a:
                raise ValueError("Clip bounds must satisfy 0 <= start <= end and be finite")
            checked.append([a, b])
        merged = []
        for a, b in sorted(checked):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(b, merged[-1][1])
            else:
                merged.append([a, b])
        item = {"drop": edit.get("drop", False), "exclude": merged}
        if "fingerprint" in edit:
            fp = edit["fingerprint"]
            if (not isinstance(fp, dict) or set(fp) != {"size", "mtime_ns"}
                or not isinstance(fp["size"], int) or isinstance(fp["size"], bool) or fp["size"] < 0
                or isinstance(fp["mtime_ns"], bool) or not isinstance(fp["mtime_ns"], (str, int))
                or not str(fp["mtime_ns"]).isascii() or not str(fp["mtime_ns"]).isdigit()):
                raise ValueError("Invalid source fingerprint")
            item["fingerprint"] = {"size": fp["size"], "mtime_ns": str(fp["mtime_ns"])}
        result["recordings"][name] = item
    return result


def load_plan(path: Path | None) -> dict:
    return empty_plan() if path is None or not path.exists() else normalize_plan(json.loads(path.read_text()))


def plan_digest(plan: dict, config: Config | None = None) -> str:
    obj = {"plan": normalize_plan(plan)}
    if config is not None:
        obj["config"] = asdict(config)
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def differentiate(values: np.ndarray, timestamps: np.ndarray, max_gap_ns: int) -> np.ndarray:
    """Nonuniform differences within acquisition-continuous runs, before cuts."""
    result = np.zeros_like(values, dtype=np.float64)
    runs = np.split(np.arange(len(timestamps)), np.flatnonzero(np.diff(timestamps) > max_gap_ns) + 1)
    previous = np.zeros(values.shape[1:], dtype=np.float64)
    for run in runs:
        if len(run) >= 2:
            seconds = (timestamps[run] - timestamps[run[0]]).astype(np.float64) / 1e9
            result[run] = np.gradient(values[run], seconds, axis=0, edge_order=1)
            previous = result[run[-1]]
        elif len(run):
            result[run] = previous
    return result


def bracket(timestamps: np.ndarray, target: np.ndarray):
    left = np.clip(np.searchsorted(timestamps, target, side="right") - 1, 0, len(timestamps) - 1)
    right = np.minimum(left + 1, len(timestamps) - 1)
    return left, right


def interpolate(values: np.ndarray, timestamps: np.ndarray, target: np.ndarray, max_gap_ns: int):
    left, right = bracket(timestamps, target)
    gap = timestamps[right] - timestamps[left]
    weight = np.zeros(len(target), dtype=np.float64)
    use = (gap > 0) & (gap <= max_gap_ns)
    weight[use] = (target[use] - timestamps[left[use]]) / gap[use]
    return values[left] + weight[:, None] * (values[right] - values[left])


class Recording:
    """Small numeric arrays and image offsets only; RGB blobs stay on disk."""

    def __init__(self, path: Path, config: Config):
        config.validate()
        self.path = Path(path)
        self.config = config
        self.fingerprint = fingerprint(self.path)
        self.timestamps = {}
        self.camera_meta = {}
        self.positions = {}
        self.max_gap_ns = round(config.max_gap_ms * 1e6)
        match = PATTERN.fullmatch(self.path.name)
        if not match:
            raise ValueError("Unrecognized recording filename")
        file_start = round(datetime.strptime(match[1], "%Y%m%d_%H%M%S").replace(tzinfo=TZ).timestamp() * 1e9)
        with h5py.File(self.path, "r") as f:
            for name in GROUPS:
                if name not in f or "timestamps_ns" not in f[name]:
                    raise ValueError(f"Missing stream: {name}")
                group = f[name]
                ts = np.asarray(group["timestamps_ns"])
                if ts.ndim != 1 or ts.dtype.kind not in "iu" or not len(ts):
                    raise ValueError(f"Empty or malformed timestamps: {name}")
                if np.any(ts < file_start - 1_000_000_000) or np.any(ts > file_start + 86_400_000_000_000):
                    raise ValueError(f"Damaged/out-of-range timestamps: {name}")
                ts = ts.astype(np.int64)
                if np.any(np.diff(ts) <= 0):
                    raise ValueError(f"Non-increasing timestamps: {name}")
                self.timestamps[name] = ts
                if name.startswith("franka_"):
                    for field in ("joints_position", "joints_velocity"):
                        if field not in group:
                            raise ValueError(f"Missing {name}/{field}")
                        arr = np.asarray(group[field], dtype=np.float64)
                        if arr.shape != (len(ts), 8) or not np.isfinite(arr).all():
                            raise ValueError(f"Invalid {name}/{field}; expected finite (N,8)")
                        self.positions[name + "/" + field] = arr
                else:
                    meta = {}
                    for field in ("width", "height", "channels", "format_type", "image_offsets", "image_lengths"):
                        if field not in group or group[field].shape != ts.shape:
                            raise ValueError(f"Missing/mismatched {name}/{field}")
                        meta[field] = np.asarray(group[field])
                    expected_channels, expected_format = (4, 1) if name == "kinec2_rgb" else (3, 5)
                    if any(np.any(meta[x] != meta[x][0]) for x in ("width", "height")):
                        raise ValueError(f"Changing RGB resolution: {name}")
                    if np.any(meta["channels"] != expected_channels) or np.any(meta["format_type"] != expected_format):
                        raise ValueError(f"Unsupported RGB encoding: {name}")
                    if np.any(meta["width"] <= 0) or np.any(meta["height"] <= 0):
                        raise ValueError(f"Invalid RGB resolution: {name}")
                    expected_bytes = meta["width"].astype(np.int64) * meta["height"] * expected_channels
                    if np.any(meta["image_lengths"] != expected_bytes):
                        raise ValueError(f"Invalid image byte lengths: {name}")
                    if "image_data" not in group or group["image_data"].ndim != 1 or group["image_data"].dtype.itemsize != 1:
                        raise ValueError(f"Missing/malformed image blob: {name}")
                    offsets = meta["image_offsets"].astype(np.int64)
                    if np.any(offsets < 0) or np.any(offsets > group["image_data"].size - expected_bytes):
                        raise ValueError(f"Image offset outside blob: {name}")
                    self.camera_meta[name] = meta
        self.origin_ns = min(int(ts[0]) for ts in self.timestamps.values())
        self.start_ns = max(int(ts[0]) for ts in self.timestamps.values())
        self.end_ns = min(int(ts[-1]) for ts in self.timestamps.values())
        if self.end_ns < self.start_ns:
            raise ValueError("Streams have no common time coverage")
        # Rounded integer grid avoids floating epoch-time precision loss.
        count = (self.end_ns - self.start_ns) * config.fps // 1_000_000_000 + 1
        self.grid = self.start_ns + np.rint(np.arange(count, dtype=np.float64) * (1e9 / config.fps)).astype(np.int64)
        self.seconds = (self.grid - self.origin_ns).astype(np.float64) / 1e9
        self.actions = self.positions["franka_command/joints_position"]
        raw_action_velocity = differentiate(self.actions, self.timestamps["franka_command"], self.max_gap_ns)
        raw_state_velocity = self.positions["franka_state/joints_velocity"].copy()
        raw_state_velocity[:, 7:] = differentiate(
            self.positions["franka_state/joints_position"][:, 7:], self.timestamps["franka_state"], self.max_gap_ns
        )
        ci, _ = bracket(self.timestamps["franka_command"], self.grid)
        self.command_indices = ci
        self.aligned = {
            "action": self.actions[ci].astype(np.float32),
            "action.velocity": raw_action_velocity[ci].astype(np.float32),
            "observation.state": interpolate(self.positions["franka_state/joints_position"], self.timestamps["franka_state"], self.grid, self.max_gap_ns).astype(np.float32),
            "observation.velocity": interpolate(raw_state_velocity, self.timestamps["franka_state"], self.grid, self.max_gap_ns).astype(np.float32),
        }
        self.camera_indices = {}
        for camera, name in CAMERAS.items():
            ts = self.timestamps[name]
            left, right = bracket(ts, self.grid)
            near_right = self.grid - ts[left] > ts[right] - self.grid
            continuous = ts[right] - ts[left] <= self.max_gap_ns
            self.camera_indices[camera] = np.where(near_right & continuous, right, left)
        self.lag = {}
        self.gaps = {}
        for name, ts in self.timestamps.items():
            left, _ = bracket(ts, self.grid)
            self.lag[name] = (self.grid - ts[left]).astype(np.float64) / 1e6
            indices = np.flatnonzero(np.diff(ts) > self.max_gap_ns)
            self.gaps[name] = [
                {"start": (int(ts[i]) - self.origin_ns) / 1e9,
                 "end": (int(ts[i + 1]) - self.origin_ns) / 1e9,
                 "gap_ms": (int(ts[i + 1]) - int(ts[i])) / 1e6}
                for i in indices
            ]

    def check_source(self):
        if fingerprint(self.path) != self.fingerprint:
            raise ValueError(f"Source changed since scan: {self.path.name}")

    def segments(self, plan: dict) -> list[np.ndarray]:
        edit = plan.get("recordings", {}).get(self.path.name, {})
        if "fingerprint" in edit and edit["fingerprint"] != self.fingerprint:
            raise ValueError(f"Clip plan belongs to a different source version: {self.path.name}")
        if edit.get("drop", False):
            return []
        keep = np.ones(len(self.grid), dtype=bool)
        for a, b in edit.get("exclude", []):
            # Inclusive frame bounds, rounded identically for UI and headless.
            lo = self.origin_ns + round(a * 1e9)
            hi = self.origin_ns + round(b * 1e9)
            keep &= ~((self.grid >= lo) & (self.grid <= hi))
        indices = np.flatnonzero(keep)
        return [] if not len(indices) else list(np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1))

    def read_rgb(self, h5: h5py.File, camera: str, index: int) -> np.ndarray:
        name = CAMERAS[camera]
        meta = self.camera_meta[name]
        i = int(self.camera_indices[camera][index])
        start = int(meta["image_offsets"][i])
        count = int(meta["image_lengths"][i])
        payload = np.asarray(h5[name]["image_data"][start:start + count]).view(np.uint8)
        image = payload.reshape(int(meta["height"][i]), int(meta["width"][i]), int(meta["channels"][i]))
        return np.ascontiguousarray(image[:, :, 2::-1] if camera == "kinec2" else image)

    def summary(self, plan: dict | None = None) -> dict:
        segments = self.segments(plan or empty_plan())
        return {
            "name": self.path.name, "valid": True, "fingerprint": self.fingerprint,
            "duration": (self.end_ns - self.origin_ns) / 1e9,
            "start": float(self.seconds[0]), "end": float(self.seconds[-1]),
            "frames": len(self.grid), "kept_frames": sum(len(s) for s in segments),
            "episodes": len(segments), "gaps": self.gaps,
            "stale_frames": {name: int(np.count_nonzero(lag > self.config.max_gap_ms)) for name, lag in self.lag.items()},
        }


def scan(config: Config, plan: dict | None = None) -> list[dict]:
    entries = []
    for path in discover(config):
        try:
            entries.append(Recording(path, config).summary(plan))
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            entries.append({"name": path.name, "valid": False, "error": str(exc)})
    return entries
