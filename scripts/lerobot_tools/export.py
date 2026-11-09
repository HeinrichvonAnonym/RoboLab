"""Official v3 writer, validation, provenance and atomic generation switch."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .core import CAMERAS, JOINT_NAMES, Config, Recording, atomic_json, discover, normalize_plan, plan_digest


def features_for(recording: Recording) -> dict:
    features = {
        name: {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES}
        for name in ("observation.state", "action", "observation.velocity", "action.velocity")
    }
    for camera, group in CAMERAS.items():
        meta = recording.camera_meta[group]
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": (int(meta["height"][0]), int(meta["width"][0]), 3),
            "names": ["height", "width", "channels"],
        }
    return features


def validate_dataset(root: Path, repo_id: str, expected_lengths: list[int], fps: int) -> dict:
    """Read every episode boundary with the official loader, including video."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text())
    total = sum(expected_lengths)
    if info["codebase_version"] != "v3.0" or info["total_frames"] != total or info["total_episodes"] != len(expected_lengths):
        raise ValueError("LeRobot metadata totals/version do not match export")
    # Small numeric tables are inspected in full; RGB is decoded at boundaries.
    tables = [pq.read_table(p) for p in sorted((root / "data").rglob("*.parquet"))]
    import pyarrow as pa
    table = pa.concat_tables(tables)
    if len(table) != total:
        raise ValueError("Parquet frame count mismatch")
    episode_indices = np.asarray(table["episode_index"])
    frame_indices = np.asarray(table["frame_index"])
    timestamps = np.asarray(table["timestamp"])
    if not np.array_equal(np.asarray(table["index"]), np.arange(total)):
        raise ValueError("Non-contiguous global frame indices")
    for key in ("action", "observation.state", "action.velocity", "observation.velocity"):
        values = np.asarray(table[key].to_pylist())
        if values.shape != (total, 8) or not np.isfinite(values).all():
            raise ValueError(f"Invalid exported numeric feature: {key}")
    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    if len(dataset) != total:
        raise ValueError("Official loader frame count mismatch")
    offset = 0
    for episode, count in enumerate(expected_lengths):
        span = slice(offset, offset + count)
        if not np.all(episode_indices[span] == episode) or not np.array_equal(frame_indices[span], np.arange(count)):
            raise ValueError("Episode/frame indices do not match retained clips")
        if not np.allclose(timestamps[span], np.arange(count) / fps, atol=1e-5):
            raise ValueError("Episode timestamps do not follow configured FPS")
        for index in sorted({offset, offset + count - 1}):
            item = dataset[index]
            for camera in CAMERAS:
                image = item[f"observation.images.{camera}"]
                if image.ndim != 3 or image.shape[0] != 3:
                    raise ValueError(f"Cannot decode exported {camera} image")
        offset += count
    stats = json.loads((root / "meta/stats.json").read_text())
    for key in ("action", "observation.state", "action.velocity", "observation.velocity"):
        if key not in stats or not np.isfinite(np.asarray(stats[key]["mean"])).all():
            raise ValueError(f"Missing/nonfinite normalization statistics: {key}")
        if int(stats[key]["count"][0]) != total:
            raise ValueError(f"Normalization count mismatch: {key}")
    return {"official_loader": True, "episodes": len(expected_lengths), "frames": total, "boundary_video_checks": True}


def export_dataset(config: Config, plan: dict, *, files: list[str] | None = None,
                   max_frames: int | None = None, progress=None) -> dict:
    """Build privately; publish by replacing one managed symlink on success.

    A failed generation never replaces a previous output. A real directory at
    the output path is never renamed/deleted. Older successful generations stay
    in the adjacent .<name>.versions directory for recovery.
    """
    config.validate()
    plan = normalize_plan(plan)
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    paths = discover(config)
    available = {p.name for p in paths}
    unknown_edits = set(plan["recordings"]) - available
    if unknown_edits:
        raise ValueError(f"Clip plan references unselected/missing sources: {sorted(unknown_edits)}")
    if files:
        missing = set(files) - available
        if missing:
            raise ValueError(f"Files not selected by input directory/since: {sorted(missing)}")
        paths = [p for p in paths if p.name in files]
    if not paths:
        raise ValueError("No completed H5 recordings match the selection")
    output = Path(config.output).expanduser().absolute()
    # Do not resolve the last component: it is our atomic current-generation link.
    output = output.parent.resolve() / output.name
    input_dir = Path(config.input_dir).resolve()
    if output == input_dir or output in input_dir.parents:
        raise ValueError("Output must not replace the input directory or an ancestor")
    output.parent.mkdir(parents=True, exist_ok=True)
    versions = output.parent / f".{output.name}.versions"
    if output.is_symlink():
        target = output.resolve()
        if target.parent != versions.resolve() or not (target / "export_report.json").is_file():
            raise ValueError("Refusing to replace an unmanaged output symlink")
    elif output.exists():
        raise ValueError("Output already exists as a directory/file; choose an unused output path")
    lock_path = output.parent / f".{output.name}.export.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(f"Another export owns {lock_path}; if it crashed, check the recorded PID before removing the lock") from None
    os.write(lock_fd, json.dumps({"pid": os.getpid(), "output": str(output)}).encode())
    os.close(lock_fd)
    generation = None
    dataset = None
    link = None
    report = {"version": 1, "config": asdict(config), "plan_digest": plan_digest(plan, config),
              "created_at": datetime.now(timezone.utc).isoformat(), "skipped": [], "recordings": [], "episodes": [],
              "source_selection": [p.name for p in paths], "max_frames_per_source": max_frames}

    def notify(**event):
        if progress:
            progress(event)

    try:
        # Validate each recording independently. Structural source failures skip
        # that recording; writer/codec/I/O failures fail the entire generation.
        selected = []
        for i, path in enumerate(paths):
            notify(phase="scan", recording=path.name, completed=i, total=len(paths))
            try:
                recording = Recording(path, config)
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                report["skipped"].append({"name": path.name, "reason": str(exc)})
                continue
            segments = recording.segments(plan)  # stale plan is an error, not a silent skip
            if max_frames is not None:
                remaining = max_frames
                limited = []
                for segment in segments:
                    part = segment[:remaining]
                    if len(part):
                        limited.append(part)
                        remaining -= len(part)
                    if not remaining:
                        break
                segments = limited
            report["recordings"].append(recording.summary(plan))
            if segments:
                selected.append((path, recording.fingerprint, [s.tolist() for s in segments]))
        if not selected:
            raise ValueError("No frames remain after source validation and clip exclusions; current dataset was preserved")
        from lerobot.configs import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import av
        import pyarrow as pa
        import pyarrow.parquet as pq

        av.logging.set_level(av.logging.WARNING)

        versions.mkdir(parents=True, exist_ok=True)
        generation = versions / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10])
        first = Recording(selected[0][0], config)
        features = features_for(first)
        # PyAV streaming avoids staging full-resolution PNGs for all episodes.
        dataset = LeRobotDataset.create(
            config.repo_id, fps=config.fps, features=features, root=generation,
            robot_type="franka", use_videos=True, video_backend="pyav",
            streaming_encoding=True, encoder_queue_maxsize=8, encoder_threads=2,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18, preset="fast"),
        )
        provenance_path = generation / "provenance.parquet"
        lengths = []
        total_frames = sum(len(segment) for _, _, segments in selected for segment in segments)
        written = 0
        with contextlib.ExitStack() as stack:
            provenance_writer = None
            for path, expected_fingerprint, segments in selected:
                recording = Recording(path, config)
                if recording.fingerprint != expected_fingerprint or features_for(recording) != features:
                    raise ValueError(f"Source changed or camera resolution differs: {path.name}")
                with h5py.File(path, "r") as h5:
                    for segment in segments:
                        episode_index = len(lengths)
                        rows = []
                        for frame_index, index in enumerate(segment):
                            frame = {key: values[index] for key, values in recording.aligned.items()}
                            for camera in CAMERAS:
                                frame[f"observation.images.{camera}"] = recording.read_rgb(h5, camera, index)
                            frame["task"] = config.task
                            dataset.add_frame(frame)
                            row = {"episode_index": episode_index, "frame_index": frame_index,
                                   "source_file": path.name, "source_grid_index": index,
                                   "source_timestamp_ns": int(recording.grid[index]),
                                   "source_seconds": float(recording.seconds[index]),
                                   "source_command_index": int(recording.command_indices[index])}
                            for camera, group in CAMERAS.items():
                                source_index = int(recording.camera_indices[camera][index])
                                row[f"{camera}_source_index"] = source_index
                                row[f"{camera}_timestamp_ns"] = int(recording.timestamps[group][source_index])
                            for group, lag in recording.lag.items():
                                row[f"{group}_age_ms"] = float(lag[index])
                            rows.append(row)
                            written += 1
                            if written % config.fps == 0 or written == total_frames:
                                notify(phase="write", recording=path.name, frames=written, total_frames=total_frames, episode=episode_index)
                        dataset.save_episode()
                        table = pa.Table.from_pylist(rows)
                        if provenance_writer is None:
                            provenance_writer = stack.enter_context(pq.ParquetWriter(provenance_path, table.schema))
                        provenance_writer.write_table(table)
                        lengths.append(len(segment))
                        report["episodes"].append({"episode_index": episode_index, "source_file": path.name,
                            "frames": len(segment), "source_start_seconds": float(recording.seconds[segment[0]]),
                            "source_end_seconds": float(recording.seconds[segment[-1]])})
                recording.check_source()
        dataset.finalize()
        dataset = None
        atomic_json(generation / "clip_plan.json", plan)
        notify(phase="validate", frames=written, total_frames=total_frames)
        report["validation"] = validate_dataset(generation, config.repo_id, lengths, config.fps)
        report["total_frames"] = written
        report["total_episodes"] = len(lengths)
        report["generation"] = str(generation)
        report["source_fingerprints"] = {path.name: fp for path, fp, _ in selected}
        atomic_json(generation / "export_report.json", report)
        link = output.parent / (f".{output.name}.link-{uuid.uuid4().hex}")
        link.symlink_to(os.path.relpath(generation, output.parent), target_is_directory=True)
        # One rename publishes data, videos, stats, provenance and plan together.
        os.replace(link, output)
        link = None
        notify(phase="complete", frames=written, total_frames=total_frames, output=str(output))
        return report
    except Exception as exc:
        notify(phase="failed", error=str(exc), skipped=report["skipped"])
        raise
    finally:
        if dataset is not None:
            with contextlib.suppress(Exception):
                dataset.finalize()
        if link is not None:
            link.unlink(missing_ok=True)
        # A finalized/published generation is retained; partial attempts are private.
        if generation is not None and not (generation / "export_report.json").exists():
            shutil.rmtree(generation, ignore_errors=True)
        lock_path.unlink(missing_ok=True)
