"""Read-only joint diagnostics, including unfiltered chunks beyond stale HDF5 EOA.

Run using the roboLab Python environment. Never writes the original recording.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import xml.etree.ElementTree as ET

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/recording_20261010_211159_1791637919729.h5"
OUT = ROOT / "analysis/recording_20261010_211159"
TZ = timezone(timedelta(hours=8))


def clock(ns):
    return datetime.fromtimestamp(int(ns) / 1e9, TZ).isoformat(timespec="milliseconds")


def read_chunks(ds, raw_file, file_size):
    """Validate raw chunks against h5py where possible; recover only plain data."""
    if ds.chunks is None or ds.id.get_create_plist().get_nfilters():
        raise ValueError(f"Unsupported storage for recovery: {ds.name}")
    if ds.ndim > 1 and ds.chunks[1:] != ds.shape[1:]:
        raise ValueError("This reader requires full trailing dimensions per chunk")
    result = np.empty(ds.shape, dtype=ds.dtype)
    audit = {"shape": list(ds.shape), "dtype": str(ds.dtype), "validated_chunks": 0,
             "recovered_chunks": [], "finite": None}
    for start in range(0, ds.shape[0], ds.chunks[0]):
        stop = min(start + ds.chunks[0], ds.shape[0])
        info = ds.id.get_chunk_info_by_coord((start,) + (0,) * (ds.ndim - 1))
        expected = int(np.prod(ds.chunks)) * ds.dtype.itemsize
        if (info.filter_mask != 0 or info.size != expected or info.byte_offset is None
                or info.byte_offset + info.size > file_size):
            raise ValueError(f"Invalid or incomplete raw chunk: {ds.name}, {start}")
        raw_file.seek(info.byte_offset)
        data = raw_file.read(info.size)
        if len(data) != info.size:
            raise ValueError("Short raw read")
        raw_chunk = np.frombuffer(data, dtype=ds.dtype).reshape(ds.chunks)[:stop-start]
        try:
            normal = ds[start:stop]
        except OSError as exc:
            if "addr overflow" not in str(exc):
                raise
            audit["recovered_chunks"].append({"rows": [start, stop], "offset": info.byte_offset,
                                              "bytes": info.size, "error": str(exc)})
            result[start:stop] = raw_chunk
        else:
            np.testing.assert_array_equal(raw_chunk, normal)
            result[start:stop] = normal
            audit["validated_chunks"] += 1
    audit["finite"] = bool(np.isfinite(result).all())
    if not audit["finite"]:
        raise ValueError(f"Non-finite values: {ds.name}")
    return result, audit


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, audit, profiles = {}, {}, {}
    size = SOURCE.stat().st_size
    names = ("timestamps_ns", "sequence", "type", "sys_time", "joints_position",
             "joints_velocity", "joints_effort")
    with h5py.File(SOURCE, "r") as f, SOURCE.open("rb") as raw:
        for group in ("franka_command", "franka_state"):
            audit[group] = {}
            for name in names:
                a, check = read_chunks(f[group][name], raw, size)
                arrays[f"{group}_{name}"] = a
                audit[group][name] = check
            n = len(arrays[f"{group}_timestamps_ns"])
            assert all(len(arrays[f"{group}_{name}"]) == n for name in names)
    urdf = ET.parse(ROOT / "ros_ws/src/franka_description/robots/franka_panda_gripper.urdf")
    limits = np.array([[float(urdf.find(f".//joint[@name='panda_joint{i}']/limit").get(k))
                        for k in ("lower", "upper")] for i in range(1, 8)])
    for group in ("franka_command", "franka_state"):
        t = arrays[f"{group}_timestamps_ns"]
        q = arrays[f"{group}_joints_position"]
        seq = arrays[f"{group}_sequence"].astype(np.int64)
        typ = arrays[f"{group}_type"]
        dt = np.diff(t) / 1e9
        steps = np.diff(q, axis=0)
        seq_steps, seq_counts = np.unique(np.diff(seq), return_counts=True)
        types, type_counts = np.unique(typ, return_counts=True)
        profile = {"samples": len(t), "first_time": clock(t[0]), "last_time": clock(t[-1]),
                   "duration_s": float((t[-1]-t[0])/1e9),
                   "dt_quantiles_s": np.quantile(dt,[0,.5,.95,1]).tolist(),
                   "nonpositive_dt": int((dt<=0).sum()),
                   "duplicate_timestamps": int(len(t)-len(np.unique(t))),
                   "sequence_step_counts": dict(zip(map(str,seq_steps), map(int,seq_counts))),
                   "types": dict(zip(map(str,types),map(int,type_counts))),
                   "first_q": q[0].tolist(), "last_q": q[-1].tolist(),
                   "min_q": q.min(0).tolist(), "max_q": q.max(0).tolist(),
                   "max_abs_step_rad": np.abs(steps).max(0).tolist(),
                   "outside_limits_count": ((q[:,:7]<limits[:,0])|(q[:,:7]>limits[:,1])).sum(0).tolist(),
                   "large_steps": {}}
        for joint in range(7):
            largest = np.argsort(np.abs(steps[:,joint]))[-5:][::-1]
            profile["large_steps"][str(joint+1)] = [
                {"index_before": int(i), "time_before": clock(t[i]), "time_after": clock(t[i+1]),
                 "dt_s": float(dt[i]), "from_rad": float(q[i,joint]),
                 "to_rad": float(q[i+1,joint]), "step_rad": float(steps[i,joint]),
                 "sequence_before": int(seq[i]), "sequence_after": int(seq[i+1])}
                for i in largest]
        profiles[group] = profile
    summary = {"source": str(SOURCE.relative_to(ROOT)), "source_size_bytes": size,
               "timezone": "Asia/Shanghai", "time_basis": "recorder receive timestamps_ns; not float32 sys_time",
               "state_machine_recorded": False, "joint_limits_rad": limits.tolist(),
               "audit": audit, "profiles": profiles}
    np.savez_compressed(OUT / "joint_data.npz", **arrays)
    (OUT / "diagnostics.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    for group, p in profiles.items():
        print(group, json.dumps({k: p[k] for k in (
            "samples", "first_time", "last_time", "outside_limits_count",
            "first_q", "min_q", "max_q")}, ensure_ascii=False))
    return summary, arrays


def motion_evidence(arrays):
    ct = arrays["franka_command_timestamps_ns"]
    st = arrays["franka_state_timestamps_ns"]
    cq = arrays["franka_command_joints_position"]
    sq = arrays["franka_state_joints_position"]
    sv = arrays["franka_state_joints_velocity"]
    moving = np.flatnonzero(np.max(np.abs(sv[:, :7]), axis=1) > 0.05)
    breaks = np.flatnonzero(np.diff(st[moving]) / 1e9 > 0.3) + 1
    episodes = []
    for indices in np.split(moving, breaks):
        if (st[indices[-1]] - st[indices[0]]) / 1e9 < 0.2:
            continue
        i, j = int(indices[0]), int(indices[-1])
        c = int(np.searchsorted(ct, st[i], side="right") - 1)
        if c < 0:
            continue
        episodes.append({
            "start": clock(st[i]), "end": clock(st[j]),
            "state_index_start": i, "state_index_end": j, "command_index_at_start": c,
            "command_age_ms": float((st[i]-ct[c])/1e6),
            "state_start_rad": sq[i, :7].tolist(), "state_end_rad": sq[j, :7].tolist(),
            "command_start_rad": cq[c, :7].tolist(),
            "initial_target_gap_deg": np.rad2deg(cq[c, :7]-sq[i, :7]).tolist(),
            "actual_motion_deg": np.rad2deg(sq[j, :7]-sq[i, :7]).tolist(),
            "max_speed_rad_s": np.max(np.abs(sv[i:j+1, :7]), axis=0).tolist(),
        })
    limits = np.array([[float(ET.parse(ROOT / "ros_ws/src/franka_description/robots/franka_panda_gripper.urdf")
                                    .find(f".//joint[@name='panda_joint{i}']/limit").get(k))
                        for k in ("lower", "upper")] for i in range(1, 8)])
    corrected = cq[:, :7].copy()
    corrected[:, 2] -= 2*np.pi
    cross = np.flatnonzero(sq[:, 2] > limits[2, 1])
    evidence = {
        "motion_segmentation": "Measured max |dq| > 0.05 rad/s; merge gaps <= 0.3s; keep duration >= 0.2s. Not state-machine events.",
        "alignment": "Latest recorded command at or before each observed state time, without interpolation.",
        "episodes": episodes,
        "hypothetical_j3_minus_2pi_range_rad": [float(corrected[:, 2].min()), float(corrected[:, 2].max())],
        "hypothetical_corrected_outside_limits_count": ((corrected < limits[:, 0]) | (corrected > limits[:, 1])).sum(0).tolist(),
        "first_j3_upper_limit_crossing_time": clock(st[cross[0]]) if len(cross) else None,
        "first_j3_upper_limit_crossing_value_rad": float(sq[cross[0], 2]) if len(cross) else None,
        "caveat": "The recording has no trigger/state-machine messages, raw encoder samples, or calibration reference pose; physical joint signs and F2 timings cannot be proven.",
    }
    (OUT / "motion_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n")
    print("Motion episodes:", len(episodes), "J3 first upper-limit crossing:", evidence["first_j3_upper_limit_crossing_time"])
    return evidence


def plot_diagnostics(arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    ct = arrays["franka_command_timestamps_ns"]
    st = arrays["franka_state_timestamps_ns"]
    cq = arrays["franka_command_joints_position"]
    sq = arrays["franka_state_joints_position"]
    sv = arrays["franka_state_joints_velocity"]
    origin = ct[0]
    xcmd, xstate = (ct-origin)/1e9, (st-origin)/1e9
    keep = xstate >= -3
    blue, actual, scenario = "#2459a6", "#252a31", "#b07a17"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(xcmd, cq[:, 0], color=blue, linewidth=1.6, label="GELLO target")
    axes[0].plot(xstate[keep], sq[keep, 0], color=actual, linewidth=1.6, label="Franka actual")
    axes[0].set_title("Joint 1: absolute targets and observed motion", loc="left", fontweight="bold")
    axes[0].set_ylabel("Joint 1 (rad)")
    axes[0].legend(loc="lower left", ncol=2, frameon=False)
    axes[1].plot(xcmd, cq[:, 2], color=blue, linewidth=1.6, label="GELLO target, recorded")
    axes[1].plot(xstate[keep], sq[keep, 2], color=actual, linewidth=1.6, label="Franka actual")
    axes[1].plot(xcmd, cq[:, 2]-2*np.pi, color=scenario, linestyle="--", linewidth=1.4,
                 label="Target minus 2*pi (offline scenario)")
    for limit in (-2.8973, 2.8973):
        axes[1].axhline(limit, color="#767c84", linestyle=":", linewidth=1.2,
                       label="Panda joint limits" if limit > 0 else None)
    axes[1].set_title("Joint 3: all recorded targets exceed the upper limit", loc="left", fontweight="bold")
    axes[1].set_ylabel("Joint 3 (rad)")
    axes[1].set_ylim(-3.4, 9.0)  # Reserve clear space above the data for the two-row legend.
    axes[1].legend(loc="upper left", ncol=2, frameon=False)
    axes[2].plot(xstate[keep], sv[keep, 0], color=blue, linewidth=1.4, label="Joint 1 actual velocity")
    axes[2].plot(xstate[keep], sv[keep, 2], color=actual, linewidth=1.4, label="Joint 3 actual velocity")
    axes[2].axhline(0, color="#767c84", linewidth=0.8)
    axes[2].set_title("Measured velocities: motion episodes are not F2 event timestamps", loc="left", fontweight="bold")
    axes[2].set_ylabel("Velocity (rad/s)")
    axes[2].legend(loc="lower left", ncol=2, frameon=False)
    for ax in axes:
        ax.grid(axis="y", color="#e0e3e8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-3, float(xcmd[-1]))
    ticks = np.unique(np.r_[0., np.arange(20, 121, 20), xcmd[-1]])
    axes[-1].set_xticks(ticks)
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: clock(origin+int(x*1e9))[11:19]))
    axes[-1].set_xlabel("2026-10-10 local time (UTC+08:00); recorder receive timestamps")
    fig.suptitle("GELLO / Franka recording: absolute target gaps and joint-limit violations", fontsize=15, fontweight="bold")
    fig.text(0.075, 0.035, "Source: recording_20261010_211159_1791637919729.h5 | 3,288 commands; 7,415 states\n"
             "Unfiltered tail chunks recovered read-only from their stored offsets; original file unchanged.\n"
             "The minus-2*pi curve is a calibration hypothesis, not a replay or a validated physical calibration.", fontsize=9)
    fig.tight_layout(rect=(0, 0.095, 1, 0.96))
    path = OUT / "joint_comparison.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    summary, arrays = analyze()
    motion_evidence(arrays)
    print("Saved plot:", plot_diagnostics(arrays))
