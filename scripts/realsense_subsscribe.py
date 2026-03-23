#!/usr/bin/env python3
"""
Zenoh subscriber for RealSense RGB + depth topics.

Default topics match `config/plugins/realsense_plugin_config.yaml`:
  - realsense/rgb
  - realsense/depth

Example:
  python3 scripts/realsense_subsscribe.py
  python3 scripts/realsense_subsscribe.py --no-gui
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from queue import Queue
from typing import Optional, Tuple


def _ensure_kinect_pb2(repo_root: Path) -> None:
    gen = repo_root / "scripts" / "gen"
    pb2 = gen / "kinect_pb2.py"
    proto = repo_root / "proto" / "kinect.proto"
    if pb2.is_file():
        return

    gen.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["protoc", "-I", str(repo_root / "proto"), f"--python_out={gen}", str(proto)],
        check=True,
    )


def _decode_msg(msg) -> Tuple[Optional[object], str, int, int]:
    import numpy as np

    w = int(msg.width)
    h = int(msg.height)
    channels = int(msg.channels)
    payload: bytes = msg.image
    if w <= 0 or h <= 0 or not payload:
        return None, "unknown", w, h

    # rgb publisher uses RGB8 (1 byte channel).
    if msg.type == 5:  # RS2_FORMAT_RGB8
        raw = np.frombuffer(payload, dtype=np.uint8)
        if channels >= 3 and raw.size >= h * w * channels:
            arr = raw[: h * w * channels].reshape((h, w, channels))
            return arr[:, :, :3], "rgb", w, h
        return None, "rgb", w, h

    # depth publisher uses Z16 (2 bytes pixel).
    if msg.type == 1:  # RS2_FORMAT_Z16
        raw = np.frombuffer(payload, dtype=np.uint16)
        if raw.size >= h * w:
            arr = raw[: h * w].reshape((h, w))
            return arr, "depth", w, h
        return None, "depth", w, h

    return None, "unknown", w, h


def main() -> int:
    parser = argparse.ArgumentParser(description="Subscribe and visualize RealSense streams")
    parser.add_argument("--rgb-key", default="realsense/rgb", help="Zenoh key for RGB frames")
    parser.add_argument("--depth-key", default="realsense/depth", help="Zenoh key for depth frames")
    parser.add_argument("--connect", "-e", action="append", default=None, help="Zenoh endpoint")
    parser.add_argument("--no-gui", action="store_true", help="Print frame stats only")
    parser.add_argument("--target-fps", type=float, default=15.0, help="GUI redraw rate")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    try:
        _ensure_kinect_pb2(repo_root)
    except Exception as e:
        print(f"Failed to generate kinect_pb2.py: {e}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(repo_root / "scripts" / "gen"))

    import kinect_pb2  # type: ignore

    try:
        import zenoh
    except Exception as e:
        print("Could not import zenoh. Try: pip install -U eclipse-zenoh protobuf", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return 1

    latest_rgb: "Queue[Tuple[Optional[object], str, int, int]]" = Queue(maxsize=1)
    latest_depth: "Queue[Tuple[Optional[object], str, int, int]]" = Queue(maxsize=1)

    def _push(q: Queue, item) -> None:
        if q.full():
            try:
                q.get_nowait()
            except Exception:
                pass
        q.put_nowait(item)

    def on_rgb(sample) -> None:
        try:
            msg = kinect_pb2.rgbImage()
            msg.ParseFromString(sample.payload.to_bytes())
            _push(latest_rgb, _decode_msg(msg))
        except Exception as e:
            print(f"rgb callback error: {e}", file=sys.stderr)

    def on_depth(sample) -> None:
        try:
            msg = kinect_pb2.rgbImage()
            msg.ParseFromString(sample.payload.to_bytes())
            _push(latest_depth, _decode_msg(msg))
        except Exception as e:
            print(f"depth callback error: {e}", file=sys.stderr)

    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", json.dumps(list(args.connect)))

    with zenoh.open(conf) as session:
        sub_rgb = session.declare_subscriber(args.rgb_key, on_rgb)
        sub_depth = session.declare_subscriber(args.depth_key, on_depth)
        print(f"Subscribed to '{args.rgb_key}' and '{args.depth_key}'. Press Ctrl+C to stop.")

        if args.no_gui:
            while True:
                if not latest_rgb.empty():
                    img, mode, w, h = latest_rgb.get()
                    print(f"[{time.strftime('%H:%M:%S')}] rgb mode={mode} size=({w},{h})")
                if not latest_depth.empty():
                    img, mode, w, h = latest_depth.get()
                    print(f"[{time.strftime('%H:%M:%S')}] depth mode={mode} size=({w},{h})")
                time.sleep(0.02)
        else:
            import matplotlib.pyplot as plt
            import numpy as np

            plt.ion()
            fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(10, 4))
            last_draw = 0.0
            rgb_img = None
            depth_img = None

            while True:
                if not latest_rgb.empty():
                    rgb_img, _, _, _ = latest_rgb.get()
                if not latest_depth.empty():
                    depth_img, _, _, _ = latest_depth.get()

                now = time.time()
                if now - last_draw < (1.0 / max(args.target_fps, 1e-6)):
                    time.sleep(0.001)
                    continue
                last_draw = now

                ax_rgb.clear()
                ax_depth.clear()
                ax_rgb.axis("off")
                ax_depth.axis("off")
                ax_rgb.set_title("RealSense RGB")
                ax_depth.set_title("RealSense Depth")

                if rgb_img is not None:
                    ax_rgb.imshow(rgb_img)
                if depth_img is not None:
                    d = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
                    ax_depth.imshow(d, cmap="inferno")

                fig.canvas.draw()
                plt.pause(0.001)

        _ = (sub_rgb, sub_depth)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

