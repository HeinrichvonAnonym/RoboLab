#!/usr/bin/env python3
"""
Zenoh subscriber to visualize Kinect v2 frames published by `kinect_plugin`.

It expects protobuf payloads of type `kinect.rgbImage` (see `proto/kinect.proto`).

Examples (from repo root):
  python3 scripts/subscriber.py   # show Kinect + RealSense together
  python3 scripts/subscriber.py --key kinec2/rgb   # legacy single-key mode

If your `zenoh` Python package is broken in your environment, install/repair it:
  pip install -U eclipse-zenoh protobuf

GUI dependencies:
  pip install numpy matplotlib
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
    try:
        subprocess.run(
            ["protoc", "-I", str(repo_root / "proto"), f"--python_out={gen}", str(proto)],
            check=True,
        )
    except FileNotFoundError as e:
        print(
            "protoc not found. Install protobuf-compiler or regenerate manually:",
            file=sys.stderr,
        )
        print("  bash scripts/regen_proto_py.sh", file=sys.stderr)
        raise SystemExit(1) from e
    except subprocess.CalledProcessError as e:
        print("protoc failed; ensure `proto/kinect.proto` is valid.", file=sys.stderr)
        raise SystemExit(1) from e


def _decode_rgb_image(msg) -> Tuple[Optional[object], str]:
    """
    Returns (img, mode) where mode is one of: rgb, depth, ir, unknown.
    `img` is either a numpy RGB uint8 image or a 2D float array for colormap.
    """
    # Lazy import so the script can still run in non-GUI contexts.
    import numpy as np

    w = int(msg.width)
    h = int(msg.height)
    channels = int(msg.channels)
    step = int(msg.step)
    payload: bytes = msg.image
    img_type = int(msg.type)
    # kinect plugin: Frame::Type Color=1, Ir=2, Depth=4
    # realsense plugin: RS2_FORMAT RGB8=5, Z16=1, Y8=9

    if w <= 0 or h <= 0 or not payload:
        return None, "unknown"

    if channels <= 0 or step <= 0:
        return None, "unknown"

    # Realsense depth (Z16) arrives as type=1 with 2-byte pixels.
    if img_type == 1 and channels == 2:
        raw_z16 = np.frombuffer(payload, dtype=np.uint16)
        if len(raw_z16) >= h * w:
            arr = raw_z16[: h * w].reshape((h, w))
            return arr, "depth"
        return None, "unknown"

    if img_type in (1, 5):  # Kinect Color or RealSense RGB8
        raw = np.frombuffer(payload, dtype=np.uint8)
        # Some pipelines may include stride; respect `step` if present.
        if len(raw) >= h * step:
            raw = raw.reshape((h, step))
            raw = raw[:, : w * channels].reshape((h, w, channels))
        else:
            raw = raw[: h * w * channels].reshape((h, w, channels))

        # libfreenect2 Color is commonly BGRX (4 bytes). Our plugin publishes raw `frame->data`.
        if channels == 4:
            bgr = raw[:, :, 0:3]
            rgb = bgr[:, :, ::-1]
            return rgb, "rgb"
        if channels == 3:
            return raw, "rgb"
        if channels == 1:
            return raw[:, :, 0], "rgb"
        return raw, "unknown"

    # Kinect Depth/IR are float32.
    # Our plugin publishes raw `frame->data` and sets channels=bytes_per_pixel (4).
    if channels == 4 and img_type in (2, 4):
        raw_f = np.frombuffer(payload, dtype=np.float32)
        if len(raw_f) >= h * w:
            arr = raw_f[: h * w].reshape((h, w))
        else:
            return None, "unknown"
        return arr, "depth" if img_type == 4 else "ir"

    # RealSense IR (Y8) published as type=9 with channels=1.
    if img_type == 9 and channels == 1:
        raw_y8 = np.frombuffer(payload, dtype=np.uint8)
        if len(raw_y8) >= h * w:
            arr = raw_y8[: h * w].reshape((h, w))
            return arr, "ir"
        return None, "unknown"

    return None, "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--kinect-rgb-key",
        default="kinec2/rgb",
        help="Kinect RGB key (default: kinec2/rgb).",
    )
    parser.add_argument(
        "--kinect-depth-key",
        default="kinec2/depth",
        help="Kinect depth key (default: kinec2/depth).",
    )
    parser.add_argument(
        "--realsense-rgb-key",
        default="realsense/rgb",
        help="RealSense RGB key (default: realsense/rgb).",
    )
    parser.add_argument(
        "--realsense-depth-key",
        default="realsense/depth",
        help="RealSense depth key (default: realsense/depth).",
    )
    parser.add_argument(
        "--key",
        default="",
        help="Legacy single-key mode. If set, subscribes to this key only.",
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
        "--no-gui",
        action="store_true",
        help="Do not open a matplotlib window; print frame stats instead.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=10.0,
        help="GUI update rate (matplotlib redraw cadence).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_kinect_pb2(repo_root)
    sys.path.insert(0, str(repo_root / "scripts" / "gen"))

    try:
        import kinect_pb2  # noqa: F401
    except ImportError as e:
        print("Could not import generated kinect_pb2.py", file=sys.stderr)
        raise SystemExit(1) from e

    try:
        import zenoh
    except Exception as e:
        print(
            "Could not import `zenoh` Python package. Fix with:",
            file=sys.stderr,
        )
        print("  pip install -U eclipse-zenoh protobuf", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return 1

    # Callback runs in Zenoh threads; push decoded frames to queues.
    latest_kinect_rgb: "Queue[Tuple[Optional[object], str, int, int]]" = Queue(maxsize=1)
    latest_kinect_depth: "Queue[Tuple[Optional[object], str, int, int]]" = Queue(maxsize=1)
    latest_rs_rgb: "Queue[Tuple[Optional[object], str, int, int]]" = Queue(maxsize=1)
    latest_rs_depth: "Queue[Tuple[Optional[object], str, int, int]]" = Queue(maxsize=1)

    def _push_latest(queue: Queue, item: Tuple[Optional[object], str, int, int]) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except Exception:
                pass
        queue.put_nowait(item)

    def on_sample_to(queue: Queue, label: str, sample) -> None:
        try:
            payload_bytes: bytes = sample.payload.to_bytes()
            msg = kinect_pb2.rgbImage()
            msg.ParseFromString(payload_bytes)
            img, mode = _decode_rgb_image(msg)
            _push_latest(queue, (img, mode, int(msg.width), int(msg.height)))
        except Exception as e:
            print(f"{label} subscriber callback error: {e}", file=sys.stderr)

    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", json.dumps(list(args.connect)))

    # Legacy override: when --key is given, use single-key mode.
    if args.key:
        single_key = args.key
    else:
        single_key = ""

    with zenoh.open(conf) as session:
        if single_key:
            sub_single = session.declare_subscriber(
                single_key, lambda s: on_sample_to(latest_kinect_rgb, "single", s)
            )
            print(f"Subscribed to '{single_key}' (single-key mode). Press Ctrl+C to stop.")
        else:
            sub_k_rgb = session.declare_subscriber(
                args.kinect_rgb_key, lambda s: on_sample_to(latest_kinect_rgb, "kinect-rgb", s)
            )
            sub_k_depth = session.declare_subscriber(
                args.kinect_depth_key, lambda s: on_sample_to(latest_kinect_depth, "kinect-depth", s)
            )
            sub_r_rgb = session.declare_subscriber(
                args.realsense_rgb_key, lambda s: on_sample_to(latest_rs_rgb, "realsense-rgb", s)
            )
            sub_r_depth = session.declare_subscriber(
                args.realsense_depth_key,
                lambda s: on_sample_to(latest_rs_depth, "realsense-depth", s),
            )
            print(
                "Subscribed to Kinect+RealSense streams:\n"
                f"  Kinect RGB={args.kinect_rgb_key}, Kinect Depth={args.kinect_depth_key}\n"
                f"  RealSense RGB={args.realsense_rgb_key}, RealSense Depth={args.realsense_depth_key}\n"
                "Press Ctrl+C to stop."
            )

        if args.no_gui:
            while True:
                if not latest_kinect_rgb.empty():
                    img, mode, w, h = latest_kinect_rgb.get_nowait()
                    if img is None:
                        print(f"[{time.strftime('%H:%M:%S')}] rgb mode={mode} empty frame size=({w},{h})")
                    elif hasattr(img, "shape"):
                        print(f"[{time.strftime('%H:%M:%S')}] rgb mode={mode} shape={img.shape} dtype={img.dtype}")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] rgb mode={mode} frame updated")

                if not single_key:
                    if not latest_kinect_depth.empty():
                        img, mode, w, h = latest_kinect_depth.get_nowait()
                        if img is None:
                            print(f"[{time.strftime('%H:%M:%S')}] kinect depth mode={mode} empty size=({w},{h})")
                        elif hasattr(img, "shape"):
                            print(f"[{time.strftime('%H:%M:%S')}] kinect depth mode={mode} shape={img.shape} dtype={img.dtype}")
                    if not latest_rs_rgb.empty():
                        img, mode, w, h = latest_rs_rgb.get_nowait()
                        if img is None:
                            print(f"[{time.strftime('%H:%M:%S')}] realsense rgb mode={mode} empty size=({w},{h})")
                        elif hasattr(img, "shape"):
                            print(f"[{time.strftime('%H:%M:%S')}] realsense rgb mode={mode} shape={img.shape} dtype={img.dtype}")
                    if not latest_rs_depth.empty():
                        img, mode, w, h = latest_rs_depth.get_nowait()
                        if img is None:
                            print(f"[{time.strftime('%H:%M:%S')}] realsense depth mode={mode} empty size=({w},{h})")
                        elif hasattr(img, "shape"):
                            print(f"[{time.strftime('%H:%M:%S')}] realsense depth mode={mode} shape={img.shape} dtype={img.dtype}")

                time.sleep(0.005)
        else:
            import matplotlib.pyplot as plt
            import numpy as np

            plt.ion()
            if single_key:
                fig, ax_k_rgb = plt.subplots(1, 1, figsize=(6, 4))
                fig.suptitle(single_key)
                ax_k_depth = None
                ax_r_rgb = None
                ax_r_depth = None
            else:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                ax_k_rgb, ax_k_depth = axes[0, 0], axes[0, 1]
                ax_r_rgb, ax_r_depth = axes[1, 0], axes[1, 1]
                fig.suptitle("Kinect + RealSense")
                ax_k_depth.set_title("Kinect Depth waiting...")
                ax_r_rgb.set_title("RealSense RGB waiting...")
                ax_r_depth.set_title("RealSense Depth waiting...")
            ax_k_rgb.set_title("Kinect RGB waiting...")

            latest_k_rgb_img = None
            latest_k_rgb_mode = "unknown"
            latest_k_rgb_size = (0, 0)
            latest_k_depth_img = None
            latest_k_depth_mode = "unknown"
            latest_k_depth_size = (0, 0)
            latest_r_rgb_img = None
            latest_r_rgb_mode = "unknown"
            latest_r_rgb_size = (0, 0)
            latest_r_depth_img = None
            latest_r_depth_mode = "unknown"
            latest_r_depth_size = (0, 0)

            last_draw = 0.0
            while True:
                if not latest_kinect_rgb.empty():
                    latest_k_rgb_img, latest_k_rgb_mode, w, h = latest_kinect_rgb.get_nowait()
                    latest_k_rgb_size = (w, h)
                if not single_key:
                    if not latest_kinect_depth.empty():
                        latest_k_depth_img, latest_k_depth_mode, w, h = latest_kinect_depth.get_nowait()
                        latest_k_depth_size = (w, h)
                    if not latest_rs_rgb.empty():
                        latest_r_rgb_img, latest_r_rgb_mode, w, h = latest_rs_rgb.get_nowait()
                        latest_r_rgb_size = (w, h)
                    if not latest_rs_depth.empty():
                        latest_r_depth_img, latest_r_depth_mode, w, h = latest_rs_depth.get_nowait()
                        latest_r_depth_size = (w, h)

                now = time.time()
                if now - last_draw < (1.0 / max(args.target_fps, 1e-6)):
                    time.sleep(0.001)
                    continue
                last_draw = now

                ax_k_rgb.clear()
                ax_k_rgb.axis("off")
                if latest_k_rgb_img is None:
                    ax_k_rgb.set_title(
                        f"Kinect RGB waiting (mode={latest_k_rgb_mode}, size={latest_k_rgb_size})"
                    )
                elif latest_k_rgb_mode == "rgb":
                    ax_k_rgb.imshow(latest_k_rgb_img)
                    ax_k_rgb.set_title(f"Kinect RGB mode={latest_k_rgb_mode} size={latest_k_rgb_size}")
                else:
                    arr = np.nan_to_num(
                        latest_k_rgb_img, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    ax_k_rgb.imshow(arr, cmap="inferno")
                    ax_k_rgb.set_title(f"Kinect RGB mode={latest_k_rgb_mode} size={latest_k_rgb_size}")

                if not single_key:
                    ax_k_depth.clear()
                    ax_k_depth.axis("off")
                    if latest_k_depth_img is None:
                        ax_k_depth.set_title(
                            f"Kinect Depth waiting (mode={latest_k_depth_mode}, size={latest_k_depth_size})"
                        )
                    elif latest_k_depth_mode in ("depth", "ir"):
                        arr = np.nan_to_num(
                            latest_k_depth_img, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        ax_k_depth.imshow(arr, cmap="inferno")
                        ax_k_depth.set_title(
                            f"Kinect Depth mode={latest_k_depth_mode} size={latest_k_depth_size}"
                        )
                    ax_r_rgb.clear()
                    ax_r_rgb.axis("off")
                    if latest_r_rgb_img is None:
                        ax_r_rgb.set_title(
                            f"RealSense RGB waiting (mode={latest_r_rgb_mode}, size={latest_r_rgb_size})"
                        )
                    elif latest_r_rgb_mode == "rgb":
                        ax_r_rgb.imshow(latest_r_rgb_img)
                        ax_r_rgb.set_title(
                            f"RealSense RGB mode={latest_r_rgb_mode} size={latest_r_rgb_size}"
                        )
                    else:
                        arr = np.nan_to_num(
                            latest_r_rgb_img, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        ax_r_rgb.imshow(arr, cmap="inferno")
                        ax_r_rgb.set_title(
                            f"RealSense RGB mode={latest_r_rgb_mode} size={latest_r_rgb_size}"
                        )

                    ax_r_depth.clear()
                    ax_r_depth.axis("off")
                    if latest_r_depth_img is None:
                        ax_r_depth.set_title(
                            f"RealSense Depth waiting (mode={latest_r_depth_mode}, size={latest_r_depth_size})"
                        )
                    elif latest_r_depth_mode in ("depth", "ir"):
                        arr = np.nan_to_num(
                            latest_r_depth_img, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        ax_r_depth.imshow(arr, cmap="inferno")
                        ax_r_depth.set_title(
                            f"RealSense Depth mode={latest_r_depth_mode} size={latest_r_depth_size}"
                        )
                    else:
                        ax_r_depth.imshow(latest_r_depth_img)
                        ax_r_depth.set_title(
                            f"RealSense Depth mode={latest_r_depth_mode} size={latest_r_depth_size}"
                        )

                fig.canvas.draw()
                plt.pause(0.05)

        if single_key:
            _ = sub_single
        else:
            _ = (sub_k_rgb, sub_k_depth, sub_r_rgb, sub_r_depth)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

