#!/usr/bin/env python3
"""
Zenoh subscriber to visualize Kinect v2 frames published by `kinect_plugin`.

It expects protobuf payloads of type `kinect.rgbImage` (see `proto/kinect.proto`).

Example (from repo root):
  python3 scripts/subscriber.py --key kinec2/rgb

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
    img_type = int(msg.type)  # libfreenect2 Frame::Type: Color=1, Ir=2, Depth=4

    if w <= 0 or h <= 0 or not payload:
        return None, "unknown"

    if channels <= 0 or step <= 0:
        return None, "unknown"

    if img_type == 1:  # Color
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

    # Depth/IR are float32 frames in libfreenect2.
    # Our plugin publishes raw `frame->data` and sets channels=bytes_per_pixel (4).
    if channels == 4 and img_type in (2, 4):
        raw_f = np.frombuffer(payload, dtype=np.float32)
        if len(raw_f) >= h * w:
            arr = raw_f[: h * w].reshape((h, w))
        else:
            return None, "unknown"
        return arr, "depth" if img_type == 4 else "ir"

    return None, "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--key",
        default="kinec2/rgb",
        help="Zenoh key expression to subscribe (e.g. kinec2/rgb).",
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

    # GUI pipeline: callback runs in Zenoh threads, so we push frames to a queue.
    latest: "Queue[Tuple[Optional[object], str, int, int]]" = Queue(maxsize=1)

    def on_sample(sample) -> None:
        try:
            payload_bytes: bytes = sample.payload.to_bytes()
            msg = kinect_pb2.rgbImage()
            msg.ParseFromString(payload_bytes)

            img, mode = _decode_rgb_image(msg)
            if latest.full():
                try:
                    latest.get_nowait()
                except Exception:
                    pass
            latest.put_nowait((img, mode, int(msg.width), int(msg.height)))
        except Exception as e:
            # Avoid crashing callback thread.
            print(f"subscriber callback error: {e}", file=sys.stderr)

    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", json.dumps(list(args.connect)))

    # Subscribe and render.
    with zenoh.open(conf) as session:
        sub = session.declare_subscriber(args.key, on_sample)
        print(f"Subscribed to '{args.key}'. Press Ctrl+C to stop.")

        if args.no_gui:
            while True:
                img, mode, w, h = latest.get()
                if img is None:
                    print(f"[{time.strftime('%H:%M:%S')}] mode={mode} empty frame size=({w},{h})")
                    continue
                try:
                    import numpy as np

                    if hasattr(img, "shape"):
                        print(f"[{time.strftime('%H:%M:%S')}] mode={mode} shape={img.shape} dtype={img.dtype}")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] mode={mode} type={type(img)}")
                except Exception:
                    print(f"[{time.strftime('%H:%M:%S')}] mode={mode} frame updated")
        else:
            import matplotlib.pyplot as plt

            # Create figure once; update image in place.
            plt.ion()
            fig, ax = plt.subplots(1, 1)
            img_artist = None
            title_artist = ax.set_title("waiting for frames...")

            last_draw = 0.0
            while True:
                img, mode, w, h = latest.get()
                now = time.time()
                if img is None:
                    ax.set_title(f"waiting (mode={mode}, size=({w},{h}))")
                    fig.canvas.draw()
                    plt.pause(0.001)
                    continue

                # Throttle redraws.
                if now - last_draw < (1.0 / max(args.target_fps, 1e-6)):
                    continue
                last_draw = now

                ax.clear()
                ax.axis("off")
                if mode == "rgb":
                    ax.imshow(img)
                elif mode in ("depth", "ir"):
                    import numpy as np

                    arr = img
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    ax.imshow(arr, cmap="inferno")
                else:
                    ax.imshow(img)

                ax.set_title(f"{args.key} mode={mode} size=({w},{h})")
                fig.canvas.draw()
                plt.pause(0.001)

        # Keep subscription alive until exit.
        _ = sub

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

