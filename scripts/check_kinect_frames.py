#!/usr/bin/env python3
"""Check Kinect messages from an already running roboLab instance.

Run with the roboLab Python environment. This only subscribes to image topics;
it does not open the camera, start plugins, or publish robot/keyboard commands.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

import yaml
import zenoh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from robo_lab.proto_gen.kinect_pb2 import rgbImage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/plugins/kinect_plugin_config.yaml")
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 3:
        parser.error("duration must be finite and at least 3 seconds")
    config = yaml.safe_load(args.config.read_text())
    streams = []
    if config.get("enable_rgb", True):
        streams.append("rgb")
    if config.get("enable_depth", True):
        streams.extend(("depth", "ir"))
    if not streams:
        parser.error("no enabled image streams")
    sizes = {"rgb": (1920, 1080), "depth": (512, 424), "ir": (512, 424)}
    types = {"rgb": 1, "depth": 4, "ir": 2}
    factors = config.get("downsample", {})
    topics = config.get("topic", {})
    expected = {}
    for key in streams:
        factor = factors.get(key, 1)
        if type(factor) is not int or factor <= 0:
            parser.error(f"downsample.{key} must be a positive integer")
        width, height = sizes[key]
        expected[key] = ((width + factor - 1) // factor, (height + factor - 1) // factor)
    lock = threading.Lock()
    stats = {key: dict(frames=0, first=None, last=None, max_gap_seconds=0.0,
                       image_bytes=0) for key in streams}
    errors = []

    def receive(key, sample):
        received = time.monotonic()
        try:
            message = rgbImage.FromString(sample.payload.to_bytes())
            if ((message.width, message.height) != expected[key]
                    or message.channels != 4 or message.type != types[key]
                    or message.step != message.width * 4
                    or len(message.image) != message.step * message.height):
                raise ValueError(f"{key}: inconsistent image dimensions, stride, length or type")
            with lock:
                row = stats[key]
                if row["last"] is not None:
                    row["max_gap_seconds"] = max(row["max_gap_seconds"], received - row["last"])
                if row["first"] is None:
                    row["first"] = received
                row["last"] = received
                row["frames"] += 1
                row["image_bytes"] += len(message.image)
        except Exception as error:
            with lock:
                if len(errors) < 5:
                    errors.append(str(error))

    with zenoh.open(zenoh.Config()) as session:
        subscribers = [session.declare_subscriber(topics.get(key, f"kinect/{key}"),
                       lambda sample, key=key: receive(key, sample)) for key in streams]
        print("Listening to " + ", ".join(topics.get(key, f"kinect/{key}") for key in streams), flush=True)
        begin = time.monotonic()
        while time.monotonic() - begin < args.duration:
            time.sleep(min(3, max(0, args.duration - (time.monotonic() - begin))))
            with lock:
                print({key: row["frames"] for key, row in stats.items()}, flush=True)
        end = time.monotonic()
        for subscriber in subscribers:
            subscriber.undeclare()
    with lock:
        result = dict(seconds=end - begin, streams={}, errors=errors)
        for key, row in stats.items():
            width, height = expected[key]
            interval = ((row["last"] - row["first"]) if row["frames"] > 1 else 0)
            result["streams"][key] = dict(
                width=width, height=height, step=width * 4, channels=4, type=types[key],
                image_bytes_per_frame=width * height * 4, frames=row["frames"],
                fps=(row["frames"] - 1) / interval if interval else 0,
                max_gap_seconds=row["max_gap_seconds"],
                last_frame_age_seconds=end - row["last"] if row["last"] is not None else None)
        result["passed"] = not errors and all(
            row["frames"] >= 3 and row["max_gap_seconds"] <= 2 and
            row["last_frame_age_seconds"] is not None and row["last_frame_age_seconds"] <= 2
            for row in result["streams"].values())
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
