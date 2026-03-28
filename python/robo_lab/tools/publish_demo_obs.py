#!/usr/bin/env python3
"""
Publish demo_inference.Observation (17 floats) over Zenoh for demo_inference_plugin.

Run from repo root (or set PYTHONPATH to include python/):

  PYTHONPATH=python python -m robo_lab.tools.publish_demo_obs --topic demo/obs/state --rate 50

Also subscribes to the policy action topic (default demo/action/command) and prints parsed Action
values when the inference plugin publishes.

Requires: pip install zenoh protobuf

Zenoh: default follows zenoh (multicast scouting on) so publisher and plugin on one host
discover each other. On busy LANs set ROBO_LAB_ZENOH_MULTICAST_SCOUTING=0 and use a router
or explicit connect/endpoints.
"""
from __future__ import annotations

import os

# Before any google.protobuf / *_pb2 import: helps when system protoc is old but
# `pip install protobuf` is 4.x+. Prefer: pip install "protobuf>=3.20,<4" (see requirements.txt).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import json
import math
import sys
import time

from google.protobuf.message import DecodeError

# Repo layout: python/robo_lab/proto_gen/demo_inference_pb2.py
if __name__ == "__main__" and __package__ is None:
    from pathlib import Path

    # python/ directory must be on PYTHONPATH
    _py_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_py_root))

from robo_lab.proto_gen import demo_inference_pb2 as di_pb2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish Observation over Zenoh and optionally print Action replies from inference."
    )
    parser.add_argument("--topic", default="demo/obs/state", help="Zenoh key expression for Observation puts")
    parser.add_argument(
        "--action-topic",
        default="demo/action/command",
        help="Zenoh key expression to subscribe for demo_inference.Action (empty disables)",
    )
    parser.add_argument("--rate", type=float, default=20.0, help="Publish rate (Hz)")
    parser.add_argument("--dim", type=int, default=19, help="Observation dimension (must match YAML/engine)")
    args = parser.parse_args()

    if args.dim < 1:
        raise SystemExit("--dim must be >= 1")

    try:
        import zenoh
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: pip install zenoh\n"
            f"Import error: {e}"
        ) from e

    conf = zenoh.Config()
    if os.environ.get("ROBO_LAB_ZENOH_MULTICAST_SCOUTING") == "0":
        conf.insert_json5("scouting/multicast/enabled", json.dumps(False))
    session = zenoh.open(conf)

    action_sub = None
    if args.action_topic:

        def on_action(sample: zenoh.Sample) -> None:
            raw = sample.payload.to_bytes()
            act = di_pb2.Action()
            try:
                act.ParseFromString(raw)
            except DecodeError:
                print(f"[action] key={sample.key_expr} len={len(raw)} (Action parse failed)")
                return
            print(
                f"[action] key={sample.key_expr} dim={len(act.values)} "
                f"values={[round(v, 6) for v in act.values]}"
            )

        print(f"Subscribing to action topic {args.action_topic!r} ...")
        action_sub = session.declare_subscriber(args.action_topic, on_action)

    period = 1.0 / args.rate if args.rate > 0 else 0.1
    t0 = time.time()
    n = 0
    try:
        while True:
            obs = di_pb2.Observation()
            # Demo signal: gentle sine on each dim (replace with real state).
            t = time.time() - t0
            for i in range(args.dim):
                obs.values.append(0.1 * math.sin(t + 0.2 * i))
            payload = obs.SerializeToString()
            session.put(args.topic, payload)
            n += 1
            if n % int(max(args.rate, 1)) == 0:
                print(f"Published {n} Observation messages on {args.topic!r} (dim={args.dim})")
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
