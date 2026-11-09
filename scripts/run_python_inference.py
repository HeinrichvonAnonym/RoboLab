#!/usr/bin/env python3
"""Run gated Zenoh inference, or verify H5/circle offline without opening Zenoh."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from robo_lab.inference.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config/inference/python_circle.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Publish on real configured topics, gated by F2")
    mode.add_argument("--offline-h5", type=Path, help="Read-only H5 and generated trajectory verification")
    parser.add_argument("--connect", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = args.output or ROOT/"outputs/python_inference"/(datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    snapshot = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}
    (out/"config.json").write_text(json.dumps(snapshot, indent=2))
    if args.offline_h5:
        from robo_lab.inference.offline import verify_h5
        report = verify_h5(args.offline_h5, cfg, out)
        print(json.dumps(report, indent=2))
        return
    import zenoh
    from robo_lab.inference.transport import JsonLog, ZenohNode, zenoh_config
    stop = False
    node = None

    def stopping(*_):
        nonlocal stop
        stop = True
        if node is not None:
            # Invalidate an in-flight inference result before it can publish.
            node.executor.set_state("standby")

    signal.signal(signal.SIGINT, stopping)
    signal.signal(signal.SIGTERM, stopping)
    log = JsonLog(out/"events.jsonl")
    try:
        with zenoh.open(zenoh_config(connect=args.connect)) as session:
            node = ZenohNode(cfg, session, log)
            print(f"Waiting for F2 state transitions; log: {out}", flush=True)
            try:
                while not stop:
                    node.executor.tick()
                    time.sleep(max(0.0005, min(0.005, node.executor.next_due-time.monotonic())))
            finally:
                node.close()
    finally:
        log.close()


if __name__ == "__main__":
    main()
