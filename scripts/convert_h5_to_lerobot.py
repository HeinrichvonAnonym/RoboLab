#!/usr/bin/env python3
"""Headless conversion of recorder H5 to synchronized LeRobot v3."""
import json
import sys

from lerobot_tools.cli import config_from_args, parser
from lerobot_tools.core import load_plan, scan
from lerobot_tools.export import export_dataset


def main() -> int:
    p = parser(__doc__)
    p.add_argument("--scan", action="store_true", help="Inspect sources and clips without writing a dataset")
    p.add_argument("--file", action="append", dest="files", help="Only this selected H5 basename; repeatable")
    p.add_argument("--max-frames", type=int, help="Limit retained frames PER SOURCE for smoke exports")
    args = p.parse_args()
    try:
        config = config_from_args(args)
        if args.clips is not None and not args.clips.is_file():
            raise ValueError(f"Clip plan not found: {args.clips}")
        plan = load_plan(args.clips)
        if args.scan:
            entries = scan(config, plan)
            if args.files:
                entries = [e for e in entries if e["name"] in args.files]
            print(json.dumps({"recordings": entries, "valid": sum(e["valid"] for e in entries), "selected": len(entries)}, ensure_ascii=False, indent=2))
        else:
            result = export_dataset(config, plan, files=args.files, max_frames=args.max_frames,
                progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
            print(json.dumps({"frames": result["total_frames"], "episodes": result["total_episodes"],
                "output": config.output, "skipped": result["skipped"]}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
