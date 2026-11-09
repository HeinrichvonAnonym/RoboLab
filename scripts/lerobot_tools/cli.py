"""Shared command-line configuration."""
import argparse
from pathlib import Path

from .core import Config


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--input-dir", default="data", help="Completed recorder H5 directory")
    p.add_argument("--since", default=Config.since, help="Inclusive recording start; naive times use Asia/Shanghai")
    p.add_argument("--output", default=Config.output, help="Managed symlink to the current dataset generation")
    p.add_argument("--fps", type=int, default=15, help="Export/synchronized grid FPS, independent of playback speed")
    p.add_argument("--task", default=Config.task)
    p.add_argument("--max-gap-ms", type=float, default=200.0, help="Above this acquisition gap hold previous samples")
    p.add_argument("--repo-id", default=Config.repo_id, help="Local metadata identifier; nothing is uploaded")
    p.add_argument("--clips", type=Path, help="Saved JSON clip plan; omitted in CLI means keep all")
    return p


def config_from_args(args) -> Config:
    return Config(**{field: getattr(args, field) for field in Config.__dataclass_fields__})
