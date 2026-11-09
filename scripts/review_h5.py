#!/usr/bin/env python3
"""Start the local dual-camera H5 player and inclusive clip editor."""
import sys
from pathlib import Path

from lerobot_tools.cli import config_from_args, parser


def main() -> int:
    p = parser(__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--speed", type=float, default=1.0, help="Initial playback speed, 0.25 through 4")
    args = p.parse_args()
    if not 0.25 <= args.speed <= 4:
        p.error("--speed must be between 0.25 and 4")
    try:
        import uvicorn
        from lerobot_tools.server import create_app
        config = config_from_args(args)
        clips = args.clips or Path("outputs/lerobot_review/clips.json")
        app = create_app(config, clips)
        app.state.initial_speed = args.speed

        @app.get("/api/player")
        def player():
            return {"speed": args.speed}

        print(f"Review: http://{args.host}:{args.port}\nClip plan: {clips}\nOutput: {config.output}", flush=True)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"Cannot start review: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
