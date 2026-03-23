from __future__ import annotations

import argparse
from pathlib import Path
from threading import Thread

import uvicorn

from .api import create_app
from .api.state import RuntimeState
from .config import default_config
from .pipeline import run_pipeline
from .repro import collect_runtime_manifest, set_global_seed, write_run_manifest
from .video_io import VideoSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime attention API + pipeline")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--camera", type=int, help="Camera index")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--max-seconds", type=int, help="Optional stop duration")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=str, help="Optional YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video and args.camera is None:
        raise SystemExit("Provide --video or --camera.")

    state = RuntimeState()
    app = create_app(state=state, dashboard_dir="dashboard")
    server = uvicorn.Server(uvicorn.Config(app=app, host=args.host, port=args.port, log_level="info"))
    Thread(target=server.run, daemon=True).start()

    config = default_config(config_path=args.config)
    config.output_dir = Path(args.output_dir)
    set_global_seed(config.seed)
    if config.save_run_manifest:
        write_run_manifest(
            config.output_dir,
            collect_runtime_manifest({"experiment_name": config.experiment_name, "seed": config.seed}),
        )

    def _publish(payload):
        state.publish_sync(payload)

    source = VideoSource(path=args.video, camera_index=args.camera)
    run_pipeline(
        config=config,
        source=source,
        max_seconds=args.max_seconds,
        output_video_path=None,
        realtime_callback=_publish,
    )
    server.should_exit = True


if __name__ == "__main__":
    main()
