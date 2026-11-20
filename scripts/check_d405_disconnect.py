#!/usr/bin/env python3
"""Operator-assisted D405 unplug/replug acceptance. Never starts a robot bringup."""
import argparse
import json
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=float, default=600)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/qa/d405/disconnect")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera_path = args.output_dir / "camera.log"
    preview_path = args.output_dir / "preview.log"
    result = {"passed": False}
    camera = preview = None

    def stop(process):
        if process and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError("Process did not stop within 10 seconds")

    def wait_for(predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.1)
        raise RuntimeError("Timed out waiting for operator/device/stream transition")

    def present():
        for entry in Path("/sys/bus/usb/devices").glob("*/idProduct"):
            try:
                if entry.read_text().strip() == "0b5b" and (entry.parent / "idVendor").read_text().strip() == "8086":
                    return True
            except OSError:
                pass
        return False

    with camera_path.open("w") as capture_log, preview_path.open("w") as consumer_log:
        def start_camera():
            return subprocess.Popen([str(ROOT / "build/apps/robo_lab_main"), "config/bringup_d405.yaml"], cwd=ROOT,
                                    stdout=capture_log, stderr=subprocess.STDOUT)
        try:
            camera = start_camera()
            preview = subprocess.Popen([str(ROOT / "build/apps/rgbd_stereo_preview"), "--headless"], cwd=ROOT,
                                       stdout=consumer_log, stderr=subprocess.STDOUT)
            wait_for(lambda: "STREAM_LIVE" in preview_path.read_text(), 15)
            initial_stale = preview_path.read_text().count("STREAM_STALE")
            initial_live = preview_path.read_text().count("STREAM_LIVE")
            print("READY: unplug only the D405 for 5 seconds, then reconnect it.", flush=True)
            wait_for(lambda: not present(), args.wait_seconds)
            disconnected_at = time.monotonic()
            print("USB removal observed", flush=True)
            wait_for(lambda: preview_path.read_text().count("STREAM_STALE") > initial_stale, 5)
            result["stale_after_usb_removal_s"] = time.monotonic() - disconnected_at
            wait_for(lambda: "capture stopped:" in camera_path.read_text(), 8)
            result["capture_stop_after_usb_removal_s"] = time.monotonic() - disconnected_at
            stop(camera)
            wait_for(present, 90)
            camera = start_camera()
            wait_for(lambda: preview_path.read_text().count("STREAM_LIVE") > initial_live, 20)
            result["passed"] = True
            print("PASS: USB removal, stale indication, capture stop, reconnect and live restart", flush=True)
        except Exception as error:
            result["error"] = str(error)
            print("FAIL:", error, flush=True)
        finally:
            stop(camera)
            stop(preview)
            (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
