#!/usr/bin/env python3
"""Track hands from Zenoh RGB frames, with GUI calls on the Python main thread.

Dependencies (install in the interpreter used to launch this script):
    python -m pip install 'mediapipe==1.1.0' 'eclipse-zenoh>=1' PyYAML numpy
    python -m pip install 'protobuf==3.20.3'
MediaPipe supplies OpenCV. Keep protobuf 3.20.3 for this repo's generated messages;
no protocol regeneration or dependency installation is performed by this script.

Run from any directory:
    python /path/to/roboLab/src/plugins/python/hand_tracker_plugin.py
    python /path/to/roboLab/src/plugins/python/hand_tracker_plugin.py --config /path/to/config.yaml

Results are available as ``plugin.latest_result``; this program publishes no messages.
Image landmarks have normalized x/y and wrist-relative z. World landmarks are
model estimates in meters relative to the hand center, not camera-frame positions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
from typing import Any
import urllib.request

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config/plugins/python/hand_tracker_plugin_config.yaml"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
WINDOW_NAME = "Hand tracker"
LOGGER = logging.getLogger("hand_tracker")
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass(frozen=True)
class TrackerConfig:
    rgb_topic: str = "kinec2/rgb"
    input_color_format: str = "BGRX"
    enable_viewer: bool = True
    viewer_max_width: int = 1280
    fps: float = 20.0
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_path: Path = Path("~/.cache/robolab/mediapipe/hand_landmarker.task")
    auto_download_model: bool = True
    connect: tuple[str, ...] = ()


def load_config(path: Path | str = DEFAULT_CONFIG) -> TrackerConfig:
    path = Path(path).expanduser().resolve()
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    defaults = TrackerConfig()
    topic = root.get("topic", {})
    zenoh = root.get("zenoh", {})
    if not isinstance(topic, dict) or not isinstance(zenoh, dict):
        raise ValueError("topic and zenoh must be YAML mappings")
    values = {name: root.get(name, getattr(defaults, name)) for name in (
        "input_color_format", "enable_viewer", "viewer_max_width", "fps", "num_hands",
        "min_hand_detection_confidence", "min_hand_presence_confidence",
        "min_tracking_confidence", "model_path", "auto_download_model",
    )}
    values["rgb_topic"] = topic.get("rgb", defaults.rgb_topic)
    if not isinstance(values["rgb_topic"], str) or not values["rgb_topic"].strip():
        raise ValueError("topic.rgb must be a nonempty string")
    color = values["input_color_format"]
    if not isinstance(color, str) or color.upper() not in {"BGRX", "RGBX", "BGR", "RGB"}:
        raise ValueError("input_color_format must be BGRX, RGBX, BGR or RGB")
    values["input_color_format"] = color.upper()
    for name in ("enable_viewer", "auto_download_model"):
        if not isinstance(values[name], bool):
            raise ValueError(f"{name} must be a YAML boolean")
    for name in ("num_hands", "viewer_max_width"):
        if type(values[name]) is not int or values[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("fps", "min_hand_detection_confidence", "min_hand_presence_confidence",
                 "min_tracking_confidence"):
        if isinstance(values[name], bool):
            raise ValueError(f"{name} must be a number")
        values[name] = float(values[name])
        if not math.isfinite(values[name]):
            raise ValueError(f"{name} must be finite")
        if name == "fps" and values[name] <= 0:
            raise ValueError("fps must be positive")
        if name != "fps" and not 0 <= values[name] <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    model_path = Path(values["model_path"]).expanduser()
    # Custom relative model paths are relative to the YAML, not the launch directory.
    values["model_path"] = (model_path if model_path.is_absolute() else path.parent / model_path).resolve()
    connect = zenoh.get("connect", [])
    if not isinstance(connect, list) or any(not isinstance(item, str) or not item for item in connect):
        raise ValueError("zenoh.connect must be a list of nonempty endpoint strings")
    values["connect"] = tuple(connect)
    return TrackerConfig(**values)


def image_message_type():
    # Reuse the repo's protocol without invoking protoc or altering generated files.
    python_root = str(REPO_ROOT / "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    try:
        from robo_lab.proto_gen.kinect_pb2 import rgbImage
    except (ImportError, TypeError) as exc:
        raise RuntimeError("Cannot load kinect.rgbImage. Use protobuf==3.20.3 for this repo.") from exc
    return rgbImage


def decode_rgb(payload: bytes, color_format: str, message_type=None) -> np.ndarray:
    """Decode packed or padded uint8 rows into a contiguous RGB image."""
    msg = (message_type or image_message_type())()
    msg.ParseFromString(payload)
    width, height, channels, step = msg.width, msg.height, msg.channels, msg.step
    if color_format not in {"RGB", "BGR", "RGBX", "BGRX"}:
        raise ValueError(f"Unsupported color format: {color_format}")
    expected_channels = len(color_format)
    if width <= 0 or height <= 0 or channels != expected_channels:
        raise ValueError(f"Invalid image dimensions/channels: {width}x{height}x{channels}")
    packed_row = width * channels
    if step < packed_row or len(msg.image) != height * step:
        raise ValueError(f"Invalid image stride/length: step={step}, bytes={len(msg.image)}")
    rows = np.frombuffer(msg.image, dtype=np.uint8).reshape(height, step)
    pixels = rows[:, :packed_row].reshape(height, width, channels)
    rgb = pixels[:, :, 2::-1] if color_format.startswith("BGR") else pixels[:, :, :3]
    return np.ascontiguousarray(rgb)


def ensure_model(config: TrackerConfig, stop_event: threading.Event) -> Path:
    if stop_event.is_set():
        raise InterruptedError("Model loading stopped")
    path = config.model_path.expanduser()
    if path.is_file() and path.stat().st_size > 0:
        return path
    if not config.auto_download_model:
        raise FileNotFoundError(f"Hand model not found: {path}. Download {MODEL_URL} to this path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading hand model to %s", path)
    partial = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".part",
                                         delete=False) as output:
            partial = Path(output.name)
            with urllib.request.urlopen(MODEL_URL, timeout=5) as response:
                content_length = getattr(response, "headers", {}).get("Content-Length")
                while not stop_event.is_set():
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if stop_event.is_set():
                raise InterruptedError("Model download stopped")
            if content_length is not None and output.tell() != int(content_length):
                raise RuntimeError("Model download was incomplete")
        if partial.stat().st_size == 0:
            raise RuntimeError("Model download was empty")
        partial.replace(path)
        return path
    except InterruptedError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Cannot download hand model to {path}: {exc}. "
                           f"Download {MODEL_URL} manually or configure model_path.") from exc
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


@dataclass(frozen=True)
class HandFeatures:
    handedness: str
    score: float
    landmarks: tuple[tuple[float, float, float], ...]
    world_landmarks: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class TrackerResult:
    timestamp_ms: int
    received_monotonic: float
    rgb: np.ndarray
    hands: tuple[HandFeatures, ...]


def extract_hands(result: Any) -> tuple[HandFeatures, ...]:
    hands = []
    for index, landmarks in enumerate(result.hand_landmarks):
        world = result.hand_world_landmarks[index]
        if len(landmarks) != 21 or len(world) != 21:
            raise ValueError("Hand Landmarker returned an incomplete 21-point hand")
        category = max(result.handedness[index], key=lambda item: item.score)
        hands.append(HandFeatures(
            handedness=category.category_name, score=float(category.score),
            landmarks=tuple((float(p.x), float(p.y), float(p.z)) for p in landmarks),
            world_landmarks=tuple((float(p.x), float(p.y), float(p.z)) for p in world),
        ))
    return tuple(hands)


class MediaPipeDetector:
    def __init__(self, config: TrackerConfig, stop_event: threading.Event):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:
            raise RuntimeError("MediaPipe dependencies are missing. Install with: "
                               "python -m pip install 'mediapipe==1.1.0' "
                               f"(import error: {exc})") from exc
        self.mp = mp
        model = ensure_model(config, stop_event)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.num_hands,
            min_hand_detection_confidence=config.min_hand_detection_confidence,
            min_hand_presence_confidence=config.min_hand_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> tuple[HandFeatures, ...]:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        return extract_hands(self.landmarker.detect_for_video(image, timestamp_ms))

    def close(self) -> None:
        self.landmarker.close()


def render_result(result: TrackerResult, cv2, max_width: int) -> np.ndarray:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Hand visualization must run on the Python main thread")
    image = cv2.cvtColor(result.rgb, cv2.COLOR_RGB2BGR)
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, (max_width, max(1, round(image.shape[0] * scale))))
    height, width = image.shape[:2]
    for hand in result.hands:
        color = (80, 220, 80) if hand.handedness == "Left" else (255, 160, 80)
        points = [(round(x * (width - 1)), round(y * (height - 1))) for x, y, _ in hand.landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(image, points[start], points[end], color, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(image, point, 3, (30, 30, 255), -1, cv2.LINE_AA)
        x, y = points[0]
        cv2.putText(image, f"{hand.handedness} {hand.score:.2f}",
                    (max(0, x), max(20, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    age = max(0.0, time.monotonic() - result.received_monotonic)
    label = f"Hands: {len(result.hands)} | frame age: {age:.1f}s | Q / Esc to quit"
    cv2.putText(image, label, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image


class HandTrackerPlugin:
    """Standalone subscriber. Call run() on the main thread when visualization is enabled."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._pending: tuple[bytes, int, float] | None = None
        self._latest_result: TrackerResult | None = None
        self._worker_error: Exception | None = None
        self._worker: threading.Thread | None = None
        self.received_frames = 0
        self.processed_frames = 0
        self.replaced_frames = 0
        self.invalid_frames = 0

    @property
    def latest_result(self) -> TrackerResult | None:
        """Latest result and matching read-only RGB image; an empty hands tuple clears old detections."""
        with self._condition:
            return self._latest_result

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

    def _on_frame(self, sample) -> None:
        received_ns = time.monotonic_ns()
        # Decode and inference never run on Zenoh's callback threads.
        payload = sample.payload.to_bytes()
        with self._condition:
            if self._stop_event.is_set():
                return
            self.received_frames += 1
            if self._pending is not None:
                self.replaced_frames += 1
            self._pending = (payload, received_ns // 1_000_000, received_ns / 1e9)
            self._condition.notify_all()

    def _process_frames(self) -> None:
        detector = None
        try:
            message_type = image_message_type()
            detector = MediaPipeDetector(self.config, self._stop_event)
            LOGGER.info("CPU Hand Landmarker ready; max %d hands", self.config.num_hands)
            last_timestamp = -1
            next_frame_at = 0.0
            last_invalid_log = -math.inf
            while not self._stop_event.is_set():
                with self._condition:
                    self._condition.wait_for(lambda: self._pending is not None or self._stop_event.is_set())
                    while not self._stop_event.is_set() and time.monotonic() < next_frame_at:
                        self._condition.wait(timeout=next_frame_at - time.monotonic())
                    if self._stop_event.is_set():
                        break
                    payload, received_ms, received_at = self._pending
                    self._pending = None
                next_frame_at = time.monotonic() + 1.0 / self.config.fps
                try:
                    rgb = decode_rgb(payload, self.config.input_color_format, message_type)
                except Exception as exc:
                    self.invalid_frames += 1
                    if time.monotonic() - last_invalid_log >= 2:
                        LOGGER.warning("Skipping invalid RGB message: %s", exc)
                        last_invalid_log = time.monotonic()
                    continue
                last_timestamp = max(received_ms, last_timestamp + 1)
                hands = detector.process(rgb, last_timestamp)
                rgb.setflags(write=False)
                result = TrackerResult(last_timestamp, received_at, rgb, hands)
                with self._condition:
                    self._latest_result = result
                    self.processed_frames += 1
        except Exception as exc:
            if not (isinstance(exc, InterruptedError) and self._stop_event.is_set()):
                self._worker_error = exc
            self.stop()
        finally:
            if detector is not None:
                try:
                    detector.close()
                except Exception as exc:
                    self._worker_error = self._worker_error or exc

    def _main_loop(self, cv2=None) -> None:
        last_report = 0.0
        try:
            if cv2 is not None:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            while not self._stop_event.is_set():
                result = self.latest_result
                if cv2 is not None:
                    if result is None:
                        display = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(display, "Waiting for " + self.config.rgb_topic, (15, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    else:
                        display = render_result(result, cv2, self.config.viewer_max_width)
                    cv2.imshow(WINDOW_NAME, display)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        self.stop()
                    try:
                        visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                    except cv2.error:
                        # Some OpenCV backends raise once the user closes the window.
                        visible = -1
                    if visible < 1:
                        self.stop()
                elif time.monotonic() - last_report >= 2:
                    LOGGER.info("received=%d processed=%d hands=%d replaced=%d invalid=%d",
                                self.received_frames, self.processed_frames,
                                len(result.hands) if result else 0, self.replaced_frames, self.invalid_frames)
                    last_report = time.monotonic()
                self._stop_event.wait(1 / 30 if cv2 is not None else 0.1)
        finally:
            if cv2 is not None:
                cv2.destroyAllWindows()

    def run(self) -> None:
        if self._worker is not None:
            raise RuntimeError("Create a new HandTrackerPlugin for each run")
        cv2 = None
        if self.config.enable_viewer:
            if threading.current_thread() is not threading.main_thread():
                raise RuntimeError("enable_viewer requires run() on the Python main thread")
            if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                raise RuntimeError("No graphical display found. Set enable_viewer: false for headless tracking.")
            import cv2
        try:
            import zenoh
        except ImportError as exc:
            raise RuntimeError("Install eclipse-zenoh: python -m pip install 'eclipse-zenoh>=1'") from exc
        config = zenoh.Config()
        if self.config.connect:
            config.insert_json5("connect/endpoints", json.dumps(self.config.connect))
        with zenoh.open(config) as session:
            subscriber = session.declare_subscriber(self.config.rgb_topic, self._on_frame)
            self._worker = threading.Thread(target=self._process_frames, name="hand-tracker-inference")
            LOGGER.info("Subscribed to %s (pixel format %s)", self.config.rgb_topic, self.config.input_color_format)
            try:
                self._worker.start()
                self._main_loop(cv2)
            finally:
                self.stop()
                try:
                    subscriber.undeclare()
                finally:
                    if self._worker.ident is not None:
                        self._worker.join()
                    with self._condition:
                        self._pending = None
        if self._worker_error is not None:
            raise RuntimeError(f"Hand tracking failed: {self._worker_error}") from self._worker_error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Hand tracker YAML configuration")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    previous_handlers = {}
    try:
        plugin = HandTrackerPlugin(load_config(args.config))
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, lambda _sig, _frame: plugin.stop())
        plugin.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
