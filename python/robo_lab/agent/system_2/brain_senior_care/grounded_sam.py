"""
Grounded-SAM2 perception wrapper for image segmentation.

Provides image decoding utilities and a GroundedSamPerception class
that wraps GSamSession for multi-source image processing.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

from robo_lab.model.sam2 import GSamSession, SegmentResult


@dataclass
class PerceptionResult:
    """Result from perception pipeline."""
    segments: SegmentResult
    objects_of_interest: list[str]
    scene_description: str
    source_topic: str = ""
    timestamp: float = field(default_factory=time.time)


def decode_protobuf_image(payload: bytes, logger: Optional[Callable[[str], None]] = None) -> Optional[np.ndarray]:
    """
    Decode image from protobuf kinect.rgbImage message.
    Converts RGBA/BGRA to RGB/BGR if needed.
    """
    log = logger or (lambda x: None)
    
    try:
        from robo_lab.proto_gen.kinect_pb2 import rgbImage
    except ImportError as e:
        log(f"Failed to import kinect_pb2: {e}")
        return decode_raw_image(payload, logger)
    
    try:
        msg = rgbImage()
        msg.ParseFromString(payload)
        
        width = msg.width
        height = msg.height
        channels = msg.channels
        image_data = msg.image
        
        if width <= 0 or height <= 0 or channels <= 0:
            log(f"Invalid image dimensions: {width}x{height}x{channels}")
            return None
        
        expected_size = width * height * channels
        if len(image_data) != expected_size:
            log(f"Image data size mismatch: {len(image_data)} vs {expected_size}")
            return None
        
        image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, channels))
        
        if channels == 4:
            image = image[:, :, :3].copy()
        
        return image
        
    except Exception as e:
        log(f"Protobuf decode failed: {e}, trying raw decode")
        return decode_raw_image(payload, logger)


def decode_raw_image(payload: bytes, logger: Optional[Callable[[str], None]] = None) -> Optional[np.ndarray]:
    """
    Decode image from raw bytes payload.
    Supports JPEG/PNG and raw formats.
    """
    import cv2
    
    log = logger or (lambda x: None)
    
    if len(payload) < 12:
        log(f"Payload too small: {len(payload)} bytes")
        return None

    if payload[:2] == b'\xff\xd8':
        arr = np.frombuffer(payload, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    if payload[:8] == b'\x89PNG\r\n\x1a\n':
        arr = np.frombuffer(payload, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    width = int.from_bytes(payload[0:4], byteorder='little')
    height = int.from_bytes(payload[4:8], byteorder='little')
    channels = int.from_bytes(payload[8:12], byteorder='little')
    
    if 0 < width < 8192 and 0 < height < 8192 and channels in (1, 3, 4):
        expected_size = 12 + width * height * channels
        if len(payload) == expected_size:
            data = payload[12:]
            image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, channels))
            if channels == 4:
                image = image[:, :, :3].copy()
            return image
    
    common_sizes = [
        (1920, 1080, 3), (1920, 1080, 4),
        (1280, 720, 3), (1280, 720, 4),
        (640, 480, 3), (640, 480, 4),
    ]
    
    for w, h, c in common_sizes:
        if len(payload) == w * h * c:
            image = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, c))
            if c == 4:
                image = image[:, :, :3].copy()
            return image
    
    log(f"Unknown image format, payload size: {len(payload)}")
    return None


@dataclass
class ImageBuffer:
    """Thread-safe image buffer for a single topic."""
    topic: str
    image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update(self, image: np.ndarray) -> None:
        with self.lock:
            self.image = image
            self.timestamp = time.time()
    
    def get(self, clear: bool = True) -> tuple[Optional[np.ndarray], float]:
        with self.lock:
            img = self.image
            ts = self.timestamp
            if clear:
                self.image = None
            return img, ts


class GroundedSamPerception:
    """
    Multi-source perception using Grounded-SAM2.
    
    Example:
        perception = GroundedSamPerception(
            topics=["realsense/rgb", "kinect2/rgb"],
            detection_targets=["human", "face"],
        )
        perception.start_zenoh(zenoh_config)
        result = perception.process_once()
    """
    
    DEFAULT_TARGETS = ["human", "person"]
    
    def __init__(
        self,
        topics: Union[str, list[str]] = "kinect2/rgb",
        detection_targets: Optional[list[str]] = None,
        device: str = "cuda",
        lazy_load: bool = True,
        name: str = "GSam",
    ):
        self.name = name
        self.device = device
        self._gsam: Optional[GSamSession] = None
        
        if isinstance(topics, str):
            self._topics = [topics]
        else:
            self._topics = list(topics)
        
        self._detection_targets = detection_targets or self.DEFAULT_TARGETS.copy()
        
        self._buffers: dict[str, ImageBuffer] = {
            topic: ImageBuffer(topic=topic) for topic in self._topics
        }
        
        self._result_lock = threading.Lock()
        self._last_result: Optional[PerceptionResult] = None
        
        self._zenoh_session = None
        self._zenoh_subscribers: list = []
        self._on_result_callbacks: list[Callable[[PerceptionResult], None]] = []
        
        if not lazy_load:
            self._init_gsam()
    
    @property
    def topics(self) -> list[str]:
        return self._topics.copy()
    
    @property
    def detection_targets(self) -> list[str]:
        return self._detection_targets.copy()
    
    @detection_targets.setter
    def detection_targets(self, targets: list[str]) -> None:
        self._detection_targets = list(targets)
    
    @property
    def detection_prompt(self) -> str:
        return ". ".join(self._detection_targets) + "."
    
    @property
    def gsam(self) -> GSamSession:
        return self._init_gsam()
    
    @property
    def zenoh_session(self):
        """Get the Zenoh session (for sharing with other components)."""
        return self._zenoh_session
    
    def _init_gsam(self) -> GSamSession:
        if self._gsam is None:
            self._gsam = GSamSession(device=self.device)
        return self._gsam
    
    def _log(self, msg: str) -> None:
        print(f"[{self.name}] {msg}")
    
    def start_zenoh(self, zenoh_config: Optional[dict] = None) -> None:
        """Initialize Zenoh session and subscribe to all topics."""
        try:
            import zenoh
        except ImportError:
            self._log("WARNING: zenoh not installed")
            return
        
        zenoh_config = zenoh_config or {}
        
        zconf = zenoh.Config()
        if zenoh_config.get("mode"):
            zconf.insert_json5("mode", f'"{zenoh_config["mode"]}"')
        if zenoh_config.get("connect"):
            zconf.insert_json5("connect/endpoints", str(zenoh_config["connect"]))
        
        self._zenoh_session = zenoh.open(zconf)
        
        for topic in self._topics:
            handler = self._make_handler(topic)
            sub = self._zenoh_session.declare_subscriber(topic, handler)
            self._zenoh_subscribers.append(sub)
            self._log(f"Subscribed to: {topic}")
    
    def _make_handler(self, topic: str) -> Callable:
        def handler(sample):
            self._on_message(topic, sample)
        return handler
    
    def _on_message(self, topic: str, sample) -> None:
        try:
            payload = sample.payload
            if hasattr(payload, 'to_bytes'):
                payload = payload.to_bytes()
            elif not isinstance(payload, bytes):
                payload = bytes(payload)
            
            image = decode_protobuf_image(payload, self._log)
            
            if image is not None and topic in self._buffers:
                self._buffers[topic].update(image)
        except Exception as e:
            self._log(f"Error handling message from {topic}: {e}")
    
    def get_image(self, topic: Optional[str] = None, clear: bool = True) -> tuple[Optional[np.ndarray], str, float]:
        """Get latest image from a topic or any topic with data."""
        if topic:
            if topic in self._buffers:
                img, ts = self._buffers[topic].get(clear)
                return img, topic, ts
            return None, topic, 0.0
        
        best_img = None
        best_topic = ""
        best_ts = 0.0
        
        for t, buf in self._buffers.items():
            img, ts = buf.get(clear=False)
            if img is not None and ts > best_ts:
                best_img = img
                best_topic = t
                best_ts = ts
        
        if best_img is not None and clear:
            self._buffers[best_topic].get(clear=True)
        
        return best_img, best_topic, best_ts
    
    def set_image(self, image: np.ndarray, topic: Optional[str] = None) -> None:
        """Manually set an image for processing."""
        topic = topic or (self._topics[0] if self._topics else "manual")
        if topic not in self._buffers:
            self._buffers[topic] = ImageBuffer(topic=topic)
        self._buffers[topic].update(image)
    
    def process_once(self, topic: Optional[str] = None) -> Optional[PerceptionResult]:
        """Process one image from buffer."""
        image, src_topic, timestamp = self.get_image(topic, clear=True)
        
        if image is None:
            return None
        
        segments = self.gsam.segment(image, self.detection_prompt)
        
        result = PerceptionResult(
            segments=segments,
            objects_of_interest=list(segments.labels),
            scene_description=self._describe_scene(segments),
            source_topic=src_topic,
            timestamp=timestamp,
        )
        
        with self._result_lock:
            self._last_result = result
        
        for callback in self._on_result_callbacks:
            try:
                callback(result)
            except Exception as e:
                self._log(f"Callback error: {e}")
        
        return result
    
    def get_last_result(self) -> Optional[PerceptionResult]:
        with self._result_lock:
            return self._last_result
    
    def on_result(self, callback: Callable[[PerceptionResult], None]) -> None:
        self._on_result_callbacks.append(callback)
    
    def _describe_scene(self, segments: SegmentResult) -> str:
        if len(segments) == 0:
            return "No objects detected."
        
        counts: dict[str, int] = {}
        for label in segments.labels:
            counts[label] = counts.get(label, 0) + 1
        
        parts = [f"{count} {obj}" + ("s" if count > 1 else "") 
                 for obj, count in counts.items()]
        return f"Scene contains: {', '.join(parts)}."
    
    def visualize(
        self,
        image: np.ndarray,
        result: Union[PerceptionResult, SegmentResult],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        segments = result.segments if isinstance(result, PerceptionResult) else result
        return self.gsam.visualize(image, segments, output_path=output_path)
    
    def stop(self) -> None:
        for sub in self._zenoh_subscribers:
            try:
                sub.undeclare()
            except Exception:
                pass
        self._zenoh_subscribers.clear()
        
        if self._zenoh_session is not None:
            self._zenoh_session.close()
            self._zenoh_session = None
        
        self._gsam = None
