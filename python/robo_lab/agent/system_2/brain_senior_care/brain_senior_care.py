"""
System 2: Brain - High-level perception and decision-making.

Uses Grounded-SAM2 for visual perception and object segmentation.
Subscribes to RGB images via Zenoh and runs inference.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np

from robo_lab.agent.base import AgentBase
from robo_lab.model.sam2 import GSamSession, SegmentResult

import cv2


@dataclass
class PerceptionResult:
    """Result from Brain perception pipeline."""
    segments: SegmentResult
    objects_of_interest: list[str]
    scene_description: str
    timestamp: float = field(default_factory=time.time)


class BrainSeniorCare(AgentBase):
    """
    High-level perception and decision-making for senior care robot.
    
    Subscribes to RGB images via Zenoh, runs Grounded-SAM2 inference
    to detect configured targets (e.g., human, face).
    
    Config options (from YAML):
        rgb_topic: Zenoh topic for RGB images (default: "kinec2/rgb")
        detection_targets: List of objects to detect (default: ["human"])
        optinal_hz: Update rate in Hz
    
    Example:
        brain = BrainSeniorCare(config={
            "rgb_topic": "kinec2/rgb",
            "detection_targets": ["human", "face"],
        })
        brain.start(threaded=True)
    """

    DEFAULT_DETECTION_TARGETS = ["human", "person"]

    def __init__(
        self,
        config: Optional[dict] = None,
        device: str = "cuda",
        lazy_load: bool = True,
    ):
        """
        Initialize BrainSeniorCare.
        
        Args:
            config: Configuration dict from YAML.
            device: "cuda" or "cpu".
            lazy_load: If True, delay GSamSession loading until first use.
        """
        rate = 10.0
        if config:
            rate = config.get("optinal_hz", config.get("rate_hz", rate))
        super().__init__(config=config, name="Brain", rate_hz=rate)
        
        self.device = device
        self._gsam: Optional[GSamSession] = None
        
        # Zenoh subscription config
        self._rgb_topic = self.config.get("rgb_topic", "kinec2/rgb")
        self._detection_targets: list[str] = self.config.get(
            "detection_targets", self.DEFAULT_DETECTION_TARGETS
        )
        
        # Thread-safe image buffer
        self._image_lock = threading.Lock()
        self._current_image: Optional[np.ndarray] = None
        self._image_timestamp: float = 0.0
        
        # Latest perception result
        self._result_lock = threading.Lock()
        self._last_perception: Optional[PerceptionResult] = None
        
        # Zenoh session
        self._zenoh_session = None
        self._zenoh_subscriber = None
        
        # Callbacks for perception results
        self._on_perception_callbacks: list[Callable[[PerceptionResult], None]] = []
        
        if not lazy_load:
            self._init_gsam()

    @property
    def detection_targets(self) -> list[str]:
        """Get current detection targets."""
        return self._detection_targets.copy()

    @detection_targets.setter
    def detection_targets(self, targets: list[str]) -> None:
        """Set detection targets dynamically."""
        self._detection_targets = list(targets)
        print(f"[{self.name}] Detection targets updated: {self._detection_targets}")

    @property
    def detection_prompt(self) -> str:
        """Build prompt string from detection targets."""
        return ". ".join(self._detection_targets) + "."

    def setup(self) -> None:
        """Initialize Zenoh subscription and perception resources."""
        self._init_zenoh()
        print(f"[{self.name}] Subscribed to RGB topic: {self._rgb_topic}")
        print(f"[{self.name}] Detection targets: {self._detection_targets}")

    def _init_zenoh(self) -> None:
        """Initialize Zenoh session and subscribe to RGB topic."""
        try:
            import zenoh
            self._zenoh_module = zenoh
        except ImportError:
            print(f"[{self.name}] WARNING: zenoh not installed, skipping subscription")
            return

        # Get zenoh config from parent config if available
        zenoh_config = self.config.get("zenoh", {})
        
        # Open zenoh session
        zconf = zenoh.Config()
        if zenoh_config.get("mode"):
            zconf.insert_json5("mode", f'"{zenoh_config["mode"]}"')
        if zenoh_config.get("connect"):
            zconf.insert_json5("connect/endpoints", str(zenoh_config["connect"]))
        
        self._zenoh_session = zenoh.open(zconf)
        
        # Subscribe to RGB topic with handler
        # Use a closure to ensure self is captured properly
        def rgb_handler(sample):
            self._on_rgb_message(sample)
        
        self._rgb_handler = rgb_handler  # Keep reference alive
        self._zenoh_subscriber = self._zenoh_session.declare_subscriber(
            self._rgb_topic,
            self._rgb_handler,
        )
        print(f"[{self.name}] Zenoh session opened, subscriber declared")

    def _on_rgb_message(self, sample) -> None:
        """
        Callback for incoming RGB messages.
        
        Decodes image from zenoh payload (protobuf kinect.rgbImage) and stores for processing.
        """
        try:
            # Handle different zenoh API versions for payload access
            payload = sample.payload
            if hasattr(payload, 'to_bytes'):
                payload = payload.to_bytes()
            elif not isinstance(payload, bytes):
                payload = bytes(payload)
            
            # 
            image = self._decode_protobuf_image(payload)
            
            if image is not None:
                # print(f"[{self.name}] Image received: {image.shape}")
                with self._image_lock:
                    self._current_image = image
                    self._image_timestamp = time.time()
            else:
                print(f"[{self.name}] Failed to decode image")
        except Exception as e:
            import traceback
            print(f"[{self.name}] Error in RGB callback: {e}")
            traceback.print_exc()

    def _decode_protobuf_image(self, payload: bytes) -> Optional[np.ndarray]:
        """
        Decode image from protobuf kinect.rgbImage message.
        Converts RGBA/BGRA to RGB/BGR if needed.
        """
        try:
            from robo_lab.proto_gen.kinect_pb2 import rgbImage
        except ImportError as e:
            print(f"[{self.name}] Failed to import kinect_pb2: {e}")
            return self._decode_image(payload)
        
        try:
            msg = rgbImage()
            msg.ParseFromString(payload)
            
            width = msg.width
            height = msg.height
            channels = msg.channels
            image_data = msg.image
            
            if width <= 0 or height <= 0 or channels <= 0:
                print(f"[{self.name}] Invalid image dimensions: {width}x{height}x{channels}")
                return None
            
            expected_size = width * height * channels
            if len(image_data) != expected_size:
                print(f"[{self.name}] Image data size mismatch: {len(image_data)} vs {expected_size}")
                return None
            
            image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, channels))
            
            # Convert 4-channel (RGBA/BGRA) to 3-channel (RGB/BGR)
            if channels == 4:
                image = image[:, :, :3].copy()
            
            return image
            
        except Exception as e:
            print(f"[{self.name}] Protobuf decode failed: {e}, trying raw decode")
            return self._decode_image(payload)

    def _decode_image(self, payload: bytes) -> Optional[np.ndarray]:
        """
        Decode image from bytes payload.
        
        Supports:
        - Raw RGB/BGR data with header (width, height, channels as first 12 bytes)
        - JPEG/PNG encoded data
        - Common raw resolutions without header
        """
        if len(payload) < 12:
            print(f"[{self.name}] Payload too small: {len(payload)} bytes")
            return None

        # Try to detect format
        # Check for JPEG magic bytes
        if payload[:2] == b'\xff\xd8':
            import cv2
            arr = np.frombuffer(payload, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                print(f"[{self.name}] Decoded JPEG image")
            return img
        
        # Check for PNG magic bytes
        if payload[:8] == b'\x89PNG\r\n\x1a\n':
            import cv2
            arr = np.frombuffer(payload, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                print(f"[{self.name}] Decoded PNG image")
            return img
        
        # Try raw format with header: [width(4), height(4), channels(4), data...]
        width = int.from_bytes(payload[0:4], byteorder='little')
        height = int.from_bytes(payload[4:8], byteorder='little')
        channels = int.from_bytes(payload[8:12], byteorder='little')
        
        # Sanity check for header values
        if 0 < width < 8192 and 0 < height < 8192 and channels in (1, 3, 4):
            expected_size = 12 + width * height * channels
            if len(payload) == expected_size:
                data = payload[12:]
                print(f"[{self.name}] Decoded raw image with header: {width}x{height}x{channels}")
                image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, channels))
                if channels == 4:
                    image = image[:, :, :3].copy()
                return image
        
        # Try common raw resolutions without header
        common_sizes = [
            (1920, 1080, 3),  # 1080p RGB
            (1920, 1080, 4),  # 1080p RGBA
            (1280, 720, 3),   # 720p RGB
            (1280, 720, 4),   # 720p RGBA
            (640, 480, 3),    # VGA RGB
            (640, 480, 4),    # VGA RGBA
            (512, 424, 3),    # Kinect depth-like
        ]
        
        for w, h, c in common_sizes:
            if len(payload) == w * h * c:
                print(f"[{self.name}] Decoded raw image (guessed): {w}x{h}x{c}")
                image = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, c))
                if c == 4:
                    image = image[:, :, :3].copy()
                return image
        
        print(f"[{self.name}] Unknown image format, payload size: {len(payload)}, first bytes: {payload[:16].hex()}")
        return None

    def step(self) -> None:
        """
        One perception cycle.
        
        If a new image is available, run inference.
        """
        import cv2
        from pathlib import Path
        
        image = None
        timestamp = 0.0
        
        with self._image_lock:
            if self._current_image is not None:
                image = self._current_image.copy()
                timestamp = self._image_timestamp
                self._current_image = None
        
        if image is None:
            return
        
        # Debug: save input image
        debug_dir = Path("/tmp/brain_debug")
        debug_dir.mkdir(exist_ok=True)
        input_path = debug_dir / "input_latest.jpg"
        cv2.imwrite(str(input_path), image)
        
        print(f"[{self.name}] Running inference on image {image.shape}, prompt: '{self.detection_prompt}'")
        print(f"[{self.name}] Input saved to: {input_path}")
        
        t_start = time.time()
        result = self.perceive(image, prompt=self.detection_prompt)
        result.timestamp = timestamp
        inference_time = (time.time() - t_start) * 1000
        
        print(f"[{self.name}] Inference done in {inference_time:.0f}ms, found {len(result.segments)} objects")
        
        # Debug: save annotated output
        if len(result.segments) > 0:
            output_path = debug_dir / "output_latest.jpg"
            self.visualize(image, result, output_path=str(output_path))
            print(f"[{self.name}] Detected: {result.objects_of_interest}")
            print(f"[{self.name}] Scores: {result.segments.scores.tolist()}")
            print(f"[{self.name}] Output saved to: {output_path}")
        else:
            print(f"[{self.name}] No detections. Check: /tmp/brain_debug/input_latest.jpg")
        
        with self._result_lock:
            self._last_perception = result
        
        # Notify callbacks
        for callback in self._on_perception_callbacks:
            try:
                callback(result)
            except Exception as e:
                print(f"[{self.name}] Callback error: {e}")

    def cleanup(self) -> None:
        """Release Zenoh and perception resources."""
        if self._zenoh_subscriber is not None:
            self._zenoh_subscriber.undeclare()
            self._zenoh_subscriber = None
        
        if self._zenoh_session is not None:
            self._zenoh_session.close()
            self._zenoh_session = None
        
        self._gsam = None

    def on_perception(self, callback: Callable[[PerceptionResult], None]) -> None:
        """
        Register callback for perception results.
        
        Args:
            callback: Function called with PerceptionResult after each inference.
        """
        self._on_perception_callbacks.append(callback)

    def set_image(self, image: np.ndarray) -> None:
        """Set image for next perception cycle (non-blocking, manual mode)."""
        with self._image_lock:
            self._current_image = image
            self._image_timestamp = time.time()

    def get_last_perception(self) -> Optional[PerceptionResult]:
        """Get result from last perception cycle."""
        with self._result_lock:
            return self._last_perception

    def _init_gsam(self) -> GSamSession:
        """Initialize GSamSession on first use."""
        if self._gsam is None:
            self._gsam = GSamSession(device=self.device)
        return self._gsam

    @property
    def gsam(self) -> GSamSession:
        """Get GSamSession instance (lazy-loaded)."""
        return self._init_gsam()

    def perceive(
        self,
        image: Union[np.ndarray, str],
        prompt: Optional[str] = None,
        use_defaults: bool = False,
    ) -> PerceptionResult:
        """
        Perceive the scene and segment objects of interest.
        
        Args:
            image: Input image (numpy array or path).
            prompt: Text prompt for objects to detect (e.g., "person. cup.").
                    If None and use_defaults=True, uses detection_targets.
            use_defaults: If True and no prompt given, use detection_targets.
        
        Returns:
            PerceptionResult with segmentation and scene analysis.
        """
        if prompt is None:
            if use_defaults:
                prompt = self.detection_prompt
            else:
                raise ValueError("Either prompt or use_defaults=True required")

        segments = self.gsam.segment(image, prompt)
        
        return PerceptionResult(
            segments=segments,
            objects_of_interest=list(segments.labels),
            scene_description=self._describe_scene(segments),
        )

    def _describe_scene(self, segments: SegmentResult) -> str:
        """Generate a simple scene description from segments."""
        if len(segments) == 0:
            return "No objects detected in scene."
        
        counts: dict[str, int] = {}
        for label in segments.labels:
            counts[label] = counts.get(label, 0) + 1
        
        parts = [f"{count} {obj}" + ("s" if count > 1 else "") 
                 for obj, count in counts.items()]
        return f"Scene contains: {', '.join(parts)}."

    def detect_targets(self, image: Union[np.ndarray, str]) -> SegmentResult:
        """Run detection with current targets."""
        return self.gsam.segment(image, self.detection_prompt)

    def visualize(
        self,
        image: np.ndarray,
        result: Union[PerceptionResult, SegmentResult],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Visualize perception results on image."""
        segments = result.segments if isinstance(result, PerceptionResult) else result
        return self.gsam.visualize(image, segments, output_path=output_path)
