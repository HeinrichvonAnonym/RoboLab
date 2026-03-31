"""
System 2: Brain - High-level perception and decision-making.

Uses GroundedSamPerception for visual perception.
Uses PredefinedTargetsPublisher for target pose publishing.
Both run in step() independently - coupling logic is user-defined.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from robo_lab.agent.base import AgentBase
from robo_lab.model.sam2 import SegmentResult

from .grounded_sam import GroundedSamPerception, PerceptionResult
from .predefined_targets_publisher import PredefinedTargetsPublisher


class BrainSeniorCare(AgentBase):
    """
    High-level perception and decision-making for senior care robot.
    
    Components:
        - GroundedSamPerception: Visual perception (GSam inference)
        - PredefinedTargetsPublisher: Pose publishing (when enabled)
    
    Config options (from YAML):
        rgb_topic: Single topic or list of topics
        detection_targets: Objects to detect
        optinal_hz: Update rate in Hz
        use_predefined_targets: Enable target publisher
        target_topic: Zenoh topic for target poses
        predefined_targets: List of target poses
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        device: str = "cuda",
        lazy_load: bool = True,
    ):
        rate = 10.0
        if config:
            rate = config.get("optinal_hz", config.get("rate_hz", rate))
        super().__init__(config=config, name="Brain", rate_hz=rate)
        
        # Perception (GSam)
        rgb_topic = self.config.get("rgb_topic", "kinect2/rgb")
        detection_targets = self.config.get("detection_targets", ["human", "person"])
        
        self._perception = GroundedSamPerception(
            topics=rgb_topic,
            detection_targets=detection_targets,
            device=device,
            lazy_load=lazy_load,
            name=self.name,
        )
        
        # Target Publisher (optional)
        self._use_predefined_targets = self.config.get("use_predefined_targets", False)
        self._target_publisher: Optional[PredefinedTargetsPublisher] = None
        
        if self._use_predefined_targets:
            target_topic = self.config.get("target_topic", "franka/target_pose")
            predefined_targets = self.config.get("predefined_targets", [])
            
            self._target_publisher = PredefinedTargetsPublisher(
                targets=predefined_targets,
                topic=target_topic,
                name=f"{self.name}:TargetPub",
            )
        
        # Debug
        self._debug_output = self.config.get("debug_output", True)
        self._debug_dir = Path("/tmp/brain_debug")

    @property
    def perception(self) -> GroundedSamPerception:
        """Get perception engine."""
        return self._perception
    
    @property
    def target_publisher(self) -> Optional[PredefinedTargetsPublisher]:
        """Get target publisher (None if disabled)."""
        return self._target_publisher

    @property
    def detection_targets(self) -> list[str]:
        return self._perception.detection_targets

    @detection_targets.setter
    def detection_targets(self, targets: list[str]) -> None:
        self._perception.detection_targets = targets
        print(f"[{self.name}] Detection targets updated: {targets}")

    @property
    def topics(self) -> list[str]:
        return self._perception.topics

    def setup(self) -> None:
        """Initialize Zenoh for perception and target publisher."""
        zenoh_config = self.config.get("zenoh", {})
        
        # Start perception
        self._perception.start_zenoh(zenoh_config)
        
        # Start target publisher with shared session
        if self._target_publisher is not None:
            session = self._perception.zenoh_session
            if session is not None:
                self._target_publisher.use_session(session)
            else:
                self._target_publisher.start_zenoh(zenoh_config)
        
        if self._debug_output:
            self._debug_dir.mkdir(exist_ok=True)
        
        print(f"[{self.name}] Topics: {self._perception.topics}")
        print(f"[{self.name}] Detection targets: {self._perception.detection_targets}")
        if self._target_publisher:
            print(f"[{self.name}] Target publisher: {self._target_publisher.topic} ({self._target_publisher.num_targets} targets)")

    def step(self) -> None:
        """
        One cycle - perception and target publishing run in parallel.
        Coupling logic (e.g., publish target on detection) is user-defined.
        """
        # --- Perception ---
        image, topic, timestamp = self._perception.get_image(clear=True)
        
        if image is not None:
            if self._debug_output:
                cv2.imwrite(str(self._debug_dir / "input_latest.jpg"), image)
            
            prompt = self._perception.detection_prompt
            print(f"[{self.name}] Inference on {topic}, shape={image.shape}")
            
            t_start = time.time()
            result = self._perception.gsam.segment(image, prompt)
            inference_ms = (time.time() - t_start) * 1000
            
            print(f"[{self.name}] Done in {inference_ms:.0f}ms, found {len(result)} objects")
            
            if len(result) > 0:
                if self._debug_output:
                    self._perception.visualize(image, result, str(self._debug_dir / "output_latest.jpg"))
                print(f"[{self.name}] Detected: {list(result.labels)}")
        
        # --- Target Publisher (independent, user controls logic) ---
        # Example: publish current target each step (user can override)
        if self._target_publisher is not None:
            print(f"[{self.name}] Publishing current target")
            self._target_publisher.publish_current()

    def cleanup(self) -> None:
        """Release resources."""
        self._perception.stop()
        if self._target_publisher is not None:
            self._target_publisher.stop()

    def perceive(
        self,
        image: Union[np.ndarray, str],
        prompt: Optional[str] = None,
    ) -> PerceptionResult:
        """Run perception on an image."""
        if isinstance(image, str):
            image = cv2.imread(image)
        
        prompt = prompt or self._perception.detection_prompt
        segments = self._perception.gsam.segment(image, prompt)
        
        return PerceptionResult(
            segments=segments,
            objects_of_interest=list(segments.labels),
            scene_description=self._describe_scene(segments),
        )

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
        return self._perception.visualize(image, result, output_path)

    def on_perception(self, callback) -> None:
        self._perception.on_result(callback)

    def set_image(self, image: np.ndarray, topic: Optional[str] = None) -> None:
        self._perception.set_image(image, topic)

    def get_last_perception(self) -> Optional[PerceptionResult]:
        return self._perception.get_last_result()
