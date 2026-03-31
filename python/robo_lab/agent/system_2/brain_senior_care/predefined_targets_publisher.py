"""
Predefined Targets Publisher for robot pose commands.

Loads predefined target poses from config and publishes them 
via Zenoh as franka.StampedPose messages.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class TargetPose:
    """A target pose with position and quaternion orientation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    
    @classmethod
    def from_dict(cls, d: dict) -> "TargetPose":
        """Create from dictionary (YAML config format)."""
        return cls(
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            z=float(d.get("z", 0.0)),
            qw=float(d.get("qw", 1.0)),
            qx=float(d.get("qx", 0.0)),
            qy=float(d.get("qy", 0.0)),
            qz=float(d.get("qz", 0.0)),
        )
    
    def to_proto(self) -> bytes:
        """Serialize to franka.StampedPose protobuf bytes."""
        try:
            from robo_lab.proto_gen.franka_pb2 import StampedPose, Pose, Point, Quatenion
        except ImportError:
            raise ImportError("franka_pb2 not found. Run: protoc --python_out=python/robo_lab/proto_gen proto/franka.proto")
        
        msg = StampedPose()
        msg.pose.pos.x = self.x
        msg.pose.pos.y = self.y
        msg.pose.pos.z = self.z
        msg.pose.rot.w = self.qw
        msg.pose.rot.x = self.qx
        msg.pose.rot.y = self.qy
        msg.pose.rot.z = self.qz
        msg.sys_time = time.time()
        
        return msg.SerializeToString()
    
    def __repr__(self) -> str:
        return f"TargetPose(pos=({self.x:.3f}, {self.y:.3f}, {self.z:.3f}), quat=({self.qw:.3f}, {self.qx:.3f}, {self.qy:.3f}, {self.qz:.3f}))"


class PredefinedTargetsPublisher:
    """
    Publishes predefined target poses via Zenoh.
    
    Loads targets from config and publishes them on demand or cyclically.
    
    Config format (in senior_care.yaml):
        target_topic: "franka/target_pose"
        predefined_targets:
          - x: 0.5
            y: 0.0
            z: 0.3
            qw: 1.0
            qx: 0.0
            qy: 0.0
            qz: 0.0
          - x: 0.6
            y: 0.1
            z: 0.4
            ...
    
    Example:
        publisher = PredefinedTargetsPublisher(
            targets=[{"x": 0.5, "y": 0, "z": 0.3, "qw": 1}],
            topic="franka/target_pose",
        )
        publisher.start_zenoh(zenoh_config)
        publisher.publish_current()
        publisher.next()
        publisher.publish_current()
    """
    
    def __init__(
        self,
        targets: Optional[list[dict]] = None,
        topic: str = "franka/target_pose",
        name: str = "TargetPublisher",
    ):
        """
        Initialize publisher.
        
        Args:
            targets: List of target pose dicts from config.
            topic: Zenoh topic to publish to.
            name: Name for logging.
        """
        self.name = name
        self._topic = topic
        self._targets: list[TargetPose] = []
        self._current_index = 0
        
        if targets:
            self.load_targets(targets)
        
        # Zenoh resources
        self._zenoh_session = None
        self._publisher = None
        
        # Callbacks
        self._on_publish_callbacks: list[Callable[[TargetPose, int], None]] = []
    
    @property
    def topic(self) -> str:
        """Get publish topic."""
        return self._topic
    
    @property
    def targets(self) -> list[TargetPose]:
        """Get all targets."""
        return self._targets.copy()
    
    @property
    def current_index(self) -> int:
        """Get current target index."""
        return self._current_index
    
    @property
    def current_target(self) -> Optional[TargetPose]:
        """Get current target pose."""
        if not self._targets:
            return None
        return self._targets[self._current_index]
    
    @property
    def num_targets(self) -> int:
        """Get number of targets."""
        return len(self._targets)
    
    def _log(self, msg: str) -> None:
        """Log a message."""
        print(f"[{self.name}] {msg}")
    
    def load_targets(self, targets: list[dict]) -> None:
        """Load targets from list of dicts (YAML config format)."""
        self._targets = [TargetPose.from_dict(t) for t in targets]
        self._current_index = 0
        self._log(f"Loaded {len(self._targets)} targets")
    
    def add_target(self, target: TargetPose) -> None:
        """Add a single target."""
        self._targets.append(target)
    
    def clear_targets(self) -> None:
        """Clear all targets."""
        self._targets.clear()
        self._current_index = 0
    
    def next(self, wrap: bool = True) -> bool:
        """
        Advance to next target.
        
        Args:
            wrap: If True, wrap to start when reaching end.
        
        Returns:
            True if advanced, False if at end and wrap=False.
        """
        if not self._targets:
            return False
        
        if self._current_index < len(self._targets) - 1:
            self._current_index += 1
            return True
        elif wrap:
            self._current_index = 0
            return True
        return False
    
    def previous(self, wrap: bool = True) -> bool:
        """Advance to previous target."""
        if not self._targets:
            return False
        
        if self._current_index > 0:
            self._current_index -= 1
            return True
        elif wrap:
            self._current_index = len(self._targets) - 1
            return True
        return False
    
    def goto(self, index: int) -> bool:
        """Go to specific target index."""
        if 0 <= index < len(self._targets):
            self._current_index = index
            return True
        return False
    
    def start_zenoh(self, zenoh_config: Optional[dict] = None) -> None:
        """
        Initialize Zenoh session and publisher.
        
        Args:
            zenoh_config: Zenoh config dict with 'mode' and 'connect' keys.
        """
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
        self._publisher = self._zenoh_session.declare_publisher(self._topic)
        self._log(f"Publisher ready on: {self._topic}")
    
    def use_session(self, session) -> None:
        """
        Use an existing Zenoh session instead of creating one.
        
        Args:
            session: Existing zenoh.Session.
        """
        self._zenoh_session = session
        self._publisher = session.declare_publisher(self._topic)
        self._log(f"Using shared session, publisher on: {self._topic}")
    
    def publish_current(self) -> bool:
        """
        Publish current target pose.
        
        Returns:
            True if published successfully.
        """
        if not self._publisher:
            self._log("Publisher not initialized")
            return False
        
        target = self.current_target
        if target is None:
            self._log("No targets loaded")
            return False
        
        try:
            payload = target.to_proto()
            self._publisher.put(payload)
            self._log(f"Published target {self._current_index}: {target}")
            
            for callback in self._on_publish_callbacks:
                try:
                    callback(target, self._current_index)
                except Exception as e:
                    self._log(f"Callback error: {e}")
            
            return True
        except Exception as e:
            self._log(f"Publish error: {e}")
            return False
    
    def publish_next(self, wrap: bool = True) -> bool:
        """Advance to next target and publish it."""
        if self.next(wrap):
            return self.publish_current()
        return False
    
    def publish_index(self, index: int) -> bool:
        """Go to specific index and publish."""
        if self.goto(index):
            return self.publish_current()
        return False
    
    def on_publish(self, callback: Callable[[TargetPose, int], None]) -> None:
        """Register callback for publish events."""
        self._on_publish_callbacks.append(callback)
    
    def stop(self) -> None:
        """Release Zenoh resources."""
        if self._publisher is not None:
            try:
                self._publisher.undeclare()
            except Exception:
                pass
            self._publisher = None
        
        if self._zenoh_session is not None:
            self._zenoh_session.close()
            self._zenoh_session = None
