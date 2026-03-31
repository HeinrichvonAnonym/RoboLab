#!/usr/bin/env python3
"""
Start script for roboLab agent pipeline.

Loads configuration and instantiates the 3-tier agent system:
  - system_0: SpinalCord (reflexes) - 100Hz
  - system_1: Cerebellum (coordination) - 50Hz
  - system_2: Brain (perception/decisions) - 10Hz

Each system runs in its own thread with independent update rates.

Usage:
    python scripts/start.py --config config/senior_care.yaml
    python scripts/start.py --config config/senior_care.yaml --test-image path/to/image.jpg
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import yaml


# Registry mapping config type names to classes
SYSTEM_REGISTRY = {
    # System 0
    "SpinalCordSeniorCare": "robo_lab.agent.system_0.spinal_cord_senior_care.SpinalCordSeniorCare",
    # System 1
    "CerebellumSeniorCare": "robo_lab.agent.system_1.cerebellum_senior_care.CerebellumSeniorCare",
    # System 2
    "BrainSeniorCare": "robo_lab.agent.system_2.brain_senior_care.BrainSeniorCare",
}


def import_class(full_path: str) -> type:
    """Import a class from its full module path."""
    module_path, class_name = full_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class AgentPipeline:
    """
    3-tier agent pipeline following the config hierarchy.
    
    Each system runs in its own thread with independent update rates:
        - system_0 (SpinalCord): 100Hz - fast reflexes
        - system_1 (Cerebellum): 50Hz - motion coordination
        - system_2 (Brain): 10Hz - perception/decisions
    
    Attributes:
        system_0: SpinalCord layer (low-level reflexes)
        system_1: Cerebellum layer (coordination)
        system_2: Brain layer (perception/decisions)
    """

    def __init__(self, config: dict):
        self.config = config
        self._shutdown_event = threading.Event()
        
        # Global zenoh config (shared across systems)
        self._zenoh_config = config.get("zenoh", {})
        
        # Instantiate systems (but don't start yet)
        self.system_0 = self._instantiate("system_0", config.get("system_0"))
        self.system_1 = self._instantiate("system_1", config.get("system_1"))
        self.system_2 = self._instantiate("system_2", config.get("system_2"))
        
        self._systems = [s for s in [self.system_0, self.system_1, self.system_2] if s]

    def _instantiate(self, system_name: str, system_config: Optional[dict]) -> Any:
        """Instantiate a system from config."""
        if system_config is None:
            print(f"[WARN] {system_name} not configured, skipping.")
            return None

        type_name = system_config.get("type")
        if type_name is None:
            print(f"[WARN] {system_name} missing 'type', skipping.")
            return None

        class_path = SYSTEM_REGISTRY.get(type_name)
        if class_path is None:
            raise ValueError(f"Unknown system type: {type_name}")

        # Merge global zenoh config into system config
        merged_config = dict(system_config)
        if self._zenoh_config and "zenoh" not in merged_config:
            merged_config["zenoh"] = self._zenoh_config

        cls = import_class(class_path)
        instance = cls(config=merged_config)
        print(f"[OK] {system_name}: {type_name} instantiated")
        return instance

    @property
    def brain(self):
        """Convenience accessor for system_2 (Brain)."""
        return self.system_2

    @property
    def cerebellum(self):
        """Convenience accessor for system_1 (Cerebellum)."""
        return self.system_1

    @property
    def spinal_cord(self):
        """Convenience accessor for system_0 (SpinalCord)."""
        return self.system_0

    def start(self, blocking: bool = True) -> None:
        """
        Start all agent systems in their own threads.
        
        Args:
            blocking: If True, block until shutdown signal. If False, return immediately.
        """
        print("\n=== Starting Agent Pipeline ===")
        
        # Start each system in a thread
        for system in self._systems:
            system.start(threaded=True)
        
        print(f"Started {len(self._systems)} systems")
        
        if blocking:
            # Wait for shutdown signal
            try:
                self._shutdown_event.wait()
            except KeyboardInterrupt:
                pass
            finally:
                self.stop()

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop all agent systems gracefully.
        
        Args:
            timeout: Max seconds to wait for each system to stop.
        """
        print("\n=== Stopping Agent Pipeline ===")
        self._shutdown_event.set()
        
        # Stop in reverse order (brain first, spinal cord last)
        for system in reversed(self._systems):
            system.stop(timeout=timeout)
        
        print("All systems stopped.")

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for all systems to finish."""
        for system in self._systems:
            system.wait(timeout=timeout)

    def request_shutdown(self) -> None:
        """Request graceful shutdown (can be called from signal handler)."""
        self._shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(description="Start roboLab agent pipeline")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/senior_care.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
        help="Optional: test perception with an image (runs once, then exits)",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="Don't block after starting (for embedding in other apps)",
    )
    args = parser.parse_args()

    # Ensure we can import robo_lab
    script_dir = Path(__file__).resolve().parent
    python_root = script_dir.parent
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

    # Resolve config path relative to python/
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = python_root / config_path

    print(f"Loading config: {config_path}")
    config = load_config(str(config_path))

    # Build pipeline
    pipeline = AgentPipeline(config)

    # Setup signal handlers
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[SIGNAL] Received {sig_name}, shutting down...")
        pipeline.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Test mode: run perception once and exit
    if args.test_image and pipeline.brain is not None:
        import cv2
        print(f"\n=== Test Mode: Perception ===")
        print(f"Image: {args.test_image}")
        
        result = pipeline.brain.perceive(
            args.test_image,
            prompt="person. cup. chair. table.",
        )
        
        print(f"  {result.scene_description}")
        print(f"  Detected {len(result.segments)} objects: {result.objects_of_interest}")
        
        # Save visualization
        img = cv2.imread(args.test_image)
        output_path = "perception_test_output.jpg"
        pipeline.brain.visualize(img, result, output_path=output_path)
        print(f"  Saved visualization to: {output_path}")
        
        return pipeline

    # Normal mode: start pipeline
    print("\nPipeline ready. Press Ctrl+C to stop.")
    pipeline.start(blocking=not args.no_block)

    return pipeline


if __name__ == "__main__":
    pipeline = main()
