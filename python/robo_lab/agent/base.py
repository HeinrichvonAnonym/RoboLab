"""
Base Agent class for all system layers.

All agents (SpinalCord, Cerebellum, Brain) inherit from AgentBase,
providing a consistent interface for lifecycle management.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Optional


class AgentBase(ABC):
    """
    Abstract base class for all agent systems.
    
    Provides:
        - Lifecycle management (start, stop, spin)
        - Thread-safe running state
        - Configurable update rate
    
    Subclasses must implement:
        - setup(): Initialize resources
        - step(): Single iteration of the agent loop
        - cleanup(): Release resources
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        name: Optional[str] = None,
        rate_hz: float = 10.0,
    ):
        """
        Initialize agent base.
        
        Args:
            config: Configuration dict from YAML.
            name: Agent name for logging. Defaults to class name.
            rate_hz: Target update rate in Hz. 0 = run as fast as possible.
        """
        self.config = config or {}
        self.name = name or self.__class__.__name__
        self.rate_hz = rate_hz
        self._period = 1.0 / rate_hz if rate_hz > 0 else 0

        self._running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        """Thread-safe check if agent is running."""
        with self._lock:
            return self._running

    @running.setter
    def running(self, value: bool):
        with self._lock:
            self._running = value

    def setup(self) -> None:
        """
        Initialize resources before main loop.
        
        Override in subclass for custom initialization.
        Called once when start() is invoked.
        """
        pass

    @abstractmethod
    def step(self) -> None:
        """
        Single iteration of the agent loop.
        
        Must be implemented by subclasses.
        Called repeatedly while agent is running.
        """
        pass

    def cleanup(self) -> None:
        """
        Release resources after main loop ends.
        
        Override in subclass for custom cleanup.
        Called once when stop() completes.
        """
        pass

    def spin(self) -> None:
        """
        Main loop: setup -> step repeatedly -> cleanup.
        
        Runs until stop() is called.
        """
        print(f"[{self.name}] Starting...")
        try:
            self.setup()
            self.running = True
            print(f"[{self.name}] Running at {self.rate_hz} Hz")

            while self.running:
                t_start = time.perf_counter()
                
                try:
                    self.step()
                except Exception as e:
                    print(f"[{self.name}] Error in step: {e}")

                # Rate limiting
                if self._period > 0:
                    elapsed = time.perf_counter() - t_start
                    sleep_time = self._period - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except Exception as e:
            print(f"[{self.name}] Fatal error: {e}")
        finally:
            self.cleanup()
            print(f"[{self.name}] Stopped.")

    def start(self, threaded: bool = False) -> Optional[threading.Thread]:
        """
        Start the agent.
        
        Args:
            threaded: If True, run in a background thread and return immediately.
                      If False, block until stop() is called.
        
        Returns:
            Thread object if threaded=True, else None.
        """
        if threaded:
            self._thread = threading.Thread(target=self.spin, name=self.name, daemon=True)
            self._thread.start()
            return self._thread
        else:
            self.spin()
            return None

    def stop(self, timeout: float = 5.0) -> bool:
        """
        Stop the agent gracefully.
        
        Args:
            timeout: Max seconds to wait for thread to finish.
        
        Returns:
            True if stopped cleanly, False if timeout.
        """
        print(f"[{self.name}] Stopping...")
        self.running = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print(f"[{self.name}] Warning: thread did not stop within {timeout}s")
                return False

        return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for agent thread to complete.
        
        Args:
            timeout: Max seconds to wait. None = wait forever.
        
        Returns:
            True if thread completed, False if timeout.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True

    def __repr__(self) -> str:
        status = "running" if self.running else "stopped"
        return f"<{self.name} [{status}] @ {self.rate_hz}Hz>"
