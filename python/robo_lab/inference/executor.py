"""Single-flight chunk execution, with state transitions checked at publication."""
import threading
import time
import numpy as np

from .observation import ObservationUnavailable
from .policy import validate_actions


class Executor:
    def __init__(self, cfg, kin, policy, store, publish, event=lambda **kw: None,
                 clock=time.monotonic):
        self.cfg, self.kin, self.policy, self.store = cfg, kin, policy, store
        self.publish, self.event, self.clock = publish, event, clock
        self.lock = threading.RLock()
        self.state = None
        self.epoch, self.active_epoch = 0, -1
        self.fault = None
        self.chunk = None
        self.index = 0
        self.previous = None
        self.next_due = 0.
        self.sequence = 0
        self.chunk_number = 0
        self.unavailable_since = None

    def set_state(self, state):
        if state not in ("init", "gohome", "standby", "inference"):
            self.fail(f"Unknown state machine value: {state!r}")
            return
        with self.lock:
            if self.state != state:
                self.state = state
                self.epoch += 1
                self.chunk = None
                self.fault = None
                self.unavailable_since = None
                self.event(event="state", state=state, epoch=self.epoch)

    def _note_unavailable(self, reason):
        now = self.clock()
        if self.unavailable_since is None:
            self.unavailable_since = now
        elif now - self.unavailable_since > self.cfg["stale_fault_s"]:
            self.fail(reason)

    def fail(self, reason):
        with self.lock:
            self.chunk = None
            if self.fault is None:
                self.fault = str(reason)
                self.event(event="fault", reason=self.fault, state=self.state)

    def tick(self):
        with self.lock:
            if self.state != "inference" or self.fault:
                return False
            epoch = self.epoch
            try:
                obs = self.store.snapshot(self.clock())
            except ObservationUnavailable as exc:
                self._note_unavailable(exc)
                return False
            except Exception as exc:
                self.fail(exc)
                return False
            self.unavailable_since = None
            try:
                if self.active_epoch != epoch:
                    home = np.asarray(self.cfg["robot_config"]["arm_home"])
                    if np.max(np.abs(obs["state"][:7]-home)) > self.cfg["home_tolerance_rad"]:
                        raise ValueError("Not at home; restart control session and home before retry")
                    self.policy.reset()
                    self.previous = obs["state"].copy()
                    self.chunk, self.index, self.chunk_number = None, 0, 0
                    self.active_epoch = epoch
                    self.next_due = self.clock()
                    self.event(event="reset", state=obs["state"].tolist())
                if self.clock() < self.next_due:
                    return False
                needs_infer = self.chunk is None
            except Exception as exc:
                self.fail(exc)
                return False

        # Do not block camera/state callbacks during inference. A transition
        # invalidates its result, including inference -> standby -> inference.
        if needs_infer:
            started = self.clock()
            try:
                result = self.policy.infer(obs)
                chunk = validate_actions(result["action"], self.previous, self.cfg, self.kin)
            except Exception as exc:
                with self.lock:
                    if self.epoch == epoch:
                        self.fail(exc)
                return False
            with self.lock:
                if self.epoch != epoch or self.fault or self.state != "inference":
                    return False
                self.chunk, self.index = chunk, 0
                self.chunk_number += 1
                self.event(event="infer", chunk=self.chunk_number, steps=len(chunk),
                           duration_ms=(self.clock()-started)*1000)

        with self.lock:
            if self.epoch != epoch or self.fault or self.state != "inference":
                return False
            try:
                now = self.clock()
                obs = self.store.snapshot(now)  # Recheck even while consuming a cached chunk.
            except ObservationUnavailable as exc:
                self._note_unavailable(exc)
                return False
            except Exception as exc:
                self.fail(exc)
                return False
            self.unavailable_since = None
            try:
                action = self.chunk[self.index].copy()
                self.sequence += 1
                self.publish(action, self.sequence)
                published = self.clock()
                self.event(event="command", sequence=self.sequence, chunk=self.chunk_number,
                           index=self.index, action=action.tolist(), state=obs["state"].tolist(),
                           pose_ee=obs["pose_ee"], ages_ms=obs["ages_ms"],
                           lateness_ms=max(0., now-self.next_due)*1000,
                           published_monotonic=published)
                self.previous = action
                self.index += 1
                if self.index == len(self.chunk):
                    self.chunk = None
                # Never catch up by rapidly publishing missed deadlines.
                self.next_due = published + 1/self.cfg["frequency"]
                return True
            except Exception as exc:
                self.fail(exc)
                return False
