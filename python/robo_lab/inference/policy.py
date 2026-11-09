from typing import Protocol
import numpy as np

from .kinematics import rotation_error
from .observation import validate_observation


class Policy(Protocol):
    def reset(self) -> None: ...
    def infer(self, observation: dict) -> dict: ...


def validate_actions(actions, previous, cfg, kin):
    a = np.asarray(actions)
    if a.shape != (cfg["chunk_size"], 8) or not np.isfinite(a).all():
        raise ValueError("Invalid action chunk shape or nonfinite values")
    if (np.any(a[:, :7] < kin.lower) or np.any(a[:, :7] > kin.upper)
            or np.any(a[:, 7] < 0) or np.any(a[:, 7] > 1)):
        raise ValueError("Action exceeds joint/gripper limits")
    deltas = np.diff(np.vstack([previous, a]), axis=0)
    if np.any(np.abs(deltas[:, :7]) > cfg["max_joint_speed"] / cfg["frequency"] + 1e-9):
        raise ValueError("Action exceeds per-step joint continuity limit")
    return a.astype(np.float64, copy=True)


class CirclePolicy:
    def __init__(self, cfg, kin):
        self.cfg, self.kin = cfg, kin
        self.reset()

    def reset(self):
        self.step = 0
        self.start = self.previous = None

    def target(self, step):
        circle = self.cfg["circle"]
        u = min(step / (self.cfg["frequency"] * circle["duration_s"] * circle["turns"]), 1.)
        smooth = 10*u**3 - 15*u**4 + 6*u**5
        angle = 2*np.pi*circle["turns"]*smooth
        a, b = {"xy": (0, 1), "yz": (1, 2), "zx": (2, 0)}[circle["plane"]]
        target = self.start.copy()
        target[a, 3] += circle["radius_m"]*(np.cos(angle)-1)
        target[b, 3] += circle["radius_m"]*np.sin(angle)
        return target

    def infer(self, observation):
        validate_observation(observation)
        if self.start is None:
            self.previous = np.asarray(observation["state"], dtype=float).copy()
            self.start = self.kin.fk(self.previous[:7])
        previous = self.previous.copy()
        actions = []
        targets = []
        for i in range(self.cfg["chunk_size"]):
            target = self.target(self.step+i+1)
            q = self.kin.solve(target, previous[:7])
            actual = self.kin.fk(q)
            if (np.linalg.norm(actual[:3, 3]-target[:3, 3]) > self.cfg["position_tolerance_m"]
                    or np.linalg.norm(rotation_error(target[:3, :3], actual[:3, :3]))
                    > self.cfg["orientation_tolerance_rad"]):
                raise ValueError("FK trajectory tolerance exceeded")
            previous = np.r_[q, previous[7]]
            actions.append(previous)
            targets.append(target[:3, 3])
        actions = validate_actions(actions, self.previous, self.cfg, self.kin)
        self.previous = actions[-1].copy()
        self.step += len(actions)
        return {"action": actions, "target_xyz": np.array(targets)}
