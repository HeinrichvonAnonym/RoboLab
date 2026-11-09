import threading
import numpy as np

CAMERAS = {"kinec2": (540, 960, 4, 1), "realsense": (480, 640, 3, 5)}


def decode_rgb(name, payload, width, height, channels, format_type, step=0):
    h, w, c, fmt = CAMERAS[name]
    if (height, width, channels, format_type) != (h, w, c, fmt):
        raise ValueError(f"{name}: unexpected image geometry/format")
    stride = step or w*c
    if stride < w*c or len(payload) != h*stride:
        raise ValueError(f"{name}: invalid image payload/row stride")
    image = np.frombuffer(payload, np.uint8).reshape(h, stride)[:, :w*c].reshape(h, w, c)
    rgb = image[:, :, 2::-1] if name == "kinec2" else image
    return np.array(rgb, copy=True, order="C")


class ObservationUnavailable(Exception):
    """Missing or older than max_age. One occurrence skips the tick; it is not fatal."""


def validate_observation(obs):
    for name, (h, w, _, _) in CAMERAS.items():
        img = obs["image"][name]
        if img.shape != (h, w, 3) or img.dtype != np.uint8:
            raise ValueError(f"{name}: expected uint8 RGB HWC {(h,w,3)}")
    state = np.asarray(obs["state"])
    if state.shape != (8,) or not np.isfinite(state).all() or not 0 <= state[7] <= 1:
        raise ValueError("state must be finite 8D, with gripper in [0,1]")
    if not isinstance(obs["prompt"], str) or not obs["prompt"].strip():
        raise ValueError("prompt must be nonempty text")


class ObservationStore:
    def __init__(self, prompt, max_age_ms):
        self.prompt, self.max_age = prompt, max_age_ms / 1000
        self.lock = threading.Lock()
        self.values = {}

    def put(self, key, value, now, pose=None):
        with self.lock:
            self.values[key] = (value, now, pose)

    def snapshot(self, now):
        with self.lock:
            missing = {"kinec2", "realsense", "state"} - self.values.keys()
            if missing:
                raise ObservationUnavailable(f"Missing observations: {sorted(missing)}")
            ages = {k: now-v[1] for k, v in self.values.items()}
            if any(a < 0 or a > self.max_age for a in ages.values()):
                raise ObservationUnavailable(f"Stale observations (seconds): {ages}")
            obs = {"image": {k: self.values[k][0] for k in CAMERAS},
                   "state": self.values["state"][0].copy(), "prompt": self.prompt,
                   "ages_ms": {k: a*1000 for k, a in ages.items()},
                   "pose_ee": self.values["state"][2]}
        validate_observation(obs)
        return obs
