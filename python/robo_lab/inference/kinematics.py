"""NumPy counterpart of cartesian_control/JacobianCalculator (Craig MDH)."""
import numpy as np


def rotation_error(target, current):
    # Spatial SO(3) log. atan2 is accurate for the small IK residuals here.
    r = target @ current.T
    v = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]]) / 2
    s = np.linalg.norm(v)
    c = np.clip((np.trace(r)-1)/2, -1, 1)
    if c < -0.999999:
        raise ValueError("IK orientation residual too close to pi")
    return v if s < 1e-12 else v * (np.arctan2(s, c) / s)


class Kinematics:
    def __init__(self, cfg):
        self.dh = np.asarray(cfg["dh_params"], dtype=float)
        self.lower = np.asarray(cfg["joint_lower"], dtype=float)
        self.upper = np.asarray(cfg["joint_upper"], dtype=float)
        self.damping = float(cfg["damping"])
        if (self.dh.shape != (7, 4) or self.lower.shape != (7,)
                or self.upper.shape != (7,) or not np.isfinite(self.dh).all()
                or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all()
                or not np.all(self.lower < self.upper)
                or not np.isfinite(self.damping) or self.damping <= 0):
            raise ValueError("Invalid kinematics configuration")
        offset = np.asarray(cfg["tool_offset"], dtype=float)
        if offset.shape != (7,) or not np.isfinite(offset).all():
            raise ValueError("Invalid TCP offset")
        xyz, quat = offset[:3], offset[3:]
        if np.linalg.norm(quat) < 1e-12:
            raise ValueError("Invalid TCP quaternion")
        x, y, z, w = quat / np.linalg.norm(quat)
        self.tool = np.eye(4)
        self.tool[:3, 3] = xyz
        self.tool[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                              [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                              [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]

    def links(self, q):
        result, t = [], np.eye(4)
        for qi, (theta, d, a, alpha) in zip(q, self.dh):
            ct, st = np.cos(qi+theta), np.sin(qi+theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            t = t @ np.array([[ct, -st, 0, a], [st*ca, ct*ca, -sa, -d*sa],
                               [st*sa, ct*sa, ca, d*ca], [0, 0, 0, 1]])
            result.append(t)
        return result

    def fk(self, q):
        return self.links(q)[-1] @ self.tool

    def jacobian(self, q):
        links = self.links(q)
        p = (links[-1] @ self.tool)[:3, 3]
        return np.stack([np.r_[np.cross(t[:3, 2], p-t[:3, 3]), t[:3, 2]]
                         for t in links], axis=1)

    def solve(self, target, seed):
        q = np.asarray(seed, dtype=float).copy()
        for _ in range(120):
            current = self.fk(q)
            error = np.r_[target[:3, 3]-current[:3, 3],
                          rotation_error(target[:3, :3], current[:3, :3])]
            if np.linalg.norm(error[:3]) < 1e-5 and np.linalg.norm(error[3:]) < 1e-4:
                return q
            j = self.jacobian(q)
            dq = j.T @ np.linalg.solve(j @ j.T + self.damping**2*np.eye(6), error)
            q = np.clip(q + np.clip(dq, -0.02, 0.02), self.lower, self.upper)
        raise ValueError("IK failed to converge; chunk rejected")
