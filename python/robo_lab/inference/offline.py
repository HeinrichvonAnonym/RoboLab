"""Read-only H5 input checks and nominal-home trajectory verification; no transport imports."""
import json
from pathlib import Path
import h5py
import numpy as np

from .kinematics import Kinematics, rotation_error
from .observation import CAMERAS, decode_rgb, validate_observation
from .policy import CirclePolicy


def verify_h5(path, cfg, output):
    report = {"source": str(Path(path).resolve()), "mode": "offline",
              "robot_commands_published": 0, "hardware_verified": False, "images": {}}
    images = {}
    with h5py.File(path, "r") as h5:
        for name in CAMERAS:
            group = h5[name+"_rgb"]
            n = len(group["timestamps_ns"])
            for i in sorted({0, n//2, n-1}):
                offset, length = int(group["image_offsets"][i]), int(group["image_lengths"][i])
                data = np.asarray(group["image_data"][offset:offset+length]).view(np.uint8).tobytes()
                w, h, c, fmt = (int(group[k][i]) for k in ("width", "height", "channels", "format_type"))
                images[name] = decode_rgb(name, data, w, h, c, fmt)
            report["images"][name] = {"shape": list(images[name].shape), "dtype": str(images[name].dtype),
                                     "sampled_indices": sorted({0, n//2, n-1})}
        for key in ("franka_state", "franka_command"):
            a = np.asarray(h5[key]["joints_position"])
            if a.ndim != 2 or a.shape[1] != 8 or not np.isfinite(a).all():
                raise ValueError(f"{key}: invalid 8D positions")
            if np.any((a[:, 7] < 0) | (a[:, 7] > 1)):
                raise ValueError(f"{key}: gripper outside [0,1]")
            report[key] = {"shape": list(a.shape), "dtype": str(a.dtype)}
        recorded_state = np.asarray(h5["franka_state/joints_position"][0])
        obs = {"image": images, "state": recorded_state, "prompt": cfg["prompt"]}
        validate_observation(obs)

    # H5 may start mid-episode. Keep input validation distinct from the mock
    # home trajectory: never pretend this substituted state is real feedback.
    obs["state"] = np.r_[cfg["robot_config"]["arm_home"], recorded_state[7]]
    report["trajectory_seed"] = "configured nominal home; recorded first gripper state"
    kin = Kinematics(cfg["kinematics_config"])
    policy = CirclePolicy(cfg, kin)
    count = int(np.ceil(cfg["frequency"]*cfg["circle"]["duration_s"]*cfg["circle"]["turns"]))
    chunks = (count+cfg["chunk_size"]-1)//cfg["chunk_size"] + 1  # Also check post-completion hold.
    actions, xyz, targets, pe, re = [], [], [], [], []
    for _ in range(chunks):
        result = policy.infer(obs)
        for action, target in zip(result["action"], result["target_xyz"]):
            actual = kin.fk(action[:7])
            actions.append(action)
            xyz.append(actual[:3, 3])
            targets.append(target)
            pe.append(np.linalg.norm(actual[:3, 3]-target))
            re.append(np.linalg.norm(rotation_error(policy.start[:3, :3], actual[:3, :3])))
    report.update(steps=len(actions), max_position_error_m=float(max(pe)),
                  max_orientation_error_rad=float(max(re)),
                  return_position_error_m=float(np.linalg.norm(xyz[count-1]-policy.start[:3, 3])),
                  hold_max_joint_change_rad=float(np.max(np.abs(np.array(actions)[count:]-actions[count-1]))))
    np.savez(Path(output)/"trajectory.npz", action=actions, fk_xyz=xyz, target_xyz=targets,
             seed=obs["state"], frequency=cfg["frequency"])
    (Path(output)/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
