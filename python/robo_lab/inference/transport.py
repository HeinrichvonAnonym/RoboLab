import json
import threading
import time

import numpy as np
import zenoh

from .observation import decode_rgb, ObservationStore
from .executor import Executor
from .kinematics import Kinematics
from .worker import ProcessPolicy, resolve_policy
from .proto import franka_pb2, kinect_pb2


def zenoh_config(connect=(), listen=(), isolated=False):
    cfg = zenoh.Config()
    if connect:
        cfg.insert_json5("connect/endpoints", json.dumps(list(connect)))
    if listen:
        cfg.insert_json5("listen/endpoints", json.dumps(list(listen)))
    if isolated:
        cfg.insert_json5("scouting/multicast/enabled", "false")
        cfg.insert_json5("scouting/gossip/enabled", "false")
        if not listen:
            cfg.insert_json5("listen/endpoints", "[]")
    return cfg


class JsonLog:
    def __init__(self, path):
        self.file = open(path, "x", buffering=1)
        self.lock = threading.Lock()

    def __call__(self, **event):
        with self.lock:
            self.file.write(json.dumps(dict(monotonic=time.monotonic(), **event), allow_nan=False)+"\n")
        if event["event"] in ("state", "fault", "reset", "infer", "policy_ready"):
            print(json.dumps(event), flush=True)

    def close(self):
        self.file.close()


class ZenohNode:
    def __init__(self, cfg, session, event=lambda **kw: None):
        self.cfg, self.session = cfg, session
        self.subscribers = []
        self.store = ObservationStore(cfg["prompt"], cfg["max_age_ms"])
        kin = Kinematics(cfg["kinematics_config"])
        self.policy = ProcessPolicy(cfg, resolve_policy(cfg))
        event(event="policy_ready", pid=self.policy.process.pid, execution="spawned_process")
        self.publisher = session.declare_publisher(cfg["topics"]["command"])
        self.executor = Executor(cfg, kin, self.policy, self.store, self.publish, event)
        for key in ("kinec2", "realsense", "state", "state_machine"):
            self.subscribers.append(session.declare_subscriber(
                cfg["topics"][key], lambda sample, key=key: self.receive(key, sample)))

    def receive(self, key, sample):
        received = time.monotonic()
        try:
            data = sample.payload.to_bytes()
            if key == "state_machine":
                self.executor.set_state(data.decode("utf-8"))
            elif key == "state":
                msg = franka_pb2.RobotObservation.FromString(data)
                state = np.array([j.position for j in msg.joints], dtype=np.float64)
                # Gripper feedback arrives after the arm. A 7D sample is not a fault;
                # a lasting gap still trips the missing-state timeout.
                if state.shape != (8,):
                    return
                if not np.isfinite(state).all() or not 0 <= state[7] <= 1:
                    raise ValueError("Invalid robot state")
                pose = None
                if msg.HasField("pose_ee"):
                    p, r = msg.pose_ee.pos, msg.pose_ee.rot
                    pose = [p.x, p.y, p.z, r.w, r.x, r.y, r.z]
                    if not np.isfinite(pose).all():
                        raise ValueError("Nonfinite end-effector feedback")
                self.store.put(key, state, received, pose)
            else:
                msg = kinect_pb2.rgbImage.FromString(data)
                rgb = decode_rgb(key, msg.image, msg.width, msg.height, msg.channels, msg.type, msg.step)
                self.store.put(key, rgb, received)
        except Exception as exc:
            self.executor.fail(f"{key}: {exc}")

    def publish(self, action, sequence):
        msg = franka_pb2.RobotCommand(type=franka_pb2.RobotCommand.TYPE_JOINT_TARGET,
                                      sequence=sequence % (2**32), mode="position",
                                      note="robolab_python_fake_circle")
        for value in action:
            msg.joints.add(position=float(value))
        self.publisher.put(msg.SerializeToString())

    def close(self):
        self.executor.set_state("standby")
        for sub in self.subscribers:
            sub.undeclare()
        self.publisher.undeclare()
        self.policy.close()
