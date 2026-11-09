"""Synchronous policy RPC in a spawned process, separate from Zenoh callbacks.

No prefetch: one request is completed before the executor consumes its chunk.
The child owns no robot transport and cannot publish commands.
"""
import multiprocessing as mp
import time

from .kinematics import Kinematics
from .policy import CirclePolicy


def resolve_policy(cfg):
    if cfg["policy"] == "fake_circle":
        return CirclePolicy
    if cfg["policy"] == "lingbot_vla_v2":
        from .lingbot_policy import LingbotPolicy
        return LingbotPolicy
    raise ValueError(f"Unknown policy: {cfg['policy']}")


def _serve(conn, cfg, policy_factory):
    try:
        policy = policy_factory(cfg, Kinematics(cfg["kinematics_config"]))
        conn.send((True, None))
        while True:
            operation, observation = conn.recv()
            try:
                if operation == "reset":
                    result = policy.reset()
                elif operation == "infer":
                    result = policy.infer(observation)
                else:
                    raise ValueError(f"Unknown policy operation: {operation}")
                conn.send((True, result))
            except Exception as exc:
                conn.send((False, f"{type(exc).__name__}: {exc}"))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        conn.close()


class ProcessPolicy:
    def __init__(self, cfg, policy_factory=CirclePolicy):
        self.cfg = cfg
        self.factory = policy_factory
        self.process = self.conn = None
        self._start()

    def _start(self):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child, self.cfg, self.factory),
                                       name="robolab-policy", daemon=True)
        self.conn = parent
        self.process.start()
        child.close()
        # Circle startup is a few milliseconds. LingBot loads a 6B checkpoint.
        self._receive(float(self.cfg.get("policy_init_timeout_s", 10.)))

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if self.conn.poll(min(.02, max(0., deadline-time.monotonic()))):
                    success, result = self.conn.recv()
                    if not success:
                        raise ValueError(result)
                    return result
                if not self.process.is_alive():
                    raise RuntimeError("Policy process exited unexpectedly")
            raise TimeoutError(f"Policy process exceeded {timeout:g}s timeout")
        except (EOFError, BrokenPipeError, OSError, TimeoutError, RuntimeError):
            self.close()
            raise

    def reset(self):
        if self.process is None:
            self._start()
        self.conn.send(("reset", None))
        self._receive(self.cfg["inference_timeout_s"])

    def infer(self, observation):
        if self.process is None:
            raise RuntimeError("Policy process stopped; reset required")
        self.conn.send(("infer", observation))
        return self._receive(self.cfg["inference_timeout_s"])

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            self.process.close()
            self.process = None
        if self.conn is not None:
            self.conn.close()
            self.conn = None
