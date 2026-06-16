import json
import pickle
from typing import Any, Dict, Optional, Sequence

import numpy as np

from gello.robots.robot import Robot

DEFAULT_ROBOT_KEY = "gello/robot"


def _open_session(connect: Optional[Sequence[str]] = None):
    try:
        import zenoh
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: pip install 'eclipse-zenoh<1.8' protobuf"
        ) from exc

    conf = zenoh.Config()
    if connect:
        conf.insert_json5("connect/endpoints", json.dumps(list(connect)))
    return zenoh.open(conf)


def _payload_to_bytes(payload) -> bytes:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    if hasattr(payload, "to_bytes"):
        return bytes(payload.to_bytes())
    return bytes(payload)


class ZenohServerRobot:
    def __init__(
        self,
        robot: Robot,
        key: str = DEFAULT_ROBOT_KEY,
        connect: Optional[Sequence[str]] = None,
    ):
        self._robot = robot
        self._key = key.rstrip("/")
        self._session = _open_session(connect)
        print(f"Robot server declaring Zenoh queryable on {self._key}, Robot: {robot}")
        self._queryable = self._session.declare_queryable(self._key, self._handle_query)

    def _handle_query(self, query) -> None:
        try:
            request = pickle.loads(_payload_to_bytes(query.payload))
            method = request.get("method")
            args = request.get("args", {})
            result: Any
            if method == "num_dofs":
                result = self._robot.num_dofs()
            elif method == "get_joint_state":
                result = self._robot.get_joint_state()
            elif method == "command_joint_state":
                result = self._robot.command_joint_state(**args)
            elif method == "get_observations":
                result = self._robot.get_observations()
            else:
                result = {"error": f"Invalid method: {method}"}
        except Exception as exc:
            result = {"error": str(exc)}
        query.reply(self._key, pickle.dumps(result))

    def serve(self) -> None:
        try:
            import time

            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        for attr in ("_queryable", "_session"):
            obj = getattr(self, attr, None)
            close = getattr(obj, "close", None)
            if close is not None:
                close()


class ZenohClientRobot(Robot):
    def __init__(
        self,
        key: str = DEFAULT_ROBOT_KEY,
        connect: Optional[Sequence[str]] = None,
        timeout_s: float = 5.0,
    ):
        self._key = key.rstrip("/")
        self._session = _open_session(connect)
        self._timeout_s = timeout_s

    def _request(self, method: str, args: Optional[Dict[str, Any]] = None):
        request = {"method": method, "args": {} if args is None else args}
        replies = self._session.get(
            self._key,
            payload=pickle.dumps(request),
            timeout=self._timeout_s,
        )
        for reply in replies:
            result = pickle.loads(_payload_to_bytes(reply.ok.payload))
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        raise RuntimeError(f"Zenoh timeout waiting for robot method '{method}'")

    def num_dofs(self) -> int:
        return self._request("num_dofs")

    def get_joint_state(self) -> np.ndarray:
        return self._request("get_joint_state")

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        return self._request(
            "command_joint_state", {"joint_state": np.asarray(joint_state)}
        )

    def get_observations(self) -> Dict[str, np.ndarray]:
        return self._request("get_observations")

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if close is not None:
            close()
