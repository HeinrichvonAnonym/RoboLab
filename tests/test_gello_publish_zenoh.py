"""Hardware-free checks: python -m unittest discover -s tests -p test_gello_publish_zenoh.py"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from unittest import mock
import uuid

import numpy as np
import yaml
import zenoh


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/third_party/gello_software"))
sys.path.insert(0, str(ROOT / "scripts/gen"))
import franka_pb2

SPEC = importlib.util.spec_from_file_location(
    "gello_publisher_under_test",
    ROOT / "python/third_party/gello_software/experiments/gello_publish_zenoh.py",
)
gp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gp
SPEC.loader.exec_module(gp)

HOME = np.array([0.1, -0.2, 0.3, -2.1, 0.4, 1.5, -0.6])


class NormalizerTests(unittest.TestCase):
    def test_limits_match_repository_configuration(self):
        config = yaml.safe_load(
            (ROOT / "config/cartesian_control/cartesian_control_config.yaml").read_text()
        )
        np.testing.assert_array_equal(gp.PANDA_JOINT_LOWER, config["joint_lower"])
        np.testing.assert_array_equal(gp.PANDA_JOINT_UPPER, config["joint_upper"])

    def test_first_frame_removes_turns_and_preserves_special_joints(self):
        raw = np.array([0.328, 0.6, 6.036, -2.105, 0.074, 3.5, -0.011, 0.37])
        original = raw.copy()
        result, clipped = gp.PandaJointPositionNormalizer().normalize(raw)
        expected = raw.copy()
        expected[2] -= 2 * np.pi
        np.testing.assert_allclose(result, expected, atol=1e-12)
        self.assertEqual(result[5], 3.5)
        self.assertEqual(result[3], raw[3])
        self.assertEqual(result[7], 0.37)
        self.assertEqual(clipped, ())
        np.testing.assert_array_equal(raw, original)

    def test_each_joint_accepts_equivalent_angles_and_exact_endpoints(self):
        for boundary in (gp.PANDA_JOINT_LOWER, gp.PANDA_JOINT_UPPER, HOME):
            result, clipped = gp.PandaJointPositionNormalizer().normalize(boundary)
            np.testing.assert_array_equal(result, boundary)
            self.assertEqual(clipped, ())
            for turns in (-5, -1, 1, 5):
                shifted = boundary + turns * 2 * np.pi
                result, _ = gp.PandaJointPositionNormalizer().normalize(shifted)
                np.testing.assert_allclose(result, boundary, atol=1e-12)

    def test_first_frame_selects_nearest_equivalent_interval_then_clamps(self):
        rng = np.random.default_rng(12)
        for _ in range(100):
            raw = rng.uniform(-40, 40, 7)
            result, _ = gp.PandaJointPositionNormalizer().normalize(raw)
            self.assertTrue(np.all(result >= gp.PANDA_JOINT_LOWER))
            self.assertTrue(np.all(result <= gp.PANDA_JOINT_UPPER))
            for j in range(7):
                candidates = raw[j] + 2 * np.pi * np.arange(-10, 11)
                bounded = np.clip(candidates, gp.PANDA_JOINT_LOWER[j], gp.PANDA_JOINT_UPPER[j])
                best = int(np.argmin(np.abs(candidates - bounded)))
                self.assertAlmostEqual(result[j], bounded[best], places=11)

    def test_each_joint_clamps_at_both_limits(self):
        for sign, boundary in ((-1, gp.PANDA_JOINT_LOWER), (1, gp.PANDA_JOINT_UPPER)):
            for j in range(7):
                raw = HOME.copy()
                raw[j] = boundary[j] + sign * 0.05
                result, clipped = gp.PandaJointPositionNormalizer().normalize(raw)
                self.assertEqual(result[j], boundary[j])
                self.assertEqual(clipped, (j + 1,))

    def test_crossing_pi_in_both_directions_stays_at_same_limit_and_recovers(self):
        for sign in (-1, 1):
            normalizer = gp.PandaJointPositionNormalizer()
            samples = sign * np.array([2.8, 3.0, 3.12, -3.10, -3.0, -3.10, 3.12, 2.8])
            outputs = []
            for angle in samples:
                raw = HOME.copy()
                raw[0] = angle
                result, _ = normalizer.normalize(raw)
                outputs.append(result[0])
            boundary = gp.PANDA_JOINT_UPPER[0] if sign > 0 else gp.PANDA_JOINT_LOWER[0]
            np.testing.assert_allclose(outputs, [sign * 2.8] + [boundary] * 6 + [sign * 2.8])
            self.assertLess(np.max(np.abs(np.diff(outputs))), 0.1)

    def test_encoder_zero_rollover_does_not_shift_target(self):
        normalizer = gp.PandaJointPositionNormalizer()
        outputs = []
        for angle in (2 * np.pi - 0.1, 2 * np.pi - 0.01, 0.01, 0.1):
            raw = HOME.copy()
            raw[2] = angle
            result, clipped = normalizer.normalize(raw)
            outputs.append(result[2])
            self.assertEqual(clipped, ())
        np.testing.assert_allclose(outputs, [-0.1, -0.01, 0.01, 0.1], atol=1e-12)

    def test_joint_six_crosses_pi_without_becoming_negative(self):
        normalizer = gp.PandaJointPositionNormalizer()
        for raw_angle, expected in ((3.0, 3.0), (3.12, 3.12), (-3.10, 2 * np.pi - 3.10),
                                    (-2.783185307179586, 3.5)):
            raw = HOME.copy()
            raw[5] = raw_angle
            result, clipped = normalizer.normalize(raw)
            self.assertAlmostEqual(result[5], expected)
            self.assertEqual(clipped, ())

    def test_invalid_frames_do_not_change_state_or_poison_initialization(self):
        invalid = [np.zeros(6), np.zeros(9), np.zeros((1, 7)), 0, ["bad"] * 7,
                   np.array([np.nan] + [0] * 6), np.array([0] * 6 + [np.inf]),
                   np.array([0] * 7 + [-np.inf]), np.array([0] * 7 + [np.nan])]
        normalizer = gp.PandaJointPositionNormalizer()
        control = gp.PandaJointPositionNormalizer()
        for bad in invalid:
            with self.subTest(frame=str(bad)), self.assertRaises((ValueError, TypeError)):
                normalizer.normalize(bad)
        for j1 in (2.8, 3.12, -3.10, 2.8):
            raw = np.append(HOME, 0.42)
            raw[0] = j1
            np.testing.assert_array_equal(normalizer.normalize(raw)[0], control.normalize(raw)[0])
            for bad in invalid:
                with self.assertRaises((ValueError, TypeError)):
                    normalizer.normalize(bad)

    def test_invalid_frequency_is_rejected_before_opening_hardware(self):
        for hz in (0, -1, float("nan"), float("inf")):
            with mock.patch.object(gp, "GelloAgent") as agent, self.assertRaises(ValueError):
                gp.main(gp.Args(gello_port="unused", hz=hz))
            agent.assert_not_called()


class EndOfSimulation(Exception):
    pass


class PublisherIntegrationTests(unittest.TestCase):
    def run_simulation(self, dry_run, include_gripper=True):
        # Only loopback connections and a unique topic: no device or existing
        # robot session can participate in this test.
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        endpoint = f"tcp/127.0.0.1:{port}"
        topic = f"test/gello_joint_norm/{uuid.uuid4().hex}"
        server_config = zenoh.Config()
        server_config.insert_json5("listen/endpoints", json.dumps([endpoint]))
        server_config.insert_json5("scouting/multicast/enabled", "false")
        server_config.insert_json5("scouting/gossip/enabled", "false")
        open_session = zenoh.open
        messages = []
        lock = threading.Lock()

        def receive(sample):
            cmd = franka_pb2.RobotCommand.FromString(sample.payload.to_bytes())
            with lock:
                messages.append(cmd)

        base = np.array([2.8, 0.6, 6.036, -2.105, 0.074, 3.5, -0.011, 0.37])
        if not include_gripper:
            base = base[:7]
        before_boundary = base.copy()
        before_boundary[0] = 3.12
        after_boundary = base.copy()
        after_boundary[0] = -3.10
        recovered = base.copy()
        recovered[0] = 2.7
        bad = base.copy()
        bad[-1] = np.nan
        frames = [np.zeros(9), base, before_boundary, bad, ["bad"] * 7,
                  after_boundary, recovered]
        processed = []
        normalizer_type = gp.PandaJointPositionNormalizer

        class RecordingNormalizer(normalizer_type):
            def normalize(self, action):
                result = super().normalize(action)
                processed.append(result[0].copy())
                return result

        class SimulatedAgent:
            def __init__(self, **kwargs):
                self.index = 0
                self.started = None

            def act(self, observation):
                if self.started is None:
                    self.started = time.monotonic()
                if time.monotonic() - self.started > 1.15:
                    raise EndOfSimulation()
                if self.index < len(frames):
                    result = frames[self.index]
                    self.index += 1
                    return result
                return recovered

        def isolated_open(config):
            config.insert_json5("mode", '"client"')
            config.insert_json5("scouting/multicast/enabled", "false")
            config.insert_json5("scouting/gossip/enabled", "false")
            config.insert_json5("listen/endpoints", "[]")
            session = open_session(config)
            time.sleep(0.15)
            return session

        stdout, stderr = io.StringIO(), io.StringIO()
        with open_session(server_config) as server:
            subscriber = server.declare_subscriber(topic, receive)
            try:
                with (mock.patch.object(gp, "GelloAgent", SimulatedAgent),
                      mock.patch.object(gp, "PandaJointPositionNormalizer", RecordingNormalizer),
                      mock.patch.object(zenoh, "open", isolated_open),
                      contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)):
                    with self.assertRaises(EndOfSimulation):
                        gp.main(gp.Args(gello_port="simulated", cmd_topic=topic, hz=100,
                                        connect=(endpoint,), dry_run=dry_run, note="simulation"))
                # Let pending callbacks finish before checking the exact count.
                deadline = time.monotonic() + 2
                while not dry_run and len(messages) < len(processed) and time.monotonic() < deadline:
                    time.sleep(0.01)
            finally:
                subscriber.undeclare()

        self.assertGreater(len(processed), 100)
        expected = base.copy()
        expected[2] -= 2 * np.pi
        np.testing.assert_allclose(processed[0], expected, atol=1e-12)
        self.assertEqual(processed[1][0], gp.PANDA_JOINT_UPPER[0])
        self.assertEqual(processed[2][0], gp.PANDA_JOINT_UPPER[0])
        self.assertAlmostEqual(processed[3][0], 2.7)
        for result in processed:
            self.assertTrue(np.all(result[:7] >= gp.PANDA_JOINT_LOWER))
            self.assertTrue(np.all(result[:7] <= gp.PANDA_JOINT_UPPER))
            self.assertEqual(len(result), 8 if include_gripper else 7)
            if include_gripper:
                self.assertEqual(result[7], 0.37)
        # Three invalid frames arrive in one second and emit one warning.
        self.assertEqual(stderr.getvalue().count("Dropping invalid GELLO frame"), 1)
        log = stdout.getvalue()
        self.assertIn("raw_q=", log)
        self.assertIn("q=", log)
        self.assertIn("clipped_since_log=[1]", log)
        self.assertIn(f"dry_run={dry_run}", log)
        if dry_run:
            self.assertEqual(messages, [])
        else:
            self.assertEqual(len(messages), len(processed))
            for sequence, (cmd, expected) in enumerate(zip(messages, processed), start=1):
                self.assertEqual(cmd.sequence, sequence)
                self.assertEqual(cmd.type, franka_pb2.RobotCommand.TYPE_JOINT_TARGET)
                self.assertEqual(cmd.mode, "position")
                self.assertEqual(cmd.note, "simulation")
                np.testing.assert_array_equal([j.position for j in cmd.joints], expected)
                self.assertTrue(all(j.velocity == 0 and j.effort == 0 for j in cmd.joints))

    def test_real_zenoh_publishes_normalized_commands_and_drops_invalid_frames(self):
        self.run_simulation(dry_run=False)

    def test_dry_run_normalizes_and_logs_without_publishing(self):
        self.run_simulation(dry_run=True)

    def test_real_zenoh_publishes_seven_arm_joints_without_gripper(self):
        self.run_simulation(dry_run=False, include_gripper=False)


if __name__ == "__main__":
    unittest.main()
