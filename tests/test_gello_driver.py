"""Hardware-free failure checks for the GELLO real-device path."""

import contextlib
import io
import time
import unittest
from threading import Event, Lock
from unittest import mock

import numpy as np

from gello.dynamixel.driver import DynamixelDriver
from gello.robots.dynamixel import DynamixelRobot
from dynamixel_sdk import COMM_RX_TIMEOUT


class DriverFailureTests(unittest.TestCase):
    def reader(self):
        driver = DynamixelDriver.__new__(DynamixelDriver)
        driver._is_fake = False
        driver._port = "/test/gello"
        driver._stop_thread = Event()
        driver._lock = Lock()
        driver._joint_angles = np.arange(8)
        driver._velocities = np.arange(8) * 2
        driver._last_read_time = time.monotonic()
        return driver

    def test_real_robot_connection_failure_cannot_produce_fake_targets(self):
        with mock.patch.object(DynamixelDriver, "_initialize_with_retries", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Failed to initialize"):
                DynamixelRobot(range(1, 8), real=True, port="/missing/gello")

    def test_busy_port_does_not_kill_owner_or_change_permissions(self):
        with mock.patch.object(DynamixelDriver, "_check_port_availability", return_value=False), \
                mock.patch("gello.dynamixel.driver.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                DynamixelDriver([1], port="/busy/gello", use_fake_fallback=False)
            run.assert_not_called()

    def test_readings_are_copied_and_converted(self):
        driver = self.reader()
        expected = driver._joint_angles.copy() / 2048.0 * np.pi
        np.testing.assert_allclose(driver.get_joints(), expected)
        positions, velocities = driver.get_positions_and_velocities()
        np.testing.assert_allclose(positions, expected)
        np.testing.assert_allclose(velocities, np.arange(8) * 2 * 0.229 * 2 * np.pi / 60)
        positions[:] = 0
        np.testing.assert_allclose(driver.get_joints(), expected)

    def test_stale_readings_are_rejected(self):
        driver = self.reader()
        driver._last_read_time -= 2
        for read in (driver.get_joints, driver.get_positions_and_velocities):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                read()

    def test_missing_initial_feedback_has_bounded_wait(self):
        driver = self.reader()
        driver._joint_angles = driver._velocities = None
        with mock.patch("gello.dynamixel.driver.time.monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(RuntimeError, "No GELLO servo feedback"):
                driver.get_joints()

    def test_close_before_initialization_and_repeated_close(self):
        driver = self.reader()
        driver.close()
        driver.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            driver.get_joints()

    def hardware_mocks(self, stack):
        module = "gello.dynamixel.driver."
        port = stack.enter_context(mock.patch(module + "PortHandler")).return_value
        packet = stack.enter_context(mock.patch(module + "PacketHandler")).return_value
        stack.enter_context(mock.patch(module + "GroupSyncRead"))
        stack.enter_context(mock.patch(module + "GroupSyncWrite"))
        stack.enter_context(mock.patch(module + "time.sleep"))
        stack.enter_context(mock.patch.object(DynamixelDriver, "_check_port_availability", return_value=True))
        reader = stack.enter_context(mock.patch.object(DynamixelDriver, "_start_reading_thread"))
        port.is_open = True
        packet.getTxRxResult.return_value = "There is no status packet!"
        return port, packet, reader

    def test_torque_timeout_retries_without_reporting_success_or_starting_reader(self):
        with contextlib.ExitStack() as stack:
            port, packet, reader = self.hardware_mocks(stack)
            packet.write1ByteTxRx.return_value = (COMM_RX_TIMEOUT, 0)
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            with self.assertRaisesRegex(RuntimeError, "ID 1.*comm=-3001"):
                DynamixelDriver(range(1, 9), port="/test/gello", use_fake_fallback=False)
            self.assertEqual(packet.write1ByteTxRx.call_count, 3)
            self.assertEqual(port.closePort.call_count, 3)
            reader.assert_not_called()
            self.assertNotIn("Successfully initialized", output.getvalue())
            self.assertNotIn("Using fake", output.getvalue())

    def test_transient_torque_timeout_recovers_before_starting_reader(self):
        with contextlib.ExitStack() as stack:
            port, packet, reader = self.hardware_mocks(stack)
            packet.write1ByteTxRx.side_effect = [(COMM_RX_TIMEOUT, 0)] + [(0, 0)] * 8
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            driver = DynamixelDriver(range(1, 9), port="/test/gello", use_fake_fallback=False)
            self.assertEqual(packet.write1ByteTxRx.call_count, 9)
            self.assertEqual(port.closePort.call_count, 1)
            reader.assert_called_once()
            self.assertEqual(output.getvalue().count("Successfully initialized"), 1)
            driver.close()
            self.assertEqual(port.closePort.call_count, 2)

    def test_robot_does_not_repeat_torque_command_after_driver_initialization(self):
        with mock.patch("gello.dynamixel.driver.DynamixelDriver") as driver:
            robot = DynamixelRobot(range(1, 8), real=True, port="/test/gello")
            driver.return_value.set_torque_mode.assert_not_called()
            robot.close()

    def test_close_after_failed_port_open(self):
        driver = self.reader()
        driver._portHandler = mock.Mock(is_open=False)
        driver.close()
        driver._portHandler.closePort.assert_not_called()


if __name__ == "__main__":
    unittest.main()
