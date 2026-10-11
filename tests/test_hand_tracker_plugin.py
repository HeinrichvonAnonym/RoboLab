"""Run with: python -m unittest discover -s tests -p test_hand_tracker_plugin.py"""

from dataclasses import replace
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "src/plugins/python/hand_tracker_plugin.py"
SPEC = importlib.util.spec_from_file_location("hand_tracker_under_test", SCRIPT)
ht = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ht
SPEC.loader.exec_module(ht)
ImageMessage = ht.image_message_type()


def payload(marker=0, width=2, height=2):
    return ImageMessage(width=width, height=height, channels=4, step=width * 4,
                        image=bytes([marker]) * width * height * 4).SerializeToString()


def sample(data):
    return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda: data))


def wait_until(check, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for tracker")


def raw_result(count):
    points = [SimpleNamespace(x=0.2 + i * 0.02, y=0.4, z=i * 0.001) for i in range(21)]
    return SimpleNamespace(
        hand_landmarks=[points] * count, hand_world_landmarks=[points] * count,
        handedness=[[SimpleNamespace(category_name=label, score=0.9)]
                    for label in ("Left", "Right")[:count]],
    )


class FakeDetector:
    instances = []

    def __init__(self, config, stop_event):
        self.timestamps = []
        self.markers = []
        self.closed = False
        self.instances.append(self)

    def process(self, rgb, timestamp_ms):
        assert threading.current_thread() is not threading.main_thread()
        self.timestamps.append(timestamp_ms)
        self.markers.append(int(rgb[0, 0, 0]))
        return ht.extract_hands(raw_result(0 if rgb[0, 0, 0] == 0 else 1))

    def close(self):
        self.closed = True


class TrackerTests(unittest.TestCase):
    def test_default_config(self):
        config = ht.load_config()
        self.assertEqual(config.rgb_topic, "kinec2/rgb")
        self.assertEqual(config.input_color_format, "BGRX")
        self.assertEqual(config.num_hands, 2)
        self.assertTrue(config.enable_viewer)
        self.assertTrue(config.model_path.is_absolute())

    def test_config_paths_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.yaml"
            config.write_text("model_path: model.task\nenable_viewer: false\n")
            self.assertEqual(ht.load_config(config).model_path, Path(folder) / "model.task")
            for invalid in ("fps: 0", "fps: .nan", "num_hands: true",
                            'enable_viewer: "false"', "input_color_format: RGBA",
                            "min_tracking_confidence: 2", "zenoh: {connect: abc}"):
                with self.subTest(invalid=invalid):
                    config.write_text(invalid)
                    with self.assertRaises((ValueError, TypeError)):
                        ht.load_config(config)

    def test_all_color_formats_with_row_padding(self):
        for color, pixel in (("BGRX", [30, 20, 10, 255]), ("RGBX", [10, 20, 30, 255]),
                             ("BGR", [30, 20, 10]), ("RGB", [10, 20, 30])):
            with self.subTest(color=color):
                row = bytes(pixel * 2 + [99, 99, 99])
                message = ImageMessage(width=2, height=2, channels=len(pixel), step=len(row), image=row * 2)
                rgb = ht.decode_rgb(message.SerializeToString(), color)
                np.testing.assert_array_equal(rgb, np.tile([10, 20, 30], (2, 2, 1)))
                self.assertTrue(rgb.flags.c_contiguous)

    def test_invalid_messages(self):
        for fields in ({"width": 0}, {"height": -1}, {"channels": 3}, {"step": 7},
                       {"image": b"short"}, {"image": b"\0" * 17}):
            message = ImageMessage(width=2, height=2, channels=4, step=8, image=b"\0" * 16)
            for name, value in fields.items():
                setattr(message, name, value)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                ht.decode_rgb(message.SerializeToString(), "BGRX")
        with self.assertRaises(Exception):
            ht.decode_rgb(b"invalid protobuf", "BGRX")

    def test_zero_one_two_hands(self):
        for count in range(3):
            hands = ht.extract_hands(raw_result(count))
            self.assertEqual(len(hands), count)
            for hand in hands:
                self.assertEqual(len(hand.landmarks), 21)
                self.assertEqual(len(hand.world_landmarks), 21)
                self.assertEqual(hand.score, 0.9)

    def test_atomic_model_download_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            config = replace(ht.TrackerConfig(), model_path=Path(folder) / "hand.task")
            with mock.patch.object(ht.urllib.request, "urlopen", return_value=io.BytesIO(b"test model")) as download:
                self.assertEqual(ht.ensure_model(config, threading.Event()), config.model_path)
                self.assertEqual(config.model_path.read_bytes(), b"test model")
                self.assertEqual(list(Path(folder).glob("*.part")), [])
                ht.ensure_model(config, threading.Event())
                download.assert_called_once()

    def test_model_download_failure_and_cancellation(self):
        with tempfile.TemporaryDirectory() as folder:
            config = replace(ht.TrackerConfig(), model_path=Path(folder) / "hand.task")
            with mock.patch.object(ht.urllib.request, "urlopen", side_effect=OSError("offline")):
                with self.assertRaisesRegex(RuntimeError, "Cannot download"):
                    ht.ensure_model(config, threading.Event())
            self.assertFalse(config.model_path.exists())
            self.assertEqual(list(Path(folder).glob("*.part")), [])
            with self.assertRaises(FileNotFoundError):
                ht.ensure_model(replace(config, auto_download_model=False), threading.Event())
            stopped = threading.Event()
            stopped.set()
            with self.assertRaises(InterruptedError):
                ht.ensure_model(config, stopped)

    def test_incomplete_download_is_not_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            config = replace(ht.TrackerConfig(), model_path=Path(folder) / "hand.task")
            response = io.BytesIO(b"truncated")
            response.headers = {"Content-Length": "100"}
            with mock.patch.object(ht.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    ht.ensure_model(config, threading.Event())
            self.assertFalse(config.model_path.exists())
            self.assertEqual(list(Path(folder).glob("*.part")), [])

    def test_latest_frame_throttle_timestamp_and_empty_result(self):
        plugin = ht.HandTrackerPlugin(replace(ht.TrackerConfig(), enable_viewer=False, fps=10))
        worker = threading.Thread(target=plugin._process_frames)
        with mock.patch.object(ht, "MediaPipeDetector", FakeDetector):
            worker.start()
            try:
                # Deliberately identical receive timestamps: VIDEO timestamps must still increase.
                with mock.patch.object(ht.time, "monotonic_ns", return_value=1000000000):
                    plugin._on_frame(sample(payload(1)))
                    wait_until(lambda: plugin.processed_frames == 1)
                    for marker in range(2, 52):
                        plugin._on_frame(sample(payload(marker)))
                    wait_until(lambda: plugin.processed_frames == 2)
                    self.assertEqual(int(plugin.latest_result.rgb[0, 0, 0]), 51)
                    self.assertGreaterEqual(plugin.replaced_frames, 49)
                    self.assertEqual(len(plugin.latest_result.hands), 1)
                    self.assertFalse(plugin.latest_result.rgb.flags.writeable)
                    plugin._on_frame(sample(payload(0)))
                    wait_until(lambda: plugin.processed_frames == 3)
                    self.assertEqual(plugin.latest_result.hands, ())
                backend = FakeDetector.instances[-1]
                self.assertEqual(backend.markers, [1, 51, 0])
                self.assertEqual(backend.timestamps, [1000, 1001, 1002])
                plugin._on_frame(sample(b"bad"))
                wait_until(lambda: plugin.invalid_frames == 1)
                self.assertIsNone(plugin._worker_error)
            finally:
                plugin.stop()
                worker.join(timeout=3)
            self.assertFalse(worker.is_alive())
            self.assertTrue(backend.closed)

    def test_render_clears_old_landmarks_and_preserves_matching_image(self):
        import cv2
        rgb = np.full((200, 400, 3), [10, 20, 30], dtype=np.uint8)
        hands = ht.extract_hands(raw_result(1))
        result = ht.TrackerResult(1, time.monotonic(), rgb, hands)
        with_hand = ht.render_result(result, cv2, 400)
        empty = ht.render_result(replace(result, hands=()), cv2, 400)
        x, y = round(hands[0].landmarks[0][0] * 399), round(hands[0].landmarks[0][1] * 199)
        self.assertFalse(np.array_equal(with_hand[y, x], empty[y, x]))
        np.testing.assert_array_equal(empty[y, x], [30, 20, 10])
        np.testing.assert_array_equal(rgb[y, x], [10, 20, 30])

    def test_gui_calls_are_on_main_thread_and_window_close_is_normal(self):
        for close_by_error in (False, True):
            plugin = ht.HandTrackerPlugin(ht.TrackerConfig())
            plugin._latest_result = ht.TrackerResult(1, time.monotonic(), np.zeros((100, 200, 3), np.uint8), ())
            cv = mock.MagicMock()
            cv.error = RuntimeError
            cv.cvtColor.return_value = np.zeros((100, 200, 3), np.uint8)
            calls = []
            def gui_call(name, return_value=None):
                def call(*args):
                    self.assertIs(threading.current_thread(), threading.main_thread())
                    calls.append(name)
                    return return_value
                return call
            for name in ("namedWindow", "imshow", "destroyAllWindows"):
                getattr(cv, name).side_effect = gui_call(name)
            cv.waitKey.side_effect = gui_call("waitKey", -1 if close_by_error else ord("q"))
            if close_by_error:
                cv.getWindowProperty.side_effect = RuntimeError("window closed")
            else:
                cv.getWindowProperty.side_effect = gui_call("getWindowProperty", 1)
            plugin._main_loop(cv)
            self.assertTrue(plugin._stop_event.is_set())
            self.assertIn("destroyAllWindows", calls)

    def test_viewer_rejects_background_thread(self):
        errors = []
        def run():
            try:
                ht.HandTrackerPlugin(ht.TrackerConfig()).run()
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=3)
        self.assertTrue(errors and "main thread" in errors[0])


if __name__ == "__main__":
    unittest.main()
