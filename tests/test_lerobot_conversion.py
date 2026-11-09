"""Hardware-free checks: .venv-lerobot/bin/python -m unittest discover -s tests -p test_lerobot_conversion.py"""
import copy
import json
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lerobot_tools.core import (CAMERAS, TZ, Config, Recording, differentiate, discover,
    empty_plan, normalize_plan, plan_digest, scan)
from lerobot_tools.export import export_dataset

NAME = "recording_20261013_211200_1791897120000000000.h5"
BASE = int(datetime(2026, 10, 13, 21, 12, tzinfo=TZ).timestamp()) * 10**9


def make_recording(path, *, gap=False, missing=False, corrupt=False):
    """Tiny RGB fixtures with asynchronous streams and recognizable colors."""
    samples = {
        "franka_command": np.arange(0, 2.001, .04),
        "franka_state": np.arange(.005, 2, .041),
        "kinec2_rgb": np.arange(.01, 2, 1 / 15),
        "realsense_rgb": np.arange(.025, 2, 1 / 14),
    }
    if gap:
        samples = {k: t[(t < .6) | (t > 1.2)] for k, t in samples.items()}
    with h5py.File(path, "w") as f:
        for name, t in samples.items():
            if name == "realsense_rgb" and missing:
                continue
            group = f.create_group(name)
            ts = BASE + np.rint(t * 1e9).astype(np.int64)
            if corrupt:
                ts[-1] = 0
            group["timestamps_ns"] = ts
            n = len(t)
            if name.startswith("franka"):
                positions = t[:, None] * np.arange(1, 9, dtype=float)[None, :]
                positions[:, 7] = (t >= 1).astype(float)
                group["joints_position"] = positions
                velocities = np.zeros_like(positions) if name == "franka_command" else np.tile(np.arange(1, 9), (n, 1))
                velocities[:, 7] = 0
                group["joints_velocity"] = velocities
            else:
                c = 4 if name == "kinec2_rgb" else 3
                rgb = np.empty((24, 32, c), dtype=np.uint8)
                rgb[:] = [30, 20, 200, 255] if c == 4 else [20, 200, 30]
                count = rgb.size
                group["width"] = np.full(n, 32, dtype=np.int32)
                group["height"] = np.full(n, 24, dtype=np.int32)
                group["channels"] = np.full(n, c, dtype=np.int32)
                group["format_type"] = np.full(n, 1 if c == 4 else 5, dtype=np.int32)
                group["image_lengths"] = np.full(n, count, dtype=np.uint64)
                group["image_offsets"] = np.arange(n, dtype=np.uint64) * count
                group["image_data"] = np.tile(rgb.ravel(), n).view(np.int8)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.source = self.data / NAME
        make_recording(self.source)
        self.config = Config(input_dir=str(self.data), output=str(self.root / "dataset"))


class CoreTests(Fixture):
    def test_discovery_is_fixed_shanghai_cutoff_and_complete_h5_only(self):
        (self.data / "recording_20261013_205959_1791896399000000000.h5").touch()
        (self.data / "recording_20261014_120000_1791950400000000000.h5.inprogress").touch()
        self.assertEqual(discover(self.config), [self.source])
        self.assertEqual(discover(replace(self.config, since="2026-10-13T21:00:00")), [self.source])

    def test_actual_color_order_and_asynchronous_alignment(self):
        rec = Recording(self.source, self.config)
        t = (rec.grid - BASE) / 1e9
        command_t = (rec.timestamps["franka_command"] - BASE) / 1e9
        self.assertTrue(np.all(command_t[rec.command_indices] <= t))
        np.testing.assert_allclose(rec.aligned["observation.state"][:, 0], t, atol=1e-7)
        np.testing.assert_allclose(rec.aligned["observation.velocity"][:, :7], np.tile(np.arange(1, 8), (len(t), 1)))
        np.testing.assert_allclose(rec.aligned["action.velocity"][:, 0], 1, atol=1e-6)
        self.assertGreater(np.max(rec.aligned["action.velocity"][:, 7]), 1)
        with h5py.File(self.source, "r") as f:
            np.testing.assert_array_equal(rec.read_rgb(f, "kinec2", 0)[0, 0], [200, 20, 30])
            np.testing.assert_array_equal(rec.read_rgb(f, "realsense", 0)[0, 0], [20, 200, 30])

    def test_nonuniform_difference_does_not_cross_acquisition_gap(self):
        t = np.array([0, 40_000_000, 110_000_000, 800_000_000, 850_000_000])
        q = (t / 1e9 * 2.5)[:, None]
        q[3:] += 40
        result = differentiate(q, t, 200_000_000)
        np.testing.assert_allclose(result, 2.5, atol=1e-10)
        np.testing.assert_array_equal(differentiate(np.array([[3.0]]), np.array([0]), 100), [[0]])

    def test_gap_holds_previous_action_state_image_and_velocity(self):
        make_recording(self.source, gap=True)
        rec = Recording(self.source, self.config)
        i = np.flatnonzero((rec.grid - BASE > 800_000_000) & (rec.grid - BASE < 1_000_000_000))[0]
        for camera, group in CAMERAS.items():
            self.assertLess(rec.timestamps[group][rec.camera_indices[camera][i]], rec.grid[i])
        state_ts = rec.timestamps["franka_state"]
        j = np.searchsorted(state_ts, rec.grid[i], side="right") - 1
        np.testing.assert_allclose(rec.aligned["observation.state"][i], rec.positions["franka_state/joints_position"][j], atol=1e-6)
        self.assertGreater(rec.lag["franka_state"][i], self.config.max_gap_ms)
        self.assertTrue(rec.gaps["franka_state"])

    def test_cuts_are_inclusive_split_and_never_change_retained_velocities(self):
        rec = Recording(self.source, self.config)
        original = {k: v.copy() for k, v in rec.aligned.items()}
        times = rec.seconds
        plan = normalize_plan({"version": 1, "recordings": {NAME: {"exclude": [
            [float(times[0]), float(times[1])], [float(times[6]), float(times[9])],
            [float(times[8]), float(times[11])], [float(times[-2]), float(times[-1])]]}}})
        segments = rec.segments(plan)
        self.assertEqual([s.tolist() for s in segments], [list(range(2, 6)), list(range(12, len(times) - 2))])
        for key in original:
            np.testing.assert_array_equal(rec.aligned[key], original[key])
        plan["recordings"][NAME]["drop"] = True
        self.assertEqual(rec.segments(plan), [])

    def test_invalid_sources_are_reported_and_plans_validated(self):
        missing = self.data / NAME.replace("211200", "211201")
        make_recording(missing, missing=True)
        corrupt = self.data / NAME.replace("211200", "211202")
        make_recording(corrupt, corrupt=True)
        entries = scan(self.config)
        self.assertEqual(sum(e["valid"] for e in entries), 1)
        self.assertIn("Missing stream", entries[1]["error"])
        self.assertIn("timestamps", entries[2]["error"])
        for pair in ([2, 1], [-1, 2], [0, float("inf")], [False, 2]):
            with self.assertRaises(ValueError):
                normalize_plan({"version": 1, "recordings": {NAME: {"exclude": [pair]}}})
        for name in ("../foo.h5", "/tmp/file.h5"):
            with self.assertRaises(ValueError):
                normalize_plan({"version": 1, "recordings": {name: {}}})
        with self.assertRaises(ValueError):
            replace(self.config, fps=0).validate()

    def test_source_fingerprint_protects_saved_edits(self):
        rec = Recording(self.source, self.config)
        self.assertIsInstance(rec.fingerprint["mtime_ns"], str)
        # Browser JSON round-trip must preserve all nanosecond digits.
        self.assertEqual(int(rec.fingerprint["mtime_ns"]), self.source.stat().st_mtime_ns)
        plan = {"version": 1, "recordings": {NAME: {"fingerprint": {"size": 1, "mtime_ns": 1}}}}
        with self.assertRaisesRegex(ValueError, "different source"):
            rec.segments(plan)


class ExportTests(Fixture):
    def test_official_roundtrip_repeat_export_and_atomic_failure(self):
        import pyarrow.parquet as pq
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        rec = Recording(self.source, self.config)
        times = rec.seconds
        plan = normalize_plan({"version": 1, "recordings": {NAME: {"exclude": [[float(times[5]), float(times[9])]]}}})
        missing = self.data / NAME.replace("211200", "211201")
        make_recording(missing, missing=True)
        result = export_dataset(self.config, plan)
        output = Path(self.config.output)
        self.assertTrue(output.is_symlink())
        first_target = output.resolve()
        self.assertEqual(result["total_episodes"], 2)
        self.assertEqual(len(result["skipped"]), 1)
        source_grid = pq.read_table(output / "provenance.parquet")["source_grid_index"].to_pylist()
        expected_indices = [i for s in rec.segments(plan) for i in s]
        self.assertEqual(source_grid, expected_indices)
        ds = LeRobotDataset(self.config.repo_id, root=output, video_backend="pyav")
        expected_count = len(expected_indices)
        self.assertEqual(len(ds), expected_count)
        for n in [0, 4, 5, expected_count - 1]:
            item = ds[n]
            np.testing.assert_allclose(item["action.velocity"].numpy(), rec.aligned["action.velocity"][expected_indices[n]])
            self.assertGreater(float(item["observation.images.kinec2"][0].mean()), .65)
            self.assertGreater(float(item["observation.images.realsense"][1].mean()), .65)
        before = (output / "export_report.json").read_bytes()
        with mock.patch("lerobot_tools.export.validate_dataset", side_effect=ValueError("injected validation error")):
            with self.assertRaisesRegex(ValueError, "injected"):
                export_dataset(self.config, empty_plan(), max_frames=3)
        self.assertEqual(output.resolve(), first_target)
        self.assertEqual((output / "export_report.json").read_bytes(), before)
        deleted = {"version": 1, "recordings": {NAME: {"drop": True}}}
        with self.assertRaisesRegex(ValueError, "No frames remain"):
            export_dataset(self.config, deleted)
        self.assertEqual(output.resolve(), first_target)
        result2 = export_dataset(self.config, empty_plan(), max_frames=3)
        self.assertNotEqual(output.resolve(), first_target)
        self.assertTrue(first_target.exists())
        self.assertEqual(result2["total_frames"], 3)
        self.assertFalse((output.parent / ".dataset.export.lock").exists())

    def test_existing_directory_and_unknown_edits_are_never_overwritten(self):
        output = Path(self.config.output)
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            export_dataset(self.config, empty_plan())
        self.assertEqual(marker.read_text(), "keep")
        with self.assertRaisesRegex(ValueError, "unselected/missing"):
            export_dataset(self.config, {"version": 1, "recordings": {NAME.replace("211200", "211205"): {}}})


class ServerTests(Fixture):
    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient
        from lerobot_tools.server import create_app
        self.clips = self.root / "clips.json"
        self.app = create_app(self.config, self.clips)
        self.client = TestClient(self.app)

    def test_player_plan_persistence_conflicts_and_frame_consistency(self):
        client = self.client
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/static/app.js").status_code, 200)
        timeline = client.get(f"/api/recordings/{NAME}/timeline").json()
        i = 5
        frame = client.get(f"/api/recordings/{NAME}/frame", params={"index": i}).json()
        self.assertEqual(frame["seconds"], timeline["seconds"][i])
        self.assertEqual(frame["action"], timeline["action"][i])
        self.assertTrue(frame["images"]["kinec2"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(client.get(f"/api/recordings/{NAME}/frame?index=-1").status_code, 400)
        saved = client.get("/api/plan").json()
        plan = copy.deepcopy(saved["plan"])
        plan["recordings"][NAME] = {"exclude": [[timeline["seconds"][3], timeline["seconds"][6]]]}
        payload = {"plan": plan, "base_revision": saved["revision"]}
        response = client.put("/api/plan", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        updated = response.json()
        self.assertIn("fingerprint", updated["plan"]["recordings"][NAME])
        self.assertEqual(json.loads(self.clips.read_text()), updated["plan"])
        self.assertEqual(client.put("/api/plan", json=payload).status_code, 409)
        self.assertEqual(client.post("/api/export", json={"revision": saved["revision"]}).status_code, 409)
        self.assertEqual(client.get("/api/export").json()["stale"], True)

    def test_export_uses_snapshot_even_while_saved_plan_changes(self):
        started, finish = threading.Event(), threading.Event()
        captured = []
        def fake_export(config, plan, progress):
            captured.append(plan)
            started.set()
            finish.wait(5)
            return {"skipped": []}
        saved = self.client.get("/api/plan").json()
        with mock.patch("lerobot_tools.server.export_dataset", side_effect=fake_export):
            response = self.client.post("/api/export", json={"revision": saved["revision"]})
            self.assertEqual(response.status_code, 202)
            self.assertTrue(started.wait(2))
            try:
                self.assertEqual(self.client.post("/api/export", json={"revision": saved["revision"]}).status_code, 409)
                plan = {"version": 1, "recordings": {NAME: {"drop": True}}}
                response = self.client.put("/api/plan", json={"plan": plan, "base_revision": saved["revision"]})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(captured[0], empty_plan())
            finally:
                finish.set()
                self.app.state.service.worker.join(5)
        self.assertEqual(self.client.get("/api/export").json()["job"]["phase"], "complete")


if __name__ == "__main__":
    unittest.main()
