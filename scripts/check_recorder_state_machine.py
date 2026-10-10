#!/usr/bin/env python3
"""Exercise recorder episode boundaries using isolated Zenoh topics and temporary H5 files.

Run after building recorder_plugin, with the roboLab Python environment:
    python scripts/check_recorder_state_machine.py
No robot or camera plugins are started.
"""

import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import h5py
import zenoh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
from robo_lab.proto_gen import franka_pb2, kinect_pb2


def wait_until(predicate, message, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(message)


def check():
    prefix = "recorder_check/" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="robolab_recorder_state_") as folder:
        root = Path(folder)
        data = root / "data"
        config = root / "recorder.yaml"
        config.write_text(f"""topic: [{prefix}/state, {prefix}/image]
topic_proto:
  {prefix}/state: franka.RobotObservation
  {prefix}/image: kinect.rgbImage
record_frequency: 1000
state_machine_topic: {prefix}/control
recording_state: inference
anchor_topic: {prefix}/state
flush_anchor_count: 2
flush_buffer_mb: 16
next_record_topic: {prefix}/checkpoint
data_dir: {data}
""")
        bringup = root / "bringup.yaml"
        bringup.write_text(f"""plugins:
  - name: recorder_plugin
    config: {config}
    lib: {REPO}/build/plugins/librecorder_plugin.so
""")
        log_path = root / "run.log"
        session = zenoh.open(zenoh.Config())
        peak_rss_kb = 0
        child = None

        def control(state):
            session.put(prefix + "/control", state.encode())
            time.sleep(0.15)

        def sample(marker, with_anchor=True):
            nonlocal peak_rss_kb
            if with_anchor:
                state = franka_pb2.RobotObservation(type=2, sequence=marker)
                state.force.fx = 1.25
                state.pose_ee.pos.x = 0.3
                session.put(prefix + "/state", state.SerializeToString())
            image = kinect_pb2.rgbImage(
                width=2048, height=1024, channels=4, step=8192, type=1,
                image=bytes([marker]) * 8388608,
            )
            session.put(prefix + "/image", image.SerializeToString())
            time.sleep(0.15)
            status = Path(f"/proc/{child.pid}/status").read_text()
            rss = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
            peak_rss_kb = max(peak_rss_kb, int(rss.split()[1]))

        def final_files():
            return sorted(data.glob("*.h5"))

        def temporary_files():
            return list(data.glob("*.h5.inprogress"))

        try:
            with log_path.open("w") as log:
                child = subprocess.Popen(
                    [str(REPO / "build/apps/robo_lab_main"), str(bringup)],
                    cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                )
                # Allow the isolated Zenoh peers to discover each other.
                time.sleep(2)
                control("init")
                sample(90)
                control("gohome")
                sample(91)
                control("standby")
                sample(92)
                session.put(prefix + "/checkpoint", b"next_record")
                time.sleep(0.2)
                assert not final_files() and not temporary_files(), "idle states recorded data"

                control("inference")
                # Images immediately after entering running are accepted without an anchor.
                sample(10, with_anchor=False)
                sample(11, with_anchor=False)
                wait_until(lambda: len(temporary_files()) == 1, "byte threshold did not checkpoint")
                assert not final_files(), "running published a finalized H5 file"
                session.put(prefix + "/checkpoint", b"next_record")
                time.sleep(0.3)
                control("inference")  # Duplicate status must not split/reset the episode.
                sample(12)
                control("standby")
                wait_until(lambda: len(final_files()) == 1, "standby did not finalize episode 1")
                sample(93)
                control("standby")
                assert len(final_files()) == 1 and not temporary_files()

                control("inference")
                sample(20)
                sample(21)
                wait_until(lambda: len(temporary_files()) == 1, "anchor did not checkpoint")
                assert len(final_files()) == 1, "checkpoint split the running episode"
                control("standby")
                wait_until(lambda: len(final_files()) == 2, "standby did not finalize episode 2")
                sample(94)

                # An empty running episode yields one file, but repeated standby yields none.
                control("inference")
                control("standby")
                wait_until(lambda: len(final_files()) == 3, "empty episode not finalized")
                control("standby")
                assert len(final_files()) == 3

                control("inference")
                sample(40)
                child.send_signal(signal.SIGTERM)
                child.wait(timeout=15)

            assert child.returncode == 0, f"recorder exit code {child.returncode}"
            assert len(final_files()) == 4 and not temporary_files()
            expected = [([10, 11, 12], [12]), ([20, 21], [20, 21]), ([], []), ([40], [40])]
            frame_counts = []
            for filename, (markers, sequences) in zip(final_files(), expected):
                with h5py.File(filename, "r") as h5:
                    if not markers:
                        assert len(h5) == 0
                        frame_counts.append(0)
                        continue
                    image = h5[(prefix + "/image").replace("/", "_")]
                    state = h5[(prefix + "/state").replace("/", "_")]
                    assert state["sequence"][:].tolist() == sequences
                    assert (state["force"][:, 0] == 1.25).all()
                    assert (state["pose_ee"][:, 0] == 0.3).all()
                    assert len(image["timestamps_ns"]) == len(markers)
                    assert image["image_data"].chunks == (1048576,)
                    offsets = image["image_offsets"][:]
                    lengths = image["image_lengths"][:]
                    assert len(offsets) == len(markers)
                    for marker, offset, length in zip(markers, offsets, lengths):
                        assert length == 8388608
                        frame = image["image_data"][int(offset):int(offset + length)].view("uint8")
                        assert (frame == marker).all(), "corrupt or out-of-episode image"
                    frame_counts.append(len(markers))
            print(json.dumps({
                "result": "passed", "files": 4, "frames_per_file": frame_counts,
                "peak_rss_mb": round(peak_rss_kb / 1024, 1),
                "checks": "idle exclusion, byte/anchor checkpoints, F1, duplicate states, "
                          "two standby boundaries, empty episode, shutdown, full image payloads",
            }))
        except Exception:
            if log_path.exists():
                print(log_path.read_text(), file=sys.stderr)
            raise
        finally:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            session.close()


if __name__ == "__main__":
    check()
