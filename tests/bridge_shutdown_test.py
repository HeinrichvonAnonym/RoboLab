"""Exercise the real app's signal ownership without connecting any hardware."""
import pathlib
import signal
import subprocess
import sys
import tempfile
import time

binary, library = sys.argv[1:]
with tempfile.TemporaryDirectory(prefix="robolab-shutdown-") as temporary:
    root = pathlib.Path(temporary)
    config = root / "bridge.yaml"
    config.write_text("roboLab_2_ros_topics: []\nros_2_roboLab_topics: []\n")
    bringup = root / "bringup.yaml"
    bringup.write_text(f"plugins:\n  - name: ros_bridge_plugin\n    config: {config}\n    lib: {library}\n")
    for sig in (signal.SIGINT, signal.SIGTERM):
        with (root / "output.log").open("w+") as log:
            process = subprocess.Popen(["stdbuf", "-oL", binary, str(bringup)], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    log.seek(0)
                    output = log.read()
                    if "main: plugins initialized" in output:
                        break
                    if process.poll() is not None:
                        raise AssertionError(output)
                    time.sleep(0.05)
                else:
                    raise AssertionError("startup timed out: " + output)
                process.send_signal(sig)
                assert process.wait(timeout=10) == 0
                log.seek(0)
                output = log.read()
                assert "main: shutdown complete" in output, output
                print(f"{sig.name}: orderly shutdown verified")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
