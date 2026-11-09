"""Optional real Chromium checks; see scripts/README_lerobot.md for setup."""
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lerobot_tools.core import Config
from lerobot_tools.server import create_app
from test_lerobot_conversion import NAME, make_recording


class BrowserTests(unittest.TestCase):
    def test_play_pause_step_clips_save_reload_and_export(self):
        import uvicorn
        from playwright.sync_api import sync_playwright

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            make_recording(data / NAME)
            config = Config(input_dir=str(data), output=str(root / "dataset"))
            clips = root / "clips.json"
            app = create_app(config, clips)

            @app.get("/api/player")
            def player():
                return {"speed": 1}

            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
            thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
            thread.start()
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue(server.started)
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
                    page = browser.new_page(viewport={"width": 1440, "height": 1200})
                    errors = []
                    page.on("pageerror", lambda exc: errors.append(str(exc)))
                    page.goto(f"http://127.0.0.1:{port}")
                    page.wait_for_function("document.querySelector('#kinec2').naturalWidth === 32 && document.querySelector('#realsense').naturalWidth === 32")
                    self.assertEqual(page.locator("#charts canvas").count(), 8)
                    page.click("#next")
                    page.wait_for_function("document.querySelector('#seek').value === '1'")
                    page.fill("#speed", "2")
                    page.locator("#speed").dispatch_event("change")
                    page.click("#play")
                    page.wait_for_function("Number(document.querySelector('#seek').value) > 3")
                    page.click("#play")
                    self.assertEqual(page.locator("#play").inner_text(), "播放")
                    paused = page.locator("#seek").input_value()
                    page.wait_for_timeout(180)
                    self.assertEqual(page.locator("#seek").input_value(), paused)

                    def seek(index):
                        page.locator("#seek").evaluate("(el, index) => {el.value = index; el.dispatchEvent(new Event('input'));}", str(index))
                        page.wait_for_function("index => document.querySelector('#frame').textContent.startsWith('帧 ' + (index + 1) + ' /')", arg=index)

                    seek(5)
                    page.click("#markStart")
                    self.assertEqual(page.locator('[data-boundary="start"]').count(), 9)
                    self.assertEqual(page.locator('[data-boundary="end"]').count(), 0)
                    start = page.locator("#start").input_value()
                    seek(9)
                    page.click("#markEnd")
                    self.assertEqual(page.locator('[data-boundary="end"]').count(), 9)
                    end = page.locator("#end").input_value()
                    # Typed bounds and clearing an endpoint update every plot immediately.
                    previous_position = page.locator('#ruler [data-boundary="start"]').evaluate("el => el.style.left")
                    page.fill("#start", "0.5")
                    positions = page.locator('[data-boundary="start"]').evaluate_all("els => els.map(el => el.style.left)")
                    self.assertEqual(len(set(positions)), 1)
                    self.assertNotEqual(positions[0], previous_position)
                    page.fill("#start", "0")
                    self.assertEqual(page.locator('#ruler [data-boundary="start"]').evaluate("el => el.style.left"), "0%")
                    page.fill("#start", start)
                    page.fill("#end", "")
                    self.assertEqual(page.locator('[data-boundary="end"]').count(), 0)
                    page.fill("#end", end)
                    self.assertEqual(page.locator('[data-boundary="end"]').count(), 9)
                    self.assertEqual(page.locator("#saved").inner_text(), "已保存")
                    self.assertEqual(page.locator(".clip").count(), 0)
                    preview = Path(__file__).resolve().parents[1] / "outputs/lerobot_review/selection_preview.png"
                    preview.parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(preview), full_page=True)
                    page.click("#addClip")
                    self.assertEqual(page.locator(".clip").count(), 1)
                    self.assertIn("未保存", page.locator("#saved").inner_text())
                    self.assertTrue(page.locator("#export").is_disabled())
                    page.click("#undo")
                    self.assertEqual(page.locator(".clip").count(), 0)
                    page.click("#addClip")
                    with page.expect_response(lambda response: response.url.endswith("/api/plan") and response.request.method == "PUT") as saved_response:
                        page.click("#save")
                    self.assertEqual(saved_response.value.status, 200, saved_response.value.text())
                    page.wait_for_function("document.querySelector('#saved').textContent === '已保存'")
                    self.assertTrue(clips.exists())
                    saved = json.loads(clips.read_text())
                    self.assertEqual(len(saved["recordings"][NAME]["exclude"]), 1)
                    self.assertIsInstance(saved["recordings"][NAME]["fingerprint"]["mtime_ns"], str)
                    page.reload()
                    page.wait_for_function("document.querySelector('#kinec2').naturalWidth === 32")
                    self.assertEqual(page.locator(".clip").count(), 1)
                    page.click("#drop")
                    self.assertIn("导出 0 个 episode", page.locator("#clipInfo").inner_text())
                    page.click("#undo")
                    self.assertIn("导出 2 个 episode", page.locator("#clipInfo").inner_text())
                    page.click("#export")
                    page.wait_for_function("document.querySelector('#exportStatus').textContent.includes('导出完成')", timeout=30000)
                    self.assertIn("2 episodes", page.locator("#exportStatus").inner_text())
                    report = json.loads((root / "dataset/export_report.json").read_text())
                    self.assertEqual(report["total_episodes"], 2)
                    self.assertEqual(report["total_frames"], 24)
                    screenshot = Path(__file__).resolve().parents[1] / "outputs/lerobot_review/browser_smoke.png"
                    screenshot.parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(screenshot), full_page=True)
                    previous_target = (root / "dataset").resolve()
                    page.click("#drop")
                    page.click("#save")
                    page.wait_for_function("document.querySelector('#exportStatus').textContent.includes('需要重新导出')")
                    page.click("#export")
                    page.wait_for_function("document.querySelector('#exportStatus').textContent.includes('导出失败')", timeout=10000)
                    self.assertEqual((root / "dataset").resolve(), previous_target)
                    self.assertFalse(errors, errors)
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(10)
                worker = app.state.service.worker
                if worker is not None:
                    worker.join(10)
                sock.close()


if __name__ == "__main__":
    unittest.main()
