"""Local review API. Edits affect JSON only until an explicit export request."""
from __future__ import annotations

import base64
import copy
import io
import json
import threading
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import h5py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .core import CAMERAS, Config, Recording, atomic_json, discover, fingerprint, load_plan, normalize_plan, plan_digest, scan
from .export import export_dataset


class PlanRequest(BaseModel):
    plan: dict
    base_revision: str


class ExportRequest(BaseModel):
    revision: str


class ReviewService:
    def __init__(self, config: Config, clips: Path):
        config.validate()
        self.config = config
        self.clips = clips
        self.lock = threading.RLock()
        self.plan = load_plan(clips)
        self.entries = scan(config, self.plan)
        self.paths = {p.name: p for p in discover(config)}
        unknown = set(self.plan["recordings"]) - self.paths.keys()
        if unknown:
            raise ValueError(f"Clip plan references missing/unselected sources: {sorted(unknown)}")
        self.job = {"phase": "idle"}
        self.worker = None

    @lru_cache(maxsize=2)
    def recording(self, name: str) -> Recording:
        if name not in self.paths:
            raise ValueError("Unknown recording")
        return Recording(self.paths[name], self.config)

    def get_recording(self, name: str) -> Recording:
        rec = self.recording(name)
        rec.check_source()
        return rec

    def revision(self):
        return plan_digest(self.plan)

    def export_status(self):
        with self.lock:
            job = copy.deepcopy(self.job)
            digest = plan_digest(self.plan, self.config)
        report_path = Path(self.config.output) / "export_report.json"
        current = None
        if report_path.is_file():
            try:
                current = json.loads(report_path.read_text())
            except (OSError, ValueError):
                pass
        stale = (current is None or current.get("plan_digest") != digest
                 or current.get("max_frames_per_source") is not None
                 or set(current.get("source_selection", [])) != set(self.paths))
        if current and not stale:
            for name, fp in current.get("source_fingerprints", {}).items():
                path = self.paths.get(name)
                if path is None or not path.is_file():
                    stale = True
                    break
                if fp != fingerprint(path):
                    stale = True
                    break
        return {"job": job, "stale": stale, "exists": current is not None,
                "output": self.config.output,
                "current": None if current is None else {k: current.get(k) for k in ("total_frames", "total_episodes", "created_at", "skipped")}}

    def start_export(self, revision: str):
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                raise HTTPException(409, "An export is already running")
            if revision != self.revision():
                raise HTTPException(409, "Saved plan changed; refresh before exporting")
            # Persist even the initial empty plan so the exact CLI input exists.
            atomic_json(self.clips, self.plan)
            snapshot = copy.deepcopy(self.plan)
            self.job = {"phase": "queued", "revision": revision}

            def progress(event):
                with self.lock:
                    self.job.update(event)

            def run():
                try:
                    report = export_dataset(self.config, snapshot, progress=progress)
                    progress({"phase": "complete", "skipped": report["skipped"]})
                except Exception as exc:
                    progress({"phase": "failed", "error": str(exc)})

            self.worker = threading.Thread(target=run, name="lerobot-export", daemon=False)
            self.worker.start()


def create_app(config: Config, clips: Path) -> FastAPI:
    service = ReviewService(config, clips)
    app = FastAPI(title="roboLab H5 review", docs_url=None, redoc_url=None)
    app.state.service = service
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    def get_rec(name):
        try:
            return service.get_recording(name)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/recordings")
    def recordings():
        with service.lock:
            entries = copy.deepcopy(service.entries)
            plan = copy.deepcopy(service.plan)
        for entry in entries:
            if entry["valid"]:
                # No H5 reads required to update the sidebar after a save.
                edit = plan["recordings"].get(entry["name"], {})
                entry["edited"] = bool(edit.get("drop") or edit.get("exclude"))
        return {"recordings": entries, "config": asdict(config), "clips_path": str(clips)}

    @app.get("/api/plan")
    def get_plan():
        with service.lock:
            return {"plan": copy.deepcopy(service.plan), "revision": service.revision()}

    @app.put("/api/plan")
    def save_plan(body: PlanRequest):
        try:
            plan = normalize_plan(body.plan)
            for name, edit in plan["recordings"].items():
                if name not in service.paths:
                    raise ValueError(f"Unselected source: {name}")
                # Fingerprints guard against replaying decisions on replaced H5.
                rec = get_rec(name)
                if "fingerprint" in edit and edit["fingerprint"] != rec.fingerprint:
                    raise ValueError(f"Stale source fingerprint: {name}")
                edit["fingerprint"] = rec.fingerprint
            with service.lock:
                if body.base_revision != service.revision():
                    raise HTTPException(409, "Another browser saved changes; reload the saved plan")
                atomic_json(clips, plan)
                service.plan = plan
                return {"plan": copy.deepcopy(plan), "revision": service.revision()}
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/recordings/{name}/timeline")
    def timeline(name: str):
        rec = get_rec(name)
        try:
            with service.lock:
                summary = rec.summary(service.plan)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {**summary, "seconds": rec.seconds.tolist(), "action": rec.aligned["action"].tolist()}

    @app.get("/api/recordings/{name}/frame")
    def frame(name: str, index: int = 0):
        rec = get_rec(name)
        if not 0 <= index < len(rec.grid):
            raise HTTPException(400, "Frame index is out of range")
        images = {}
        with h5py.File(rec.path, "r") as f:
            for camera in CAMERAS:
                rgb = rec.read_rgb(f, camera, index)
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=82)
                images[camera] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        return {"name": name, "index": index, "seconds": float(rec.seconds[index]),
                "action": rec.aligned["action"][index].tolist(), "images": images,
                "age_ms": {key: float(value[index]) for key, value in rec.lag.items()}}

    @app.get("/api/export")
    def status():
        return service.export_status()

    @app.post("/api/export", status_code=202)
    def start_export(body: ExportRequest):
        service.start_export(body.revision)
        return service.export_status()

    return app
