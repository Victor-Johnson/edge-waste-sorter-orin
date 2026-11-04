"""FastAPI dashboard for the edge pipeline."""

from __future__ import annotations

import json
import hashlib
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from edge.pipeline.core import Settings, frames, infer_frame, new_decisions
from edge.pipeline.decision_gate import DuplicateDecisionGate, GateConfig
from edge.pipeline.policy import MaterialPolicy

WEB = Path(__file__).resolve().parent / "web"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PipelineState:
    def __init__(self, settings: Settings | None = None):
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.events = deque(maxlen=100)
        self.seen_track_ids: set[int] = set()
        self.duplicate_gate = (DuplicateDecisionGate(GateConfig(
            settings.duplicate_gate_min_iou, settings.duplicate_gate_max_track_gap_frames))
            if settings and settings.duplicate_gate_enabled else None)
        self.material_policy = (MaterialPolicy(
            settings.policy_config,
            model_sha256=_sha256(settings.model),
            source_sha256=_sha256(settings.source) if Path(settings.source).is_file() else None)
            if settings and settings.policy_enabled else None)
        self.suppressed_duplicate_candidates = 0
        self.recent_suppressions = deque(maxlen=20)
        self.decisions_total = 0
        self.latest: list[dict] = []
        self.frame_id = 0
        self.latencies = deque(maxlen=300)
        self.started = time.monotonic()
        self.ended = None
        self.error: str | None = None
        self.finished = False

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = max((self.ended or time.monotonic()) - self.started, 0.001)
            sorted_latency = sorted(self.latencies)
            return {
                "frame_id": self.frame_id,
                "fps": round(self.frame_id / elapsed, 2),
                "pipeline_p50_ms": round(sorted_latency[len(sorted_latency) // 2], 1) if sorted_latency else None,
                "pipeline_p95_ms": round(sorted_latency[min(len(sorted_latency) - 1, math.ceil(len(sorted_latency) * .95) - 1)], 1) if sorted_latency else None,
                "detections": self.latest,
                "recent_events": list(self.events),
                "decisions_total": self.decisions_total,
                "duplicate_gate_enabled": self.duplicate_gate is not None,
                "material_policy_enabled": self.material_policy is not None,
                "policy_version": (self.material_policy.config.version
                                   if self.material_policy else None),
                "pending_tracks": (len(self.material_policy.active)
                                   if self.material_policy else 0),
                "suppressed_duplicate_candidates": self.suppressed_duplicate_candidates,
                "recent_suppressions": list(self.recent_suppressions),
                "error": self.error,
                "finished": self.finished,
            }


def run_pipeline(settings: Settings, state: PipelineState) -> None:
    try:
        from ultralytics import YOLO

        model = YOLO(settings.model, task="detect")
        for frame_id, frame in enumerate(frames(settings.source), 1):
            if frame_id == 1:
                with state.lock:
                    state.started = time.monotonic()
            annotated, events, latency = infer_frame(model, frame, settings, frame_id)
            encoded, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with state.lock:
                eligible = events
                if state.duplicate_gate is not None:
                    batch = state.duplicate_gate.process(events, frame_id)
                    state.suppressed_duplicate_candidates += len(batch.suppressed)
                    state.recent_suppressions.extend(batch.suppressed)
                    eligible = [event for event in events if event["track_id"] is None
                                or event["track_id"] in state.duplicate_gate.approved_ids]
                if state.material_policy is not None:
                    decisions = state.material_policy.process(eligible, frame_id)
                elif state.duplicate_gate is not None:
                    decisions = batch.decisions
                else:
                    decisions = new_decisions(events, state.seen_track_ids)
                state.frame_id = frame_id
                state.latest = events
                state.events.extend(decisions)
                state.decisions_total += len(decisions)
                state.latencies.append(latency)
                if encoded:
                    state.jpeg = buffer.tobytes()
        if state.material_policy is not None:
            with state.lock:
                final_decisions = state.material_policy.finish()
                state.events.extend(final_decisions)
                state.decisions_total += len(final_decisions)
    except Exception as exc:
        with state.lock:
            state.error = str(exc)
    finally:
        with state.lock:
            state.finished = True
            state.ended = time.monotonic()


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="Waste sorter edge dashboard")
    state = PipelineState(settings)

    @app.on_event("startup")
    def start():
        threading.Thread(target=run_pipeline, args=(settings, state), daemon=True).start()

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/app.js")
    def script():
        return FileResponse(WEB / "app.js", media_type="text/javascript")

    @app.get("/style.css")
    def stylesheet():
        return FileResponse(WEB / "style.css", media_type="text/css")

    @app.get("/api/status")
    def status():
        return state.snapshot()

    @app.get("/api/events")
    def events():
        def stream():
            while True:
                yield f"data: {json.dumps(state.snapshot())}\n\n"
                time.sleep(0.2)
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/stream.mjpg")
    def video():
        def stream():
            while True:
                with state.lock:
                    jpeg = state.jpeg
                    finished = state.finished
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                if finished:
                    break
                time.sleep(0.05)
        return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app
