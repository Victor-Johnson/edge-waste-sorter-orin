"""FastAPI dashboard for the edge pipeline."""

from __future__ import annotations

import asyncio
import base64
import json
import hashlib
import math
import threading
import time
from dataclasses import replace
from collections import deque
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from edge.pipeline.core import (Settings, class_names, detection_events, frames,
                                infer_frame, new_decisions)
from edge.pipeline.decision_gate import DuplicateDecisionGate, GateConfig
from edge.pipeline.policy import MaterialPolicy

WEB = Path(__file__).resolve().parent / "web"
UPLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PipelineState:
    def __init__(self, settings: Settings | None = None, *, idle: bool = False):
        self.lock = threading.Lock()
        self.run_id = uuid4().hex
        self.idle = idle
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
        self.ended = self.started if idle else None
        self.error: str | None = None
        self.finished = idle

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = max((self.ended or time.monotonic()) - self.started, 0.001)
            sorted_latency = sorted(self.latencies)
            return {
                "run_id": self.run_id,
                "idle": self.idle,
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


def predict_still(settings: Settings, image: np.ndarray) -> dict:
    """Run one detector prediction; a still has no track or multi-frame decision."""
    from ultralytics import YOLO

    model = YOLO(settings.model, task="detect")
    started = time.perf_counter()
    result = model.predict(image, verbose=False, conf=settings.confidence,
                           imgsz=settings.image_size)[0]
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    if settings.classes_file:
        result.names = class_names(settings.classes_file)
    detections = detection_events(result, settings.lanes, 1, time.time())
    encoded, buffer = cv2.imencode(".jpg", result.plot(),
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not encoded:
        raise RuntimeError("Could not encode the annotated image")
    return {
        "detections": detections,
        "inference_ms": latency_ms,
        "annotated_jpeg_base64": base64.b64encode(buffer).decode("ascii"),
    }


def create_app(settings: Settings, *, wait_for_upload: bool = False,
               upload_dir: Path = UPLOAD_DIR) -> FastAPI:
    app = FastAPI(title="Waste sorter edge dashboard")
    current = {"state": PipelineState(settings, idle=wait_for_upload),
               "thread": None, "upload_path": None, "uploading": False,
               "image_processing": False}
    control_lock = threading.Lock()

    @app.on_event("startup")
    def start():
        if not wait_for_upload:
            worker = threading.Thread(target=run_pipeline,
                                      args=(settings, current["state"]), daemon=True)
            current["thread"] = worker
            worker.start()

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
    async def status():
        return current["state"].snapshot()

    @app.post("/api/upload-video", status_code=202)
    async def upload_video(request: Request):
        """Accept one MP4 at a time, then run the existing Jetson pipeline on it."""
        if request.headers.get("content-type", "").split(";")[0].strip() != "video/mp4":
            raise HTTPException(415, "Upload an MP4 video")
        size_header = request.headers.get("content-length")
        if size_header:
            try:
                if int(size_header) > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video exceeds the 200 MiB limit")
            except ValueError as exc:
                raise HTTPException(400, "Invalid content length") from exc
        with control_lock:
            if (current["uploading"] or current["image_processing"]
                    or not current["state"].finished):
                raise HTTPException(409, "Wait for the current upload or video to finish")
            current["uploading"] = True

        path = upload_dir / f"{uuid4().hex}.mp4"
        received = 0
        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Video exceeds the 200 MiB limit")
                    output.write(chunk)
            if received == 0:
                raise HTTPException(400, "Video is empty")
            capture = cv2.VideoCapture(str(path))
            try:
                opened = capture.isOpened()
                readable, _ = capture.read() if opened else (False, None)
            finally:
                capture.release()
            if not readable:
                raise HTTPException(415, "This MP4 could not be decoded")

            with control_lock:
                if not current["state"].finished:
                    raise HTTPException(409, "Another video started during upload")
                next_settings = replace(settings, source=str(path))
                next_state = PipelineState(next_settings)
                previous_upload = current["upload_path"]
                worker = threading.Thread(target=run_pipeline,
                                          args=(next_settings, next_state), daemon=True)
                current.update(state=next_state, thread=worker, upload_path=path)
                worker.start()
            if previous_upload is not None:
                try:
                    previous_upload.unlink(missing_ok=True)
                except OSError:
                    pass  # The new run is valid even if old upload cleanup fails.
            return {"run_id": next_state.run_id, "size_bytes": received}
        finally:
            try:
                if current["upload_path"] != path:
                    path.unlink(missing_ok=True)
            finally:
                with control_lock:
                    current["uploading"] = False

    @app.post("/api/upload-image")
    async def upload_image(request: Request):
        """Run a bounded JPEG/PNG through the detector without storing it."""
        media_type = request.headers.get("content-type", "").split(";")[0].strip()
        if media_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(415, "Upload a JPEG or PNG image")
        size_header = request.headers.get("content-length")
        if size_header:
            try:
                if int(size_header) > MAX_IMAGE_BYTES:
                    raise HTTPException(413, "Image exceeds the 10 MiB limit")
            except ValueError as exc:
                raise HTTPException(400, "Invalid content length") from exc
        with control_lock:
            if (current["uploading"] or current["image_processing"]
                    or not current["state"].finished):
                raise HTTPException(409, "Wait for the current video or image to finish")
            current["image_processing"] = True
        try:
            data = bytearray()
            async for chunk in request.stream():
                if len(data) + len(chunk) > MAX_IMAGE_BYTES:
                    raise HTTPException(413, "Image exceeds the 10 MiB limit")
                data.extend(chunk)
            if not data:
                raise HTTPException(400, "Image is empty")
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(415, "This image could not be decoded")
            height, width = image.shape[:2]
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image exceeds the 16 megapixel limit")
            return predict_still(settings, image)
        finally:
            with control_lock:
                current["image_processing"] = False

    @app.get("/api/events")
    async def events():
        async def stream():
            while True:
                yield f"data: {json.dumps(current['state'].snapshot())}\n\n"
                await asyncio.sleep(0.2)
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/stream.mjpg")
    async def video():
        async def stream():
            while True:
                state = current["state"]
                with state.lock:
                    jpeg = state.jpeg
                    finished = state.finished
                    idle = state.idle
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                if finished and not idle:
                    break
                await asyncio.sleep(0.05)
        return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app
