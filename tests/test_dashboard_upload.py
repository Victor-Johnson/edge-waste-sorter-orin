"""Upload control checks without loading a detector or needing Jetson hardware."""

import asyncio
import threading
import time

import httpx
import cv2
import numpy as np

from edge import dashboard
from edge.pipeline.core import Settings


class ReadableVideo:
    def __init__(self, path):
        self.path = path

    def isOpened(self):
        return True

    def read(self):
        return True, object()

    def release(self):
        pass


class UnreadableVideo(ReadableVideo):
    def isOpened(self):
        return False


def test_upload_starts_new_run_and_replaces_previous_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard.cv2, "VideoCapture", ReadableVideo)

    def finish_immediately(settings, state):
        with state.lock:
            state.finished = True
            state.ended = time.monotonic()

    monkeypatch.setattr(dashboard, "run_pipeline", finish_immediately)
    model = tmp_path / "model.engine"
    model.write_bytes(b"test model")
    settings = Settings(model=str(model), source="0", lanes={}, policy_enabled=True)
    app = dashboard.create_app(settings, wait_for_upload=True,
                               upload_dir=tmp_path / "uploads")

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://test") as client:
            initial = (await client.get("/api/status")).json()
            assert initial["idle"] is True
            first = await client.post("/api/upload-video", content=b"first",
                                      headers={"content-type": "video/mp4"})
            assert first.status_code == 202
            assert first.json()["run_id"] != initial["run_id"]
            for _ in range(100):
                status = (await client.get("/api/status")).json()
                if status["finished"]:
                    break
                await asyncio.sleep(.01)
            assert status["material_policy_enabled"] is True
            assert len(list((tmp_path / "uploads").glob("*.mp4"))) == 1

            second = await client.post("/api/upload-video", content=b"second",
                                       headers={"content-type": "video/mp4"})
            assert second.status_code == 202
            assert second.json()["run_id"] != first.json()["run_id"]
            files = list((tmp_path / "uploads").glob("*.mp4"))
            assert len(files) == 1
            assert files[0].read_bytes() == b"second"

    asyncio.run(exercise())


def test_upload_rejects_invalid_input_and_busy_run(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard.cv2, "VideoCapture", ReadableVideo)
    monkeypatch.setattr(dashboard, "MAX_UPLOAD_BYTES", 4)
    started = threading.Event()
    release = threading.Event()

    def wait_for_release(settings, state):
        started.set()
        release.wait(timeout=2)
        with state.lock:
            state.finished = True
            state.ended = time.monotonic()

    monkeypatch.setattr(dashboard, "run_pipeline", wait_for_release)
    settings = Settings(model=str(tmp_path / "model.engine"), source="0", lanes={})
    app = dashboard.create_app(settings, wait_for_upload=True,
                               upload_dir=tmp_path / "uploads")

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://test") as client:
            assert (await client.post("/api/upload-video", content=b"x",
                                      headers={"content-type": "text/plain"})).status_code == 415
            assert (await client.post("/api/upload-video", content=b"",
                                      headers={"content-type": "video/mp4"})).status_code == 400
            assert (await client.post("/api/upload-video", content=b"12345",
                                      headers={"content-type": "video/mp4"})).status_code == 413
            monkeypatch.setattr(dashboard.cv2, "VideoCapture", UnreadableVideo)
            assert (await client.post("/api/upload-video", content=b"bad",
                                      headers={"content-type": "video/mp4"})).status_code == 415
            monkeypatch.setattr(dashboard.cv2, "VideoCapture", ReadableVideo)
            assert not list((tmp_path / "uploads").glob("*.mp4"))

            accepted = await client.post("/api/upload-video", content=b"good",
                                         headers={"content-type": "video/mp4"})
            assert accepted.status_code == 202
            assert started.wait(timeout=1)
            assert (await client.post("/api/upload-video", content=b"more",
                                      headers={"content-type": "video/mp4"})).status_code == 409

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_image_upload_runs_detector_once_without_creating_video_run(tmp_path, monkeypatch):
    calls = []

    def fake_predict(settings, image):
        calls.append(image.shape)
        return {"detections": [{"class": "cans", "lane": "metal",
                                "confidence": .82, "track_id": None}],
                "inference_ms": 12.3, "annotated_jpeg_base64": "aGVsbG8="}

    monkeypatch.setattr(dashboard, "predict_still", fake_predict)
    settings = Settings(model=str(tmp_path / "model.engine"), source="0", lanes={})
    app = dashboard.create_app(settings, wait_for_upload=True,
                               upload_dir=tmp_path / "uploads")
    encoded, image = cv2.imencode(".png", np.zeros((12, 16, 3), dtype=np.uint8))
    assert encoded

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://test") as client:
            before = (await client.get("/api/status")).json()
            response = await client.post("/api/upload-image", content=image.tobytes(),
                                         headers={"content-type": "image/png"})
            assert response.status_code == 200
            assert response.json()["detections"][0]["lane"] == "metal"
            after = (await client.get("/api/status")).json()
            assert after["run_id"] == before["run_id"]
            assert after["decisions_total"] == 0
            assert not (tmp_path / "uploads").exists()

    asyncio.run(exercise())
    assert calls == [(12, 16, 3)]


def test_image_upload_rejects_bad_data_and_busy_video(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "MAX_IMAGE_BYTES", 8)
    settings = Settings(model=str(tmp_path / "model.engine"), source="0", lanes={})
    idle = dashboard.create_app(settings, wait_for_upload=True)
    busy = dashboard.create_app(settings, wait_for_upload=False)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=idle),
                                    base_url="http://test") as client:
            assert (await client.post("/api/upload-image", content=b"bad",
                                      headers={"content-type": "text/plain"})).status_code == 415
            assert (await client.post("/api/upload-image", content=b"",
                                      headers={"content-type": "image/png"})).status_code == 400
            assert (await client.post("/api/upload-image", content=b"123456789",
                                      headers={"content-type": "image/png"})).status_code == 413
            assert (await client.post("/api/upload-image", content=b"bad",
                                      headers={"content-type": "image/png"})).status_code == 415
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=busy),
                                    base_url="http://test") as client:
            assert (await client.post("/api/upload-image", content=b"x",
                                      headers={"content-type": "image/png"})).status_code == 409

    asyncio.run(exercise())
