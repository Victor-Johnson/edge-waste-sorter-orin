"""Shared camera/video pipeline and sorting event conversion."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model: str
    source: str
    lanes: dict[str, str]
    confidence: float = 0.35
    image_size: int = 640
    classes_file: str | None = None


def load_settings(path: str | Path, *, model_override: str | None = None,
                  source_override: str | None = None) -> Settings:
    import yaml

    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    model = Path(model_override or raw["model"])
    if not model.is_absolute():
        model = (config_path.parent.parent / model).resolve()
    if not model.exists():
        raise FileNotFoundError(f"Model missing: {model}")
    source = str(source_override or raw.get("source", "0"))
    if source != "0" and not source.isdigit():
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = (config_path.parent.parent / source_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Video missing: {source_path}")
        source = str(source_path)
    confidence = float(raw.get("confidence", 0.35))
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    classes_file = None
    if raw.get("classes_file"):
        classes_path = Path(raw["classes_file"])
        if not classes_path.is_absolute():
            classes_path = (config_path.parent.parent / classes_path).resolve()
        if not classes_path.is_file():
            raise FileNotFoundError(f"Class list missing: {classes_path}")
        classes_file = str(classes_path)
    return Settings(str(model), source, dict(raw.get("lanes") or {}), confidence,
                    int(raw.get("image_size", 640)), classes_file)


def detection_events(result, lanes: dict[str, str], frame_id: int, timestamp: float) -> list[dict]:
    """Convert Ultralytics boxes into the public event shape."""
    events = []
    boxes = result.boxes
    if boxes is None:
        return events
    for index, box in enumerate(boxes):
        class_id = int(box.cls.item())
        name = str(result.names[class_id])
        xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        track_id = int(box.id.item()) if box.id is not None else None
        events.append({
            "timestamp": timestamp,
            "frame_id": frame_id,
            "track_id": track_id,
            "class": name,
            "confidence": round(float(box.conf.item()), 4),
            "bbox_xyxy": xyxy,
            "lane": lanes.get(name, "review"),
        })
    return events


def new_decisions(events: list[dict], seen_track_ids: set[int]) -> list[dict]:
    """Emit one sorting decision per confirmed track, including within a frame."""
    decisions = []
    for event in events:
        track_id = event["track_id"]
        if track_id is None or track_id in seen_track_ids:
            continue
        seen_track_ids.add(track_id)
        decisions.append(event)
    return decisions


@lru_cache(maxsize=4)
def class_names(path: str) -> dict[int, str]:
    names = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    return dict(enumerate(names))


def frames(source: str):
    import cv2

    camera = source.isdigit()
    capture = cv2.VideoCapture(int(source) if camera else source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    try:
        failures = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                if not camera:
                    break
                failures += 1
                if failures >= 5:
                    raise RuntimeError(f"Camera stopped delivering frames: {source}")
                time.sleep(.2)
                continue
            failures = 0
            yield frame
    finally:
        capture.release()


def infer_frame(model, frame, settings: Settings, frame_id: int):
    started = time.perf_counter()
    result = model.track(frame, persist=True, verbose=False, conf=settings.confidence,
                         imgsz=settings.image_size, tracker="botsort.yaml")[0]
    if settings.classes_file:
        result.names = class_names(settings.classes_file)
    events = detection_events(result, settings.lanes, frame_id, time.time())
    annotated = result.plot()
    latency_ms = (time.perf_counter() - started) * 1000
    return annotated, events, latency_ms
