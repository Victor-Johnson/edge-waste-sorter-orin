"""Shared camera/video pipeline and sorting event conversion."""

from __future__ import annotations

import time
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from edge.pipeline.policy import PolicyConfig


@dataclass(frozen=True)
class Settings:
    model: str
    source: str
    lanes: dict[str, str]
    confidence: float = 0.35
    image_size: int = 640
    classes_file: str | None = None
    tracker: str = "botsort.yaml"
    duplicate_gate_enabled: bool = False
    duplicate_gate_min_iou: float = 0.60
    duplicate_gate_max_track_gap_frames: int = 5
    policy_enabled: bool = False
    policy_config: PolicyConfig = PolicyConfig()
    deployment_manifest: str | None = None


def load_settings(path: str | Path, *, model_override: str | None = None,
                  source_override: str | None = None,
                  tracker_override: str | None = None,
                  duplicate_gate_override: bool | None = None,
                  policy_override: bool | None = None,
                  manifest_override: str | None = None) -> Settings:
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
        from edge.pipeline.class_contract import validate_class_order

        validate_class_order(classes_path, config_path.parent / "class_order.yaml")
        classes_file = str(classes_path)
    tracker = str(tracker_override or raw.get("tracker", "botsort.yaml"))
    tracker_path = Path(tracker)
    if tracker_path.is_absolute() or tracker_path.parent != Path("."):
        if not tracker_path.is_absolute():
            tracker_path = (config_path.parent.parent / tracker_path).resolve()
        if not tracker_path.is_file():
            raise FileNotFoundError(f"Tracker config missing: {tracker_path}")
        tracker = str(tracker_path)
    from edge.pipeline.decision_gate import GateConfig

    gate_config = GateConfig(float(raw.get("duplicate_gate_min_iou", .60)),
                             int(raw.get("duplicate_gate_max_track_gap_frames", 5)))
    enabled = (bool(raw.get("duplicate_gate_enabled", False))
               if duplicate_gate_override is None else duplicate_gate_override)
    policy_enabled = (bool(raw.get("policy_enabled", False))
                      if policy_override is None else policy_override)
    policy_config = PolicyConfig()
    policy_path = None
    if policy_enabled:
        policy_path = Path(raw.get("policy_config", "configs/policy.yaml"))
        if not policy_path.is_absolute():
            policy_path = (config_path.parent.parent / policy_path).resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(f"Policy config missing: {policy_path}")
        policy_config = PolicyConfig(**(yaml.safe_load(policy_path.read_text()) or {}))
    manifest_path = manifest_override or raw.get("deployment_manifest")
    if manifest_path:
        from edge.pipeline.manifest import verify_deployment

        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = (config_path.parent.parent / manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text())
        verify_deployment(manifest, model=str(model), classes_file=classes_file,
                          lanes=dict(raw.get("lanes") or {}), tracker=tracker,
                          policy_enabled=policy_enabled,
                          policy_config_path=str(policy_path) if policy_path else None,
                          duplicate_gate_enabled=enabled)
    return Settings(model=str(model), source=source, lanes=dict(raw.get("lanes") or {}),
                    confidence=confidence, image_size=int(raw.get("image_size", 640)),
                    classes_file=classes_file, tracker=tracker,
                    duplicate_gate_enabled=enabled,
                    duplicate_gate_min_iou=gate_config.min_iou,
                    duplicate_gate_max_track_gap_frames=gate_config.max_track_gap_frames,
                    policy_enabled=policy_enabled, policy_config=policy_config,
                    deployment_manifest=str(manifest_path) if manifest_path else None)


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
                         imgsz=settings.image_size, tracker=settings.tracker)[0]
    if settings.classes_file:
        result.names = class_names(settings.classes_file)
    events = detection_events(result, settings.lanes, frame_id, time.time())
    annotated = result.plot()
    latency_ms = (time.perf_counter() - started) * 1000
    return annotated, events, latency_ms
