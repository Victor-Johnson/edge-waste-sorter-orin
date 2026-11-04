"""Measure a video-to-JPEG inference path without dashboard or tracker dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import ultralytics
from ultralytics import YOLO


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": round(statistics.mean(values), 2),
            "p50": round(percentile(values, .5), 2),
            "p95": round(percentile(values, .95), 2)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=.35)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("--frames must be positive and --warmup nonnegative")
    for path in (args.model, args.video, args.classes):
        if not path.is_file():
            parser.error(f"Missing file: {path}")

    names = [line.strip() for line in args.classes.read_text().splitlines() if line.strip()]
    model = YOLO(str(args.model), task="detect")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    times = {key: [] for key in ("decode", "predict", "draw", "jpeg", "frame")}
    classes = Counter()
    measured_start = None
    try:
        for index in range(args.warmup + args.frames):
            if index == args.warmup:
                measured_start = time.perf_counter()
            start = time.perf_counter()
            ok, frame = capture.read()
            decoded = time.perf_counter()
            if not ok:
                break
            result = model.predict(frame, imgsz=640, conf=args.confidence,
                                   device=args.device, verbose=False)[0]
            predicted = time.perf_counter()
            result.names = dict(enumerate(names))  # trtexec engines have no Ultralytics class metadata
            annotated = result.plot()
            drawn = time.perf_counter()
            encoded, _ = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not encoded:
                raise RuntimeError(f"JPEG encoding failed at frame {index}")
            finished = time.perf_counter()
            if index >= args.warmup:
                for key, value in (("decode", decoded - start), ("predict", predicted - decoded),
                                   ("draw", drawn - predicted), ("jpeg", finished - drawn),
                                   ("frame", finished - start)):
                    times[key].append(value * 1000)
                classes.update(names[int(class_id)] for class_id in result.boxes.cls.tolist())
    finally:
        capture.release()
    if not times["frame"]:
        raise RuntimeError("Video ended before any measured frames")
    elapsed = time.perf_counter() - measured_start
    record = {
        "stage": "benchmark", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_scope": "video decode + prediction + annotation + JPEG; excludes tracking and network streaming",
        "model": str(args.model.resolve()), "sha256": sha256(args.model),
        "source": str(args.video.resolve()), "source_sha256": sha256(args.video),
        "image_size": 640, "confidence": args.confidence,
        "warmup_frames": args.warmup, "measured_frames": len(times["frame"]),
        "end_to_end_fps": round(len(times["frame"]) / elapsed, 2),
        "pipeline_ms": summary(times["frame"]),
        "decode_ms": summary(times["decode"]),
        "predict_ms": summary(times["predict"]),
        "draw_ms": summary(times["draw"]),
        "jpeg_ms": summary(times["jpeg"]),
        "detections": sum(classes.values()), "per_class_detections": dict(classes),
        "track_enabled": False, "platform": platform.platform(),
        "environment": {"hardware": platform.machine(), "ultralytics": ultralytics.__version__,
                        "opencv": cv2.__version__, "device": args.device},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
