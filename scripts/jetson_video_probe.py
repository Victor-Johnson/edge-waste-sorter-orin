"""Measure video-to-JPEG prediction, with optional tracking and tracker timings."""

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
    parser.add_argument("--track", action="store_true",
                        help="Use persistent BoT-SORT tracking, matching the dashboard")
    parser.add_argument("--tracker", default="botsort.yaml",
                        help="Ultralytics tracker config path; used only with --track")
    parser.add_argument("--profile-tracker", action="store_true",
                        help="Time BoT-SORT update and global motion compensation separately")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("--frames must be positive and --warmup nonnegative")
    if args.profile_tracker and not args.track:
        parser.error("--profile-tracker requires --track")
    for path in (args.model, args.video, args.classes):
        if not path.is_file():
            parser.error(f"Missing file: {path}")

    names = [line.strip() for line in args.classes.read_text().splitlines() if line.strip()]
    model = YOLO(str(args.model), task="detect")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    times = {key: [] for key in ("decode", "predict", "draw", "jpeg", "frame")}
    tracker_timing = {"update_ms": 0.0, "gmc_ms": 0.0, "update_calls": 0, "gmc_calls": 0}
    original_update = original_gmc = None
    if args.profile_tracker:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.utils.gmc import GMC

        original_update = BYTETracker.update
        original_gmc = GMC.apply

        def timed_update(self, *a, **kw):
            started = time.perf_counter()
            try:
                return original_update(self, *a, **kw)
            finally:
                tracker_timing["update_ms"] += (time.perf_counter() - started) * 1000
                tracker_timing["update_calls"] += 1

        def timed_gmc(self, *a, **kw):
            started = time.perf_counter()
            try:
                return original_gmc(self, *a, **kw)
            finally:
                tracker_timing["gmc_ms"] += (time.perf_counter() - started) * 1000
                tracker_timing["gmc_calls"] += 1

        BYTETracker.update = timed_update
        GMC.apply = timed_gmc
        times.update({key: [] for key in ("tracker_update", "gmc",
                                             "tracker_update_excluding_gmc",
                                             "track_call_outside_update",
                                             "tracker_update_calls", "gmc_calls")})
    classes = Counter()
    observed_track_ids = set()
    untracked_detections = 0
    measured_start = None
    measured_start_utc = None
    try:
        for index in range(args.warmup + args.frames):
            if index == args.warmup:
                measured_start = time.perf_counter()
                measured_start_utc = datetime.now(timezone.utc)
            start = time.perf_counter()
            ok, frame = capture.read()
            decoded = time.perf_counter()
            if not ok:
                break
            if args.profile_tracker:
                tracker_timing.update(update_ms=0.0, gmc_ms=0.0, update_calls=0, gmc_calls=0)
            if args.track:
                result = model.track(frame, imgsz=640, conf=args.confidence,
                                     device=args.device, tracker=args.tracker,
                                     persist=True, verbose=False)[0]
            else:
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
                if args.profile_tracker:
                    update_ms = tracker_timing["update_ms"]
                    gmc_ms = tracker_timing["gmc_ms"]
                    times["tracker_update"].append(update_ms)
                    times["gmc"].append(gmc_ms)
                    times["tracker_update_excluding_gmc"].append(max(0, update_ms - gmc_ms))
                    times["track_call_outside_update"].append(max(0, (predicted - decoded) * 1000 - update_ms))
                    times["tracker_update_calls"].append(tracker_timing["update_calls"])
                    times["gmc_calls"].append(tracker_timing["gmc_calls"])
                classes.update(names[int(class_id)] for class_id in result.boxes.cls.tolist())
                if args.track:
                    ids = result.boxes.id
                    if ids is None:
                        untracked_detections += len(result.boxes)
                    else:
                        observed_track_ids.update(int(track_id) for track_id in ids.tolist())
    finally:
        if args.profile_tracker:
            BYTETracker.update = original_update
            GMC.apply = original_gmc
        capture.release()
    if not times["frame"]:
        raise RuntimeError("Video ended before any measured frames")
    measured_end_utc = datetime.now(timezone.utc)
    elapsed = time.perf_counter() - measured_start
    record = {
        "stage": "benchmark", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_scope": ("video decode + prediction + BoT-SORT tracking + annotation + JPEG; "
                              "excludes network streaming" if args.track else
                              "video decode + prediction + annotation + JPEG; excludes tracking and network streaming"),
        "model": str(args.model.resolve()), "sha256": sha256(args.model),
        "source": str(args.video.resolve()), "source_sha256": sha256(args.video),
        "image_size": 640, "confidence": args.confidence,
        "measurement_start_utc": measured_start_utc.isoformat(),
        "measurement_end_utc": measured_end_utc.isoformat(),
        "warmup_frames": args.warmup, "measured_frames": len(times["frame"]),
        "end_to_end_fps": round(len(times["frame"]) / elapsed, 2),
        "pipeline_ms": summary(times["frame"]),
        "decode_ms": summary(times["decode"]),
        "predict_ms": summary(times["predict"]),
        "draw_ms": summary(times["draw"]),
        "jpeg_ms": summary(times["jpeg"]),
        "detections": sum(classes.values()), "per_class_detections": dict(classes),
        "track_enabled": args.track, "platform": platform.platform(),
        "environment": {"hardware": platform.machine(), "ultralytics": ultralytics.__version__,
                        "opencv": cv2.__version__, "device": args.device},
    }
    if args.track:
        record["tracker"] = args.tracker
        if Path(args.tracker).is_file():
            record["tracker_config_sha256"] = sha256(Path(args.tracker))
        record["unique_track_ids_observed"] = len(observed_track_ids)
        record["untracked_detections"] = untracked_detections
    if args.profile_tracker:
        record["profile_tracker"] = True
        record["tracker_update_ms"] = summary(times["tracker_update"])
        record["gmc_ms"] = summary(times["gmc"])
        record["tracker_update_excluding_gmc_ms"] = summary(times["tracker_update_excluding_gmc"])
        record["track_call_outside_update_ms"] = summary(times["track_call_outside_update"])
        record["tracker_update_calls"] = sum(times["tracker_update_calls"])
        record["gmc_calls"] = sum(times["gmc_calls"])
        record["profile_note"] = ("tracker_update_ms includes gmc_ms; times are measured inside the "
                                  "Python tracker call and may differ from an uninstrumented run")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
