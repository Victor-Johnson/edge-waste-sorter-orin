"""End-to-end video benchmark with reproducible JSON output."""

import argparse
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.pipeline.core import frames, infer_frame, load_settings
from edge.tracking import log_record


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, help="Use the same video for comparable runs")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mlflow", action="store_true", help="Log this run to the waste-sorter-edge MLflow experiment")
    args = parser.parse_args()
    from ultralytics import YOLO
    settings = load_settings(args.config, model_override=args.model,
                             source_override=args.source)
    model = YOLO(settings.model, task="detect")
    latencies = []
    counts = 0
    start = None
    for index, frame in enumerate(frames(settings.source)):
        if index >= args.warmup + args.frames:
            break
        if index == args.warmup:
            start = time.perf_counter()
        annotated, events, latency = infer_frame(model, frame, settings, index)
        cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if index >= args.warmup:
            latencies.append(latency)
            counts += len(events)
    if not latencies:
        raise RuntimeError("Video ended before any measured frames")
    elapsed = time.perf_counter() - start
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "model": settings.model,
              "source": settings.source, "warmup_frames": args.warmup, "measured_frames": len(latencies),
              "image_size": settings.image_size, "confidence": settings.confidence,
              "end_to_end_fps": round(len(latencies) / elapsed, 2),
              "pipeline_ms": {"mean": round(statistics.mean(latencies), 2),
                               "p50": round(percentile(latencies, .5), 2),
                               "p95": round(percentile(latencies, .95), 2)},
              "detections": counts, "platform": platform.platform()}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if args.mlflow:
        print("MLflow run:", log_record(record, output, "benchmark"))


if __name__ == "__main__":
    main()
