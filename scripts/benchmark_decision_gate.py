"""Replay a tracker JSONL trace through the legacy and duplicate-gated policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.pipeline.core import new_decisions
from edge.pipeline.decision_gate import DuplicateDecisionGate, GateConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latency_summary(values_ns: list[int]) -> dict:
    ordered = sorted(values_ns)
    return {"mean_us": round(statistics.mean(ordered) / 1000, 3),
            "p50_us": round(ordered[len(ordered) // 2] / 1000, 3),
            "p95_us": round(ordered[min(len(ordered) - 1,
                                        math.ceil(len(ordered) * .95) - 1)] / 1000, 3)}


def replay(rows: list[dict], *, gated: bool, config: GateConfig,
           iterations: int, warmup: int) -> tuple[dict, list[int]]:
    timings: list[int] = []
    first_result = {}
    for run in range(warmup + iterations):
        gate = DuplicateDecisionGate(config) if gated else None
        seen: set[int] = set()
        decisions: list[int] = []
        suppressed: list[dict] = []
        for row in rows:
            events = [{"track_id": item["track_id"], "class": item["class_name"],
                       "confidence": item["confidence"], "bbox_xyxy": item["bbox_xyxy"],
                       "frame_id": row["frame_id"]} for item in row["objects"]]
            start = time.perf_counter_ns()
            if gate is None:
                emitted = new_decisions(events, seen)
            else:
                batch = gate.process(events, row["frame_id"])
                emitted = batch.decisions
                suppressed.extend(batch.suppressed)
            elapsed = time.perf_counter_ns() - start
            if run >= warmup:
                timings.append(elapsed)
            decisions.extend(item["track_id"] for item in emitted)
        if run == warmup:
            first_result = {"decision_count": len(decisions), "decision_track_ids": decisions,
                            "suppressed_count": len(suppressed), "suppressed": suppressed}
    return first_result, timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--min-iou", type=float, default=.60)
    parser.add_argument("--max-track-gap-frames", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations >= 1 and warmup >= 0 required")
    config = GateConfig(args.min_iou, args.max_track_gap_frames)
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    if not rows:
        parser.error("Trace is empty")
    summary = json.loads(args.trace.with_suffix(".summary.json").read_text())
    if len(rows) != summary["frames"]:
        parser.error("Trace and summary frame counts differ")

    legacy, legacy_times = replay(rows, gated=False, config=config,
                                  iterations=args.iterations, warmup=args.warmup)
    gated, gated_times = replay(rows, gated=True, config=config,
                                iterations=args.iterations, warmup=args.warmup)
    result = {"stage": "decision_gate_replay_benchmark",
              "scope": "Pure Python decision processing of saved tracker events; excludes video decoding, TensorRT, drawing, JPEG, and network streaming",
              "trace": str(args.trace), "trace_sha256": sha256(args.trace),
              "model": summary["model"], "source": summary["source"],
              "tracker": summary["tracker"], "ultralytics": summary["ultralytics"],
              "source_sha256": summary["source_sha256"],
              "model_sha256": summary["model_sha256"],
              "tracker_config_sha256": summary["tracker_config_sha256"],
              "frames": len(rows), "iterations": args.iterations,
              "warmup_iterations": args.warmup,
              "gate_config": {"min_iou": config.min_iou,
                              "max_track_gap_frames": config.max_track_gap_frames},
              "legacy": {**legacy, "policy_latency": latency_summary(legacy_times)},
              "gated": {**gated, "policy_latency": latency_summary(gated_times)},
              "platform": platform.platform(), "python": platform.python_version()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
