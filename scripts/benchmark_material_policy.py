"""Measure material policy processing from a saved observation tape on this device."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.pipeline.policy import MaterialPolicy, PolicyConfig
from scripts.evaluate_policy import read_tape, sha256


def summary(values: list[int]) -> dict:
    ordered = sorted(values)
    return {"mean_us": round(statistics.mean(ordered) / 1000, 3),
            "p50_us": round(ordered[len(ordered) // 2] / 1000, 3),
            "p95_us": round(ordered[(len(ordered) * 95 + 99) // 100 - 1] / 1000, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/policy.yaml"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    manifest = json.loads(args.manifest.read_text())
    frames = read_tape(args.tape, manifest)
    config = PolicyConfig(**(yaml.safe_load(args.config.read_text()) or {}))
    timings = []
    first_counts = None
    for run in range(args.warmup + args.iterations):
        policy = MaterialPolicy(config)
        counts = {"route": 0, "review": 0}
        for frame_id, events in enumerate(frames, 1):
            start = time.perf_counter_ns()
            decisions = policy.process(events, frame_id)
            elapsed = time.perf_counter_ns() - start
            if run >= args.warmup:
                timings.append(elapsed)
            for decision in decisions:
                counts[decision["status"]] += 1
        for decision in policy.finish():
            counts[decision["status"]] += 1
        if run == args.warmup:
            first_counts = counts
    result = {"stage": "material_policy_replay_benchmark",
              "scope": "Pure Python policy processing from saved tracker observations; excludes TensorRT, video, tracking, drawing, JPEG, network, and final EOF flush",
              "tape_sha256": manifest["tape_sha256"],
              "manifest_sha256": sha256(args.manifest),
              "model_sha256": manifest["model_sha256"],
              "source_sha256": manifest["source_sha256"],
              "tracker_config_sha256": manifest["tracker_config_sha256"],
              "policy_config_sha256": sha256(args.config),
              "config": asdict(config), "frames": len(frames),
              "iterations": args.iterations, "warmup_iterations": args.warmup,
              "outcomes": first_counts, "policy_latency": summary(timings),
              "platform": platform.platform(), "python": platform.python_version()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
