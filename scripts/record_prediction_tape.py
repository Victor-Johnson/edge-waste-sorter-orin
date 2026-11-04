"""Flatten an existing Jetson track-review trace into a reusable observation tape.

The trace was already captured by scripts/jetson_track_review.py; conversion
does not run TensorRT again or store video frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.pipeline.class_contract import validate_class_order


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL tape")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    names = validate_class_order(args.classes, args.config.parent / "class_order.yaml")
    lanes = (yaml.safe_load(args.config.read_text()) or {}).get("lanes") or {}
    if len(names) != 28 or len(set(names)) != 28:
        raise ValueError("Expected 28 unique class names in index order")
    if set(lanes) != set(names):
        raise ValueError(f"Lane map mismatch: missing={sorted(set(names)-set(lanes))}, "
                         f"extra={sorted(set(lanes)-set(names))}")
    if summary["classes_sha256"] != sha256(args.classes):
        raise ValueError("Class-list hash differs from tracker trace")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    observation_count = 0
    with args.trace.open() as source, args.out.open("w") as target:
        for line in source:
            frame = json.loads(line)
            frame_id = frame["frame_id"]
            if frame_id != frame_count + 1:
                raise ValueError(f"Frame sequence gap at {frame_id}")
            frame_count += 1
            for item in frame["objects"]:
                class_id = int(item["class_id"])
                if not 0 <= class_id < len(names) or item["class_name"] != names[class_id]:
                    raise ValueError(f"Class-order mismatch at frame {frame_id}, index {class_id}")
                row = {"frame_id": frame_id, "track_id": item["track_id"],
                       "class_id": class_id, "class_name": names[class_id],
                       "confidence": item["confidence"],
                       "bbox_xyxy": item["bbox_xyxy"], "lane": lanes[names[class_id]]}
                target.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                observation_count += 1
    if frame_count != summary["frames"] or observation_count != summary["frame_level_detections"]:
        raise ValueError("Trace counts differ from tracker summary")
    manifest = {
        "format": "waste-sorter-observation-tape-v1",
        "tape": args.out.name,
        "tape_sha256": sha256(args.out),
        "trace_sha256": sha256(args.trace),
        "model_sha256": summary["model_sha256"],
        "source_sha256": summary["source_sha256"],
        "classes_sha256": sha256(args.classes),
        "lane_config_sha256": sha256(args.config),
        "tracker_config_sha256": summary["tracker_config_sha256"],
        "tracker": summary["tracker"],
        "confidence": summary["confidence"],
        "image_size": summary["image_size"],
        "ultralytics": summary["ultralytics"],
        "frames": frame_count,
        "observations": observation_count,
        "scope": "Saved tracker predictions, no ground-truth material labels",
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"tape": str(args.out), "manifest": str(manifest_path),
                      "frames": frame_count, "observations": observation_count}, indent=2))


if __name__ == "__main__":
    main()
