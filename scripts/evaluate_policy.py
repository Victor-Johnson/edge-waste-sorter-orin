"""Replay a saved prediction tape at multiple advisory policy thresholds.

Without --labels, this reports decision volume only. A per-track labelled lane
file is required before coverage or cross-material error can be computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.pipeline.policy import MaterialPolicy, PolicyConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def read_tape(path: Path, manifest: dict) -> list[list[dict]]:
    if sha256(path) != manifest["tape_sha256"]:
        raise ValueError("Tape hash does not match manifest")
    frames: list[list[dict]] = [[] for _ in range(manifest["frames"])]
    count = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        frame_id = row["frame_id"]
        if not 1 <= frame_id <= len(frames):
            raise ValueError(f"Invalid frame ID {frame_id}")
        frames[frame_id - 1].append({"frame_id": frame_id,
                                      "track_id": row["track_id"],
                                      "class": row["class_name"],
                                      "confidence": row["confidence"],
                                      "bbox_xyxy": row["bbox_xyxy"],
                                      "lane": row["lane"]})
        count += 1
    if count != manifest["observations"]:
        raise ValueError("Observation count does not match manifest")
    return frames


def score(decisions: list[dict], labels: dict[int, str]) -> dict:
    by_track = {item["track_id"]: item for item in decisions if item["track_id"] is not None}
    missing = sorted(set(labels) - set(by_track))
    routed = [(true, by_track[track_id]["lane"]) for track_id, true in labels.items()
              if track_id in by_track and by_track[track_id]["status"] == "route"]
    wrong = [(true, proposed) for true, proposed in routed if true != proposed]
    return {"labelled_tracks": len(labels), "missing_decisions": missing,
            "automatic_coverage": round(len(routed) / len(labels), 4) if labels else None,
            "cross_material_error_per_route": round(len(wrong) / len(routed), 4) if routed else None,
            "wrong_routes": len(wrong),
            "material_confusion": dict(sorted(Counter(f"{true}->{proposed}" for true, proposed in wrong).items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/policy.yaml"))
    parser.add_argument("--classes", type=Path, help="Verify current class list against tape")
    parser.add_argument("--labels", type=Path, help="JSON mapping track ID to ground-truth material lane")
    parser.add_argument("--thresholds", default="0.40,0.55,0.70")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["format"] != "waste-sorter-observation-tape-v1":
        raise ValueError("Unsupported tape format")
    if args.classes and sha256(args.classes) != manifest["classes_sha256"]:
        raise ValueError("Class-list hash does not match manifest")
    frames = read_tape(args.tape, manifest)
    config = PolicyConfig(**(yaml.safe_load(args.config.read_text()) or {}))
    labels = None
    if args.labels:
        raw = json.loads(args.labels.read_text())
        labels = {int(track_id): lane for track_id, lane in raw.items()}
        if not all(lane in {"plastic", "metal", "carton", "glass"} for lane in labels.values()):
            raise ValueError("Labels must be material lanes: plastic, metal, carton, glass")
    reports = []
    for threshold in [float(value.strip()) for value in args.thresholds.split(",")]:
        policy = MaterialPolicy(replace(config, min_mean_confidence=threshold),
                                model_sha256=manifest["model_sha256"],
                                source_sha256=manifest["source_sha256"])
        decisions = []
        for frame_id, observations in enumerate(frames, 1):
            decisions.extend(policy.process(observations, frame_id))
        decisions.extend(policy.finish())
        counts = Counter(item["reason"] for item in decisions)
        report = {"threshold": threshold, "proposed_routes": sum(item["status"] == "route" for item in decisions),
                  "proposed_reviews": sum(item["status"] == "review" for item in decisions),
                  "reason_counts": dict(sorted(counts.items())), "decisions": decisions}
        if labels is not None:
            report["labelled_evaluation"] = score(decisions, labels)
        reports.append(report)
    result = {"scope": ("Labelled per-track material evaluation" if labels is not None
                        else "Unlabelled prediction replay; no accuracy claim"),
              "tape_sha256": manifest["tape_sha256"],
              "model_sha256": manifest["model_sha256"],
              "source_sha256": manifest["source_sha256"],
              "classes_sha256": manifest["classes_sha256"],
              "tracker_config_sha256": manifest["tracker_config_sha256"],
              "manifest_sha256": sha256(args.manifest),
              "policy_config_sha256": sha256(args.config),
              "label_sha256": sha256(args.labels) if args.labels else None,
              "policies": reports}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"scope": result["scope"], "policies": [
        {key: value for key, value in report.items() if key != "decisions"} for report in reports]}, indent=2))


if __name__ == "__main__":
    main()
