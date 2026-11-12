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
    return {"label_unit": "track_id", "labelled_tracks": len(labels), "missing_decisions": missing,
            "automatic_coverage": round(len(routed) / len(labels), 4) if labels else None,
            "cross_material_error_per_route": round(len(wrong) / len(routed), 4) if routed else None,
            "wrong_routes": len(wrong),
            "material_confusion": dict(sorted(Counter(f"{true}->{proposed}" for true, proposed in wrong).items()))}


def score_physical_items(decisions: list[dict], labels: dict) -> dict:
    """Score proposals by human-labelled physical item, preserving ID splits."""
    if labels.get("schema") != "physical-items-v1":
        raise ValueError("Physical labels need schema physical-items-v1")
    if not isinstance(labels.get("complete"), bool):
        raise ValueError("Physical labels must declare complete: true or false")
    items = labels.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Physical labels need at least one item")
    known_lanes = {"plastic", "metal", "carton", "glass"}
    track_to_item: dict[int, str] = {}
    physical_ids: set[str] = set()
    for item in items:
        physical_id = str(item["physical_id"])
        if not physical_id or physical_id in physical_ids:
            raise ValueError(f"Duplicate or empty physical ID: {physical_id!r}")
        physical_ids.add(physical_id)
        if item["lane"] not in known_lanes:
            raise ValueError(f"Unknown material lane for {physical_id}")
        if not isinstance(item.get("track_ids"), list):
            raise ValueError(f"track_ids must be a list for {physical_id}")
        for track_id in item["track_ids"]:
            if not isinstance(track_id, int) or track_id in track_to_item:
                raise ValueError(f"Track ID {track_id!r} is duplicated or invalid")
            track_to_item[track_id] = physical_id
    by_track = {item["track_id"]: item for item in decisions if item["track_id"] is not None}
    detail = []
    wrong_pairs = Counter()
    routed_items = 0
    single_route_items = 0
    duplicate_decision_items = 0
    duplicate_route_items = 0
    missed_items = 0
    reviewed_only_items = 0
    wrong_routes = 0
    all_routes = 0
    for item in items:
        proposals = [by_track[track_id] for track_id in item["track_ids"] if track_id in by_track]
        routes = [proposal for proposal in proposals if proposal["status"] == "route"]
        missed_items += len(proposals) == 0
        reviewed_only_items += bool(proposals) and not routes
        routed_items += bool(routes)
        single_route_items += len(routes) == 1
        duplicate_decision_items += len(proposals) > 1
        duplicate_route_items += len(routes) > 1
        all_routes += len(routes)
        for route in routes:
            if route["lane"] != item["lane"]:
                wrong_routes += 1
                wrong_pairs[f"{item['lane']}->{route['lane']}"] += 1
        detail.append({"physical_id": item["physical_id"], "true_lane": item["lane"],
                       "track_ids": item["track_ids"], "proposal_count": len(proposals),
                       "route_count": len(routes),
                       "review_count": len(proposals) - len(routes)})
    complete = labels["complete"]
    unmatched = sorted(track_id for track_id in by_track if track_id not in track_to_item)
    declared_false_positives = labels.get("false_positive_track_ids", [])
    if complete and sorted(declared_false_positives) != unmatched:
        raise ValueError("Complete labels must account for all unmatched track IDs "
                         "in false_positive_track_ids")
    return {"label_unit": "physical_item", "labels_complete": complete,
            "labelled_items": len(items), "items_with_any_route": routed_items,
            "items_with_single_route": single_route_items,
            "missed_items": missed_items, "reviewed_only_items": reviewed_only_items,
            "duplicate_decision_items": duplicate_decision_items,
            "duplicate_route_items": duplicate_route_items,
            "false_positive_route_proposals": sum(by_track[track_id]["status"] == "route"
                                                 for track_id in declared_false_positives
                                                 if track_id in by_track),
            "route_proposals": all_routes, "wrong_route_proposals": wrong_routes,
            "automatic_coverage": round(routed_items / len(items), 4) if complete else None,
            "cross_material_error_per_route": (round(wrong_routes / all_routes, 4)
                                               if complete and all_routes else None),
            "material_confusion": dict(sorted(wrong_pairs.items())),
            "unmatched_track_ids": unmatched,
            "items": detail}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/policy.yaml"))
    parser.add_argument("--classes", type=Path, help="Verify current class list against tape")
    parser.add_argument("--labels", type=Path,
                        help="JSON track-to-lane mapping or physical-items-v1 annotation")
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
        if raw.get("schema") == "physical-items-v1":
            if raw.get("source_sha256") != manifest["source_sha256"]:
                raise ValueError("Physical-label source hash does not match tape")
            labels = raw
        else:
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
            report["labelled_evaluation"] = (score_physical_items(decisions, labels)
                                              if labels.get("schema") == "physical-items-v1"
                                              else score(decisions, labels))
        reports.append(report)
    result = {"scope": ("Partial physical-item review; no general accuracy claim"
                        if labels is not None and labels.get("schema") == "physical-items-v1"
                        and not labels["complete"] else
                        "Labelled material evaluation" if labels is not None else
                        "Unlabelled prediction replay; no accuracy claim"),
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
