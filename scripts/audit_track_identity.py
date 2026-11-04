"""Audit a few manually identified objects against per-frame tracker traces.

Landmarks are points selected in the original video, before viewing tracker boxes.
This is a small visual audit, not a MOT benchmark or ground-truth identity score.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def overlap_iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def contains(box: list[float], point: list[int]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def audit_trace(trace_path: Path, annotations: dict, min_iou: float) -> dict:
    summary_path = trace_path.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    if summary["source_sha256"] != annotations["source_sha256"]:
        raise ValueError(f"Source hash mismatch for {trace_path}")
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    frames = {row["frame_id"]: row for row in rows}
    if len(frames) != len(rows) or len(rows) != summary["frames"]:
        raise ValueError(f"Trace frame count mismatch for {trace_path}")
    first_sighting = {}
    for row in rows:
        for box in row["objects"]:
            track_id = box["track_id"]
            if track_id is not None:
                first_sighting.setdefault(track_id, row["frame_id"])

    objects = []
    for annotated in annotations["objects"]:
        observations = []
        for landmark in annotated["landmarks"]:
            frame_id = landmark["frame_id"]
            if frame_id not in frames:
                raise ValueError(f"Missing frame {frame_id} in {trace_path}")
            point = landmark["point_xy"]
            matches = [box for box in frames[frame_id]["objects"]
                       if contains(box["bbox_xyxy"], point)]
            observations.append({
                "frame_id": frame_id,
                "point_xy": point,
                "track_ids_covering_point": sorted({box["track_id"] for box in matches
                                                    if box["track_id"] is not None}),
                "classes_covering_point": sorted({box["class_name"] for box in matches}),
            })
        observed_ids = sorted({track_id for observation in observations
                               for track_id in observation["track_ids_covering_point"]})
        objects.append({
            "object_id": annotated["object_id"],
            "description": annotated["description"],
            "observed_track_ids": observed_ids,
            "first_sighting_frame_by_track_id": {
                str(track_id): first_sighting[track_id] for track_id in observed_ids},
            "missing_frames": [observation["frame_id"] for observation in observations
                               if not observation["track_ids_covering_point"]],
            "multiple_id_frames": [observation["frame_id"] for observation in observations
                                   if len(observation["track_ids_covering_point"]) > 1],
            "observations": observations,
        })

    overlaps: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        boxes = row["objects"]
        for index, first in enumerate(boxes):
            for second in boxes[index + 1:]:
                if (first["track_id"] is None or second["track_id"] is None
                        or first["track_id"] == second["track_id"]
                        or first["class_id"] != second["class_id"]):
                    continue
                iou = overlap_iou(first["bbox_xyxy"], second["bbox_xyxy"])
                if iou >= min_iou:
                    pair = tuple(sorted((first["track_id"], second["track_id"])))
                    overlaps[pair].append((row["frame_id"], iou))
    candidates = [{
        "track_ids": list(pair),
        "frames": [frame_id for frame_id, _ in samples],
        "overlap_frame_count": len(samples),
        "median_iou": round(statistics.median(score for _, score in samples), 3),
    } for pair, samples in sorted(overlaps.items())]

    return {"trace": str(trace_path), "tracker": summary["tracker"],
            "tracker_config_sha256": summary["tracker_config_sha256"],
            "frames": len(rows), "frame_level_detections": summary["frame_level_detections"],
            "objects": objects, "same_class_overlap_candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.min_iou <= 1:
        parser.error("--min-iou must be between 0 and 1")
    annotations = json.loads(args.annotations.read_text())
    if not annotations.get("objects"):
        parser.error("Annotations must contain at least one object")
    variants = [audit_trace(trace, annotations, args.min_iou) for trace in args.trace]
    result = {"stage": "manual_landmark_track_audit",
              "annotation_file": str(args.annotations),
              "source_sha256": annotations["source_sha256"],
              "same_class_overlap_iou_threshold": args.min_iou,
              "method_note": "A landmark point is manually placed on one visible physical object in selected source frames. Multiple covering track IDs flag a possible duplicate proposed decision. IoU candidates require visual review; this is not an ID-switch rate or MOT score.",
              "variants": variants}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
