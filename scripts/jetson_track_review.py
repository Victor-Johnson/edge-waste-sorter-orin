"""Record per-frame track IDs and an annotated review video (not a speed benchmark)."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import ultralytics
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tracker_config_path(name: str) -> Path:
    path = Path(name)
    if path.is_file():
        return path
    built_in = Path(ultralytics.__file__).resolve().parent / "cfg" / "trackers" / name
    if built_in.is_file():
        return built_in
    raise FileNotFoundError(f"Tracker config missing: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--scene-frames", type=int, default=75)
    parser.add_argument("--confidence", type=float, default=.35)
    parser.add_argument("--limit", type=int, default=0, help="0 means all frames")
    args = parser.parse_args()
    for path in (args.model, args.video, args.classes):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    if args.scene_frames < 1 or not 0 <= args.confidence <= 1 or args.limit < 0:
        parser.error("scene-frames >= 1, confidence in [0,1], limit >= 0 required")
    tracker_path = tracker_config_path(args.tracker)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.out_prefix.with_suffix(".jsonl")
    video_path = args.out_prefix.with_suffix(".mp4")
    summary_path = args.out_prefix.with_suffix(".summary.json")
    names = [line.strip() for line in args.classes.read_text().splitlines() if line.strip()]
    model = YOLO(str(args.model), task="detect")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot write video: {video_path}")
    frame_count = 0
    detections = 0
    untracked = 0
    scene_ids: dict[int, set[int]] = {}
    try:
        with trace_path.open("w") as trace:
            while not args.limit or frame_count < args.limit:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.track(frame, persist=True, imgsz=640,
                                     conf=args.confidence, tracker=args.tracker,
                                     device=0, verbose=False)[0]
                result.names = dict(enumerate(names))
                frame_count += 1
                scene = (frame_count - 1) // args.scene_frames + 1
                scene_ids.setdefault(scene, set())
                objects = []
                annotated = frame.copy()
                for box in result.boxes:
                    track_id = int(box.id.item()) if box.id is not None else None
                    class_id = int(box.cls.item())
                    xyxy = [round(float(value), 1) for value in box.xyxy[0].tolist()]
                    confidence = round(float(box.conf.item()), 4)
                    objects.append({"track_id": track_id, "class_id": class_id,
                                    "class_name": names[class_id], "confidence": confidence,
                                    "bbox_xyxy": xyxy})
                    detections += 1
                    if track_id is None:
                        untracked += 1
                    else:
                        scene_ids[scene].add(track_id)
                    x1, y1, x2, y2 = (int(value) for value in xyxy)
                    colour = (40, 220, 80) if track_id is not None else (50, 100, 230)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                    label = f"ID {track_id if track_id is not None else '?'} | {names[class_id]}"
                    cv2.putText(annotated, label, (x1, max(50, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, .52, colour, 2)
                cv2.putText(annotated, f"scene {scene} | frame {frame_count} | {args.tracker}",
                            (12, height - 15), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                writer.write(annotated)
                trace.write(json.dumps({"frame_id": frame_count, "scene": scene,
                                        "objects": objects}) + "\n")
    finally:
        capture.release()
        writer.release()

    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "model": str(args.model.resolve()), "model_sha256": sha256(args.model),
              "source": str(args.video.resolve()), "source_sha256": sha256(args.video),
              "classes_sha256": sha256(args.classes), "tracker": args.tracker,
              "tracker_config_sha256": sha256(tracker_path),
              "confidence": args.confidence, "image_size": 640,
              "scene_frames": args.scene_frames, "frames": frame_count,
              "frame_level_detections": detections, "untracked_detections": untracked,
              "track_ids_per_scene": {str(scene): len(ids) for scene, ids in scene_ids.items()},
              "ultralytics": ultralytics.__version__,
              "trace": trace_path.name, "review_video": video_path.name,
              "scope": "Visual and ID review; MP4 writing included, so no speed claim"}
    summary_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
