"""Build TensorRT engines on the Jetson from the trained checkpoint."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/detection/02_yolo26m_best.pt")
    parser.add_argument("--precision", choices=["fp16", "int8"], required=True)
    parser.add_argument("--dataset", help="Required for INT8 calibration; YOLO dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        parser.error(f"Missing checkpoint: {checkpoint}")
    if args.precision == "int8" and not args.dataset:
        parser.error("--dataset is required for representative INT8 calibration")
    dataset = Path(args.dataset).resolve() if args.dataset else None
    if dataset and not dataset.exists():
        parser.error(f"Missing dataset YAML: {dataset}")
    options = {"format": "engine", "imgsz": args.imgsz, "batch": 1, "dynamic": False,
               "half": args.precision == "fp16", "int8": args.precision == "int8",
               "device": 0}
    if dataset:
        options["data"] = str(dataset)
    engine = Path(YOLO(str(checkpoint)).export(**options))
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint": str(checkpoint),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "engine": str(engine), "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
              "precision": args.precision, "dataset_yaml": str(dataset) if dataset else None,
              "settings": options, "platform": platform.platform()}
    output = ROOT / "benchmarks/results" / f"build_{args.precision}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
