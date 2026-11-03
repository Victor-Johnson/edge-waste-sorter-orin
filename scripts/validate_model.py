"""Evaluate a checkpoint, ONNX graph, or TensorRT engine on one validation split."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/project.yaml").read_text())
    model_path = resolve(args.model)
    dataset = resolve(config["dataset_yaml"])
    if not dataset.exists():
        raise FileNotFoundError(f"Set dataset_yaml in configs/project.yaml: {dataset}")
    model = YOLO(str(model_path))
    settings = config["val"]
    metrics = model.val(data=str(dataset), imgsz=config["imgsz"], batch=settings["batch"],
                        rect=settings["rect"], split=settings["split"], conf=settings["conf"],
                        iou=settings["iou"], project=str(resolve(config["runs_dir"])),
                        name=model_path.stem, exist_ok=True, plots=False)
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "model": str(model_path), "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "dataset_yaml": str(dataset), "settings": {"imgsz": config["imgsz"], **settings},
              "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
              "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
              "per_class_mAP50_95": {str(metrics.names[int(c)]): float(metrics.box.maps[int(c)])
                                       for c in metrics.box.ap_class_index}}
    output = resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
