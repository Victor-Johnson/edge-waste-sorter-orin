"""Build a checked Jetson deployment manifest from a model and evidence JSONs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.pipeline.manifest import lanes_sha256, sha256, tracker_path, verify_deployment
from edge.pipeline.class_contract import validate_class_order
from edge.pipeline.policy import PolicyConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--demo-config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--policy-config", type=Path, default=Path("configs/policy.yaml"))
    parser.add_argument("--material-policy", action="store_true")
    parser.add_argument("--duplicate-gate", action="store_true")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    names = validate_class_order(args.classes, args.demo_config.parent / "class_order.yaml")
    lanes = (yaml.safe_load(args.demo_config.read_text()) or {}).get("lanes") or {}
    if len(names) != 28 or len(set(names)) != 28:
        raise ValueError("Expected 28 unique classes in model index order")
    if set(names) != set(lanes):
        raise ValueError(f"Lane map class mismatch: missing={sorted(set(names)-set(lanes))}, "
                         f"extra={sorted(set(lanes)-set(names))}")
    if any(lane not in {"plastic", "metal", "carton", "glass", "review"}
           for lane in lanes.values()):
        raise ValueError("Lane map contains an unknown material lane")
    validation = json.loads(args.validation.read_text())
    benchmark = json.loads(args.benchmark.read_text())
    model_hash = sha256(args.model)
    if validation.get("sha256") != model_hash or benchmark.get("sha256") != model_hash:
        raise ValueError("Model hash does not match validation and benchmark evidence")
    if validation.get("settings", {}).get("imgsz") != benchmark.get("image_size"):
        raise ValueError("Validation and benchmark use different image sizes")
    if benchmark.get("tracker") != args.tracker:
        raise ValueError("Benchmark tracker differs from requested deployment tracker")
    if args.material_policy:
        config = PolicyConfig(**(yaml.safe_load(args.policy_config.read_text()) or {}))
        policy_version = config.version
    else:
        policy_version = None
    manifest = {
        "format": "waste-sorter-deployment-v1",
        "model": args.model.name,
        "model_sha256": model_hash,
        "classes_sha256": sha256(args.classes),
        "lanes_sha256": lanes_sha256(lanes),
        "tracker": args.tracker,
        "tracker_config_sha256": sha256(tracker_path(args.tracker)),
        "policy_enabled": args.material_policy,
        "policy_version": policy_version,
        "policy_config_sha256": sha256(args.policy_config) if args.material_policy else None,
        "duplicate_gate_enabled": args.duplicate_gate,
        "validation_artifact_sha256": sha256(args.validation),
        "benchmark_artifact_sha256": sha256(args.benchmark),
        "validation_mAP50_95": validation.get("mAP50_95"),
        "benchmark_end_to_end_fps": benchmark.get("end_to_end_fps"),
        "validation_scope": validation.get("settings"),
        "benchmark_scope": benchmark.get("measurement_scope"),
        "environment": validation.get("environment"),
        "note": "Advisory visual decisions only; no physical actuator path",
    }
    verify_deployment(manifest, model=str(args.model), classes_file=str(args.classes),
                      lanes=lanes, tracker=args.tracker,
                      policy_enabled=args.material_policy,
                      policy_config_path=str(args.policy_config) if args.material_policy else None,
                      duplicate_gate_enabled=args.duplicate_gate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.out), "model_sha256": model_hash,
                      "policy_version": policy_version}, indent=2))


if __name__ == "__main__":
    main()
