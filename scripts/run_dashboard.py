"""Run the edge dashboard: python scripts/run_dashboard.py --source 0"""

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.dashboard import create_app
from edge.pipeline.core import load_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--source", help="Camera index or video path")
    parser.add_argument("--model", help="PyTorch checkpoint or Jetson TensorRT engine")
    parser.add_argument("--tracker", help="Tracker config name or YAML path")
    parser.add_argument("--duplicate-gate", action="store_true",
                        help="Enable experimental same-class duplicate-decision filter")
    parser.add_argument("--material-policy", action="store_true",
                        help="Enable multi-frame advisory route/review policy")
    parser.add_argument("--manifest", help="Reject engine/config mismatches using a deployment manifest")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = load_settings(args.config, model_override=args.model,
                             source_override=args.source, tracker_override=args.tracker,
                             duplicate_gate_override=True if args.duplicate_gate else None,
                             policy_override=True if args.material_policy else None,
                             manifest_override=args.manifest)
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
