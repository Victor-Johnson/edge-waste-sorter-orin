"""Import an existing validation or benchmark JSON into MLflow."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.tracking import log_record, tracking_uri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--stage", choices=["validation", "benchmark", "engine_benchmark"], required=True)
    args = parser.parse_args()
    path = Path(args.file).resolve()
    record = json.loads(path.read_text())
    run_id = log_record(record, path, args.stage)
    print(f"Run ID: {run_id}\nTracking URI: {tracking_uri()}")


if __name__ == "__main__":
    main()
