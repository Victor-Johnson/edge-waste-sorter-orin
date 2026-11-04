"""Hash-checked deployment contract for an engine and its decision settings."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tracker_path(name: str) -> Path:
    path = Path(name)
    if path.is_file():
        return path
    spec = importlib.util.find_spec("ultralytics")
    if spec is not None and spec.origin:
        built_in = Path(spec.origin).parent / "cfg" / "trackers" / name
        if built_in.is_file():
            return built_in
    raise FileNotFoundError(f"Tracker config missing: {name}")


def lanes_sha256(lanes: dict[str, str]) -> str:
    payload = json.dumps(lanes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_deployment(manifest: dict, *, model: str, classes_file: str | None,
                      lanes: dict[str, str], tracker: str, policy_enabled: bool,
                      policy_config_path: str | None,
                      duplicate_gate_enabled: bool) -> None:
    if manifest.get("format") != "waste-sorter-deployment-v1":
        raise ValueError("Unsupported deployment manifest format")
    checks = {
        "model_sha256": sha256(model),
        "classes_sha256": sha256(classes_file) if classes_file else None,
        "lanes_sha256": lanes_sha256(lanes),
        "tracker_config_sha256": sha256(tracker_path(tracker)),
        "policy_config_sha256": (sha256(policy_config_path)
                                 if policy_enabled and policy_config_path else None),
        "policy_enabled": policy_enabled,
        "duplicate_gate_enabled": duplicate_gate_enabled,
    }
    for field, actual in checks.items():
        if manifest.get(field) != actual:
            raise ValueError(f"Deployment mismatch for {field}: expected {manifest.get(field)!r}, "
                             f"got {actual!r}")
