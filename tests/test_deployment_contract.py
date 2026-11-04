"""Detect class-order and deployment-artifact mismatches before inference."""

import json
from pathlib import Path

import pytest

from edge.pipeline.class_contract import validate_class_order
from edge.pipeline.manifest import lanes_sha256, sha256, verify_deployment


ROOT = Path(__file__).resolve().parents[1]


def test_swapped_classes_identifies_first_bad_index(tmp_path):
    source = ROOT / "data/Warp-D/classes.txt"
    names = source.read_text().splitlines()
    names[25], names[26] = names[26], names[25]
    swapped = tmp_path / "classes.txt"
    swapped.write_text("\n".join(names) + "\n")
    with pytest.raises(ValueError, match="index 25.*glass-transp.*glass-dark"):
        validate_class_order(swapped, ROOT / "configs/class_order.yaml")
    assert len(validate_class_order(source, ROOT / "configs/class_order.yaml")) == 28


def test_manifest_rejects_wrong_engine_and_class_list(tmp_path):
    model = tmp_path / "candidate.engine"
    model.write_bytes(b"validated engine")
    classes = ROOT / "data/Warp-D/classes.txt"
    tracker = tmp_path / "tracker.yaml"
    tracker.write_text("tracker_type: botsort\n")
    lanes = {"bottle-blue": "plastic"}
    manifest = {"format": "waste-sorter-deployment-v1",
                "model_sha256": sha256(model), "classes_sha256": sha256(classes),
                "lanes_sha256": lanes_sha256(lanes),
                "tracker_config_sha256": sha256(tracker),
                "policy_config_sha256": None, "policy_enabled": False,
                "duplicate_gate_enabled": False}
    options = {"model": str(model), "classes_file": str(classes), "lanes": lanes,
               "tracker": str(tracker), "policy_enabled": False,
               "policy_config_path": None, "duplicate_gate_enabled": False}
    verify_deployment(manifest, **options)
    model.write_bytes(b"different engine")
    with pytest.raises(ValueError, match="model_sha256"):
        verify_deployment(manifest, **options)
    model.write_bytes(b"validated engine")
    with pytest.raises(ValueError, match="classes_sha256"):
        verify_deployment(manifest, **{**options, "classes_file": str(tracker)})


def test_manifest_rejects_policy_or_lane_change(tmp_path):
    model = tmp_path / "candidate.engine"
    model.write_bytes(b"engine")
    classes = ROOT / "data/Warp-D/classes.txt"
    tracker = tmp_path / "tracker.yaml"
    tracker.write_text("tracker_type: botsort\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: material-v1\n")
    lanes = {"bottle-blue": "plastic"}
    manifest = {"format": "waste-sorter-deployment-v1",
                "model_sha256": sha256(model), "classes_sha256": sha256(classes),
                "lanes_sha256": lanes_sha256(lanes),
                "tracker_config_sha256": sha256(tracker),
                "policy_config_sha256": sha256(policy), "policy_enabled": True,
                "duplicate_gate_enabled": False}
    options = {"model": str(model), "classes_file": str(classes), "lanes": lanes,
               "tracker": str(tracker), "policy_enabled": True,
               "policy_config_path": str(policy), "duplicate_gate_enabled": False}
    verify_deployment(manifest, **options)
    policy.write_text("version: material-v2\n")
    with pytest.raises(ValueError, match="policy_config_sha256"):
        verify_deployment(manifest, **options)
    policy.write_text("version: material-v1\n")
    with pytest.raises(ValueError, match="lanes_sha256"):
        verify_deployment(manifest, **{**options, "lanes": {"bottle-blue": "glass"}})
