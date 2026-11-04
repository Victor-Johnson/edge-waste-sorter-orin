"""Behavior checks for the optional duplicate-decision experiment."""

import json
from pathlib import Path

import pytest

from edge.pipeline.decision_gate import DuplicateDecisionGate, GateConfig, box_iou


def samples():
    path = Path(__file__).resolve().parents[1] / "benchmarks/annotations/decision_gate_pairs.json"
    return {row["name"]: row for row in json.loads(path.read_text())["pairs"]}


def test_hand_checked_duplicate_and_distinct_pairs():
    labelled = samples()
    positive = labelled["same_large_bottle_two_ids"]
    gate = DuplicateDecisionGate()
    assert [item["track_id"] for item in gate.process([positive["first"]], 1).decisions] == [1]
    batch = gate.process([positive["second"], positive["first"]], 2)
    assert batch.decisions == []
    assert [(item["track_id"], item["matched_track_id"]) for item in batch.suppressed] == [(7, 1)]
    assert gate.process([positive["second"]], 3).decisions == []

    for name in ("adjacent_dark_and_blue_bottles", "two_distinct_blue5l_bottles"):
        example = labelled[name]
        assert example["relation"] == "different_physical_objects"
        gate = DuplicateDecisionGate()
        assert len(gate.process([example["first"], example["second"]], 1).decisions) == 2


def test_returning_accepted_track_can_suppress_new_duplicate():
    positive = samples()["same_large_bottle_two_ids"]
    gate = DuplicateDecisionGate(GateConfig(max_track_gap_frames=2))
    gate.process([positive["first"]], 1)
    for frame in range(2, 9):
        gate.process([], frame)
    batch = gate.process([positive["second"], positive["first"]], 9)
    assert batch.decisions == []
    assert batch.suppressed[0]["matched_track_id"] == 1


def test_untracked_and_old_separate_tracks():
    positive = samples()["same_large_bottle_two_ids"]
    gate = DuplicateDecisionGate(GateConfig(max_track_gap_frames=1))
    assert gate.process([{**positive["first"], "track_id": None}], 1).decisions == []
    assert len(gate.process([positive["first"]], 2).decisions) == 1
    gate.process([], 3)
    batch = gate.process([positive["second"]], 4)
    assert [item["track_id"] for item in batch.decisions] == [7]


def test_iou_threshold_from_real_boxes():
    positive = samples()["same_large_bottle_two_ids"]
    distinct = samples()["adjacent_dark_and_blue_bottles"]
    assert box_iou(positive["first"]["bbox_xyxy"], positive["second"]["bbox_xyxy"]) >= .60
    assert box_iou(distinct["first"]["bbox_xyxy"], distinct["second"]["bbox_xyxy"]) < .60


def test_reused_id_with_changed_class_does_not_hide_new_item():
    positive = samples()["same_large_bottle_two_ids"]
    gate = DuplicateDecisionGate()
    gate.process([positive["first"]], 1)
    reused = {**positive["first"], "class": "cans", "lane": "metal"}
    new_can = {**positive["second"], "class": "cans", "lane": "metal"}
    batch = gate.process([reused, new_can], 2)
    assert [item["track_id"] for item in batch.decisions] == [7]
    assert batch.suppressed == []


def test_zero_iou_threshold_is_rejected():
    with pytest.raises(ValueError):
        GateConfig(min_iou=0)
