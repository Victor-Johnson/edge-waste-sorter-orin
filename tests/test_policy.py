"""Material-policy behavior: routing, review reasons, and replay invariants."""

import json

import pytest

from edge.pipeline.policy import MaterialPolicy, PolicyConfig
from scripts.evaluate_policy import score, score_physical_items


def observation(track_id=7, lane="plastic", confidence=.8, name="bottle-blue"):
    return {"track_id": track_id, "lane": lane, "confidence": confidence,
            "class": name, "bbox_xyxy": [0, 0, 20, 20], "timestamp": 123.0}


def test_three_stable_observations_route_once_with_provenance():
    policy = MaterialPolicy(model_sha256="model", source_sha256="source")
    assert policy.process([observation()], 1) == []
    assert policy.process([observation()], 2) == []
    result = policy.process([observation()], 3)
    assert len(result) == 1
    assert {key: result[0][key] for key in ("status", "lane", "reason", "observations_used",
                                            "first_frame", "decision_frame", "model_sha256")} == {
        "status": "route", "lane": "plastic", "reason": "stable_material_evidence",
        "observations_used": 3, "first_frame": 1, "decision_frame": 3,
        "model_sha256": "model"}
    assert policy.process([observation()], 4) == []
    assert policy.finish() == []


def test_conflict_unknown_and_missing_id_review():
    policy = MaterialPolicy()
    policy.process([observation(1), observation(2)], 1)
    results = policy.process([observation(1, "glass", name="glass-transp"),
                              observation(2, "review", name="mystery"),
                              observation(None)], 2)
    assert {item["track_id"]: item["reason"] for item in results} == {
        None: "missing_track_id", 1: "conflicting_materials", 2: "unknown_class"}
    assert all(item["status"] == "review" for item in results)


def test_disappearance_timeout_and_low_confidence_review():
    policy = MaterialPolicy(PolicyConfig(min_observations=3, max_wait_frames=5,
                                         missing_grace_frames=1))
    policy.process([observation(1)], 1)
    policy.process([], 2)
    assert policy.process([], 3)[0]["reason"] == "track_disappeared"
    for frame in (4, 5, 6):
        policy.process([observation(2, confidence=.4)], frame)
    assert policy.process([], 7) == []
    assert policy.process([], 8)[0]["reason"] == "low_confidence"
    assert policy.process([observation(2)], 9) == []


def test_max_wait_and_same_frame_duplicate_vote():
    policy = MaterialPolicy(PolicyConfig(min_observations=3, max_wait_frames=3,
                                         missing_grace_frames=10))
    policy.process([observation(4, confidence=.7), observation(4, confidence=.8)], 1)
    policy.process([], 2)
    policy.process([], 3)
    result = policy.process([], 4)
    assert result[0]["reason"] == "insufficient_evidence"
    assert result[0]["observations_used"] == 1


def test_same_frame_material_conflict_and_late_return():
    policy = MaterialPolicy(PolicyConfig(missing_grace_frames=1))
    decision = policy.process([observation(1, "plastic"),
                               observation(1, "glass", name="glass-transp")], 1)
    assert decision[0]["reason"] == "conflicting_materials"
    policy.process([observation(2)], 2)
    late = policy.process([observation(2)], 5)
    assert len(late) == 1
    assert late[0]["reason"] == "track_disappeared"
    assert late[0]["observations_used"] == 1


def test_end_of_video_and_deterministic_replay():
    def replay():
        policy = MaterialPolicy()
        decisions = []
        for frame, events in enumerate(([observation(1)], [observation(1)], []), 1):
            decisions.extend(policy.process(events, frame))
        decisions.extend(policy.finish())
        return json.dumps(decisions, sort_keys=True)

    assert replay() == replay()
    assert json.loads(replay())[0]["reason"] == "track_disappeared"


def test_labelled_evaluation_counts_wrong_routes_and_missing():
    decisions = [{"track_id": 1, "status": "route", "lane": "glass"},
                 {"track_id": 2, "status": "review", "lane": "review"}]
    result = score(decisions, {1: "plastic", 2: "glass", 3: "metal"})
    assert result["automatic_coverage"] == .3333
    assert result["cross_material_error_per_route"] == 1.0
    assert result["material_confusion"] == {"plastic->glass": 1}
    assert result["missing_decisions"] == [3]


def test_physical_item_labels_group_split_track_ids_without_false_rates():
    decisions = [{"track_id": 1, "status": "review", "lane": "review"},
                 {"track_id": 7, "status": "route", "lane": "plastic"},
                 {"track_id": 9, "status": "route", "lane": "glass"}]
    labels = {"schema": "physical-items-v1", "complete": False,
              "items": [{"physical_id": "one-bottle", "lane": "plastic", "track_ids": [1, 7]}]}
    result = score_physical_items(decisions, labels)
    assert result["labelled_items"] == 1
    assert result["duplicate_decision_items"] == 1
    assert result["items_with_any_route"] == 1
    assert result["automatic_coverage"] is None
    assert result["cross_material_error_per_route"] is None
    assert result["unmatched_track_ids"] == [9]


def test_complete_physical_item_labels_report_cross_material_error():
    decisions = [{"track_id": 1, "status": "route", "lane": "glass"},
                 {"track_id": 7, "status": "route", "lane": "plastic"}]
    labels = {"schema": "physical-items-v1", "complete": True,
              "items": [{"physical_id": "one-bottle", "lane": "plastic", "track_ids": [1, 7]}]}
    result = score_physical_items(decisions, labels)
    assert result["automatic_coverage"] == 1.0
    assert result["cross_material_error_per_route"] == .5
    assert result["duplicate_route_items"] == 1
    assert result["material_confusion"] == {"plastic->glass": 1}


def test_complete_physical_labels_must_account_for_false_positive_tracks():
    decisions = [{"track_id": 8, "status": "route", "lane": "metal"}]
    labels = {"schema": "physical-items-v1", "complete": True,
              "items": [{"physical_id": "missed-bottle", "lane": "plastic", "track_ids": []}]}
    with pytest.raises(ValueError, match="unmatched track IDs"):
        score_physical_items(decisions, labels)
    labels["false_positive_track_ids"] = [8]
    report = score_physical_items(decisions, labels)
    assert report["missed_items"] == 1
    assert report["false_positive_route_proposals"] == 1
    assert report["automatic_coverage"] == 0.0


def test_invalid_policy_parameters():
    with pytest.raises(ValueError):
        PolicyConfig(min_observations=0)
    with pytest.raises(ValueError):
        PolicyConfig(min_mean_confidence=1.1)
    with pytest.raises(ValueError):
        PolicyConfig(max_wait_frames=2)
