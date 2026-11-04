from types import SimpleNamespace

import pytest

from edge.pipeline.core import detection_events, load_settings, new_decisions
from scripts.benchmark import percentile


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Coords:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def test_detection_event_and_review_fallback():
    box = SimpleNamespace(cls=Scalar(1), conf=Scalar(.82), id=Scalar(9),
                          xyxy=[Coords([1, 2, 20, 30])])
    result = SimpleNamespace(boxes=[box], names={1: "plastic"})
    event = detection_events(result, {}, 4, 123.0)[0]
    assert event == {"timestamp": 123.0, "frame_id": 4, "track_id": 9,
                     "class": "plastic", "confidence": .82,
                     "bbox_xyxy": [1.0, 2.0, 20.0, 30.0], "lane": "review"}
    assert detection_events(result, {"plastic": "recycle"}, 4, 123.0)[0]["lane"] == "recycle"


def test_settings_reject_missing_model(tmp_path):
    config = tmp_path / "demo.yaml"
    config.write_text("model: missing.pt\nsource: '0'\n")
    with pytest.raises(FileNotFoundError):
        load_settings(config)


def test_settings_resolve_overrides_before_validation(tmp_path):
    config = tmp_path / "configs" / "demo.yaml"
    config.parent.mkdir()
    config.write_text("model: missing.pt\nsource: missing.mp4\n")
    model = tmp_path / "chosen.engine"
    video = tmp_path / "replay.mp4"
    model.touch()
    video.touch()
    settings = load_settings(config, model_override="chosen.engine",
                             source_override="replay.mp4")
    assert settings.model == str(model)
    assert settings.source == str(video)


def test_percentile():
    assert percentile([9, 1, 5], .5) == 5
    assert percentile([9, 1], .95) == 9


def test_new_decisions_across_frames():
    seen = set()
    first = [{"track_id": 7}, {"track_id": 7}, {"track_id": None}, {"track_id": 8}]
    assert new_decisions(first, seen) == [first[0], first[3]]
    assert seen == {7, 8}
    second = [{"track_id": 7}, {"track_id": 9}]
    assert new_decisions(second, seen) == [second[1]]
    assert seen == {7, 8, 9}
