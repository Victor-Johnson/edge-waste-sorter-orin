from types import SimpleNamespace

import pytest

from edge.pipeline.core import detection_events, load_settings
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


def test_percentile():
    assert percentile([9, 1, 5], .5) == 5
    assert percentile([9, 1], .95) == 9
