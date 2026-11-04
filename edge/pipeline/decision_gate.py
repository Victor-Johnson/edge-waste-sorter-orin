"""Conservative, optional spatial duplicate filter for proposed decisions.

The filter only considers a *new* track ID a duplicate when it strongly overlaps
a recently accepted track of the same predicted class. It does not change model
boxes or tracker IDs. Use it as an experiment until identity labels are broader.
"""

from __future__ import annotations

from dataclasses import dataclass


def box_iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


@dataclass(frozen=True)
class GateConfig:
    min_iou: float = 0.60
    max_track_gap_frames: int = 5

    def __post_init__(self) -> None:
        if not 0.0 < self.min_iou <= 1.0:
            raise ValueError("min_iou must be in (0, 1]")
        if self.max_track_gap_frames < 0:
            raise ValueError("max_track_gap_frames must be nonnegative")


@dataclass(frozen=True)
class DecisionBatch:
    decisions: list[dict]
    suppressed: list[dict]


class DuplicateDecisionGate:
    """Emit a first-ID decision unless a strong same-class overlap is present.

    A suppressed ID stays suppressed for the remainder of this stream. This
    deliberately favors avoiding an extra proposal and can miss a distinct item
    that enters while overlapping another. The gate must remain opt-in until
    that trade-off is measured on more labelled moving objects.
    """

    def __init__(self, config: GateConfig = GateConfig()) -> None:
        self.config = config
        self.seen_ids: set[int] = set()
        self.approved_ids: set[int] = set()
        self.approved_class: dict[int, str] = {}
        self.accepted: dict[int, tuple[dict, int]] = {}

    def process(self, events: list[dict], frame_id: int) -> DecisionBatch:
        if frame_id < 1:
            raise ValueError("frame_id must be positive")
        for event in events:
            track_id = event["track_id"]
            if track_id in self.approved_ids:
                if event["class"] == self.approved_class[track_id]:
                    self.accepted[track_id] = (event, frame_id)
                else:
                    # A reused ID with a new class must not suppress that class's new ID.
                    self.accepted.pop(track_id, None)
        self.accepted = {track_id: item for track_id, item in self.accepted.items()
                         if frame_id - item[1] <= self.config.max_track_gap_frames}

        decisions: list[dict] = []
        suppressed: list[dict] = []
        new_events = (event for event in events if event["track_id"] is not None
                      and event["track_id"] not in self.seen_ids)
        for event in sorted(new_events, key=lambda value: -value.get("confidence", 0.0)):
            track_id = event["track_id"]
            if track_id in self.seen_ids:
                continue
            best_match: tuple[int, float] | None = None
            for accepted_id, (reference, _) in self.accepted.items():
                if event["class"] != reference["class"]:
                    continue
                iou = box_iou(event["bbox_xyxy"], reference["bbox_xyxy"])
                if iou >= self.config.min_iou and (best_match is None or iou > best_match[1]):
                    best_match = (accepted_id, iou)
            self.seen_ids.add(track_id)
            if best_match is None:
                self.approved_ids.add(track_id)
                self.approved_class[track_id] = event["class"]
                self.accepted[track_id] = (event, frame_id)
                decisions.append(event)
            else:
                suppressed.append({"track_id": track_id,
                                   "matched_track_id": best_match[0],
                                   "frame_id": frame_id,
                                   "class": event["class"],
                                   "iou": round(best_match[1], 4),
                                   "reason": "same_class_spatial_duplicate_candidate"})
        return DecisionBatch(decisions, suppressed)
