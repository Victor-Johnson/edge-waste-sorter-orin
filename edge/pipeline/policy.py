"""Deterministic, advisory material-routing policy over tracked observations.

Confidence is an uncalibrated model score. This policy makes no safety or
accuracy guarantee; thresholds must be selected with labelled validation data.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyConfig:
    version: str = "material-v1"
    min_observations: int = 3
    max_wait_frames: int = 10
    missing_grace_frames: int = 2
    min_mean_confidence: float = 0.55

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("policy version is required")
        if self.min_observations < 1:
            raise ValueError("min_observations must be positive")
        if self.max_wait_frames < self.min_observations:
            raise ValueError("max_wait_frames must be >= min_observations")
        if self.missing_grace_frames < 0:
            raise ValueError("missing_grace_frames must be nonnegative")
        if not 0 <= self.min_mean_confidence <= 1:
            raise ValueError("min_mean_confidence must be in [0, 1]")


@dataclass
class _Track:
    first_frame: int
    last_frame: int
    observations: list[dict] = field(default_factory=list)


class MaterialPolicy:
    """Consume one complete frame at a time and emit immutable proposals."""

    def __init__(self, config: PolicyConfig = PolicyConfig(), *,
                 model_sha256: str | None = None,
                 source_sha256: str | None = None) -> None:
        self.config = config
        self.model_sha256 = model_sha256
        self.source_sha256 = source_sha256
        self.active: dict[int, _Track] = {}
        self.completed: set[int] = set()
        self.last_frame = 0

    def _decision(self, track_id: int | None, track: _Track, frame_id: int,
                  reason: str, lane: str = "review") -> dict:
        observations = track.observations
        return {
            "timestamp": observations[-1].get("timestamp"),
            "track_id": track_id,
            "policy_version": self.config.version,
            "status": "review" if lane == "review" else "route",
            "lane": lane,
            "reason": reason,
            "observations_used": len(observations),
            "first_frame": track.first_frame,
            "decision_frame": frame_id,
            "confidence": round(sum(float(x["confidence"]) for x in observations)
                                / len(observations), 4),
            "class": observations[-1]["class"],
            "bbox_xyxy": observations[-1].get("bbox_xyxy"),
            "last_observation_frame": track.last_frame,
            "model_sha256": self.model_sha256,
            "source_sha256": self.source_sha256,
        }

    def _resolve(self, track_id: int, track: _Track, frame_id: int,
                 *, expired: bool) -> dict | None:
        lanes = {item["lane"] for item in track.observations}
        if "review" in lanes:
            return self._decision(track_id, track, frame_id, "unknown_class")
        if len(lanes) > 1:
            return self._decision(track_id, track, frame_id, "conflicting_materials")
        if len(track.observations) >= self.config.min_observations:
            mean = sum(float(x["confidence"]) for x in track.observations) / len(track.observations)
            if mean >= self.config.min_mean_confidence:
                return self._decision(track_id, track, frame_id,
                                      "stable_material_evidence", next(iter(lanes)))
            if expired:
                return self._decision(track_id, track, frame_id, "low_confidence")
        if expired:
            reason = ("insufficient_evidence" if frame_id - track.first_frame >= self.config.max_wait_frames
                      else "track_disappeared")
            return self._decision(track_id, track, frame_id, reason)
        return None

    def process(self, events: list[dict], frame_id: int) -> list[dict]:
        if frame_id <= self.last_frame:
            raise ValueError("frame_id must increase strictly")
        self.last_frame = frame_id
        decisions: list[dict] = []
        # Expire before accepting this frame. A stale ID returning after a gap
        # must not retroactively make an old track look stable.
        for track_id in sorted(list(self.active)):
            track = self.active[track_id]
            expired = (frame_id - track.first_frame >= self.config.max_wait_frames
                       or frame_id - track.last_frame > self.config.missing_grace_frames)
            if expired:
                decisions.append(self._resolve(track_id, track, frame_id, expired=True))
                self.completed.add(track_id)
                del self.active[track_id]
        # One vote per ID per frame. Keep the strongest box if a tracker emits
        # duplicates. Different lanes under one ID in a frame trigger review.
        by_id: dict[int, dict] = {}
        frame_lanes: dict[int, set[str]] = {}
        untracked: list[dict] = []
        for event in events:
            track_id = event.get("track_id")
            if track_id is None:
                untracked.append(event)
            elif track_id not in self.completed:
                frame_lanes.setdefault(track_id, set()).add(event["lane"])
                previous = by_id.get(track_id)
                if previous is None or float(event["confidence"]) > float(previous["confidence"]):
                    by_id[track_id] = event

        for event in untracked:
            track = _Track(frame_id, frame_id, [event])
            decisions.append(self._decision(None, track, frame_id, "missing_track_id"))
        for track_id in sorted(by_id):
            event = by_id[track_id]
            track = self.active.setdefault(track_id, _Track(frame_id, frame_id))
            track.last_frame = frame_id
            track.observations.append(event)
            decision = (self._decision(track_id, track, frame_id, "conflicting_materials")
                        if len(frame_lanes[track_id]) > 1 else
                        self._resolve(track_id, track, frame_id, expired=False))
            if decision is not None:
                decisions.append(decision)
                self.completed.add(track_id)
                del self.active[track_id]

        return decisions

    def finish(self) -> list[dict]:
        """Close tracks left open when a finite video ends."""
        decisions = []
        for track_id in sorted(self.active):
            track = self.active[track_id]
            decisions.append(self._resolve(track_id, track, self.last_frame, expired=True))
            self.completed.add(track_id)
        self.active.clear()
        return decisions
