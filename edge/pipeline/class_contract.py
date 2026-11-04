"""Check detector class index order against the pinned evaluated class list."""

from __future__ import annotations

from pathlib import Path

import yaml


def validate_class_order(classes_file: str | Path, pinned_file: str | Path) -> list[str]:
    actual = [line.strip() for line in Path(classes_file).read_text().splitlines() if line.strip()]
    expected = (yaml.safe_load(Path(pinned_file).read_text()) or {}).get("names")
    if not isinstance(expected, list) or len(expected) != 28 or len(set(expected)) != 28:
        raise ValueError("Pinned class order must contain exactly 28 unique names")
    if len(actual) != len(expected):
        raise ValueError(f"Expected {len(expected)} classes, found {len(actual)}")
    for index, (want, got) in enumerate(zip(expected, actual)):
        if want != got:
            raise ValueError(f"Class-order mismatch at index {index}: expected {want!r}, found {got!r}")
    return actual
