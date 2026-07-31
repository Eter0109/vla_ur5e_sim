"""Helpers for rejecting mixed experiment provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def immutable_mismatches(
    previous: Mapping[str, Any], current: Mapping[str, Any], fields: Sequence[str]
) -> list[str]:
    """Return locked fields whose values differ between two metadata records."""

    return [field for field in fields if previous.get(field) != current.get(field)]
