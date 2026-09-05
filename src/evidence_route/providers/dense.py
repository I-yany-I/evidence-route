"""Interfaces for deterministic local dense retrieval encoders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class DenseEncoder(Protocol):
    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        raise NotImplementedError


__all__ = ["DenseEncoder"]
