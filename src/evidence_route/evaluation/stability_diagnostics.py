"""Deterministic, provider-free projection for frozen stability diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evidence_route.artifacts import atomic_write_json
from evidence_route.evaluation.stability import (
    RepeatSnapshot,
    StabilityCategory,
    StabilityClaimDiagnostic,
    StabilityDiagnosticSummary,
)

_CATEGORY_ORDER = {category: index for index, category in enumerate(StabilityCategory)}
_EXPECTED_REPEATS = 3
_VALID_FINAL_STATUSES = {"completed", "partial"}


def _complete_repeats(record: StabilityClaimDiagnostic) -> list[RepeatSnapshot]:
    repeats: dict[int, RepeatSnapshot] = {}
    for snapshot in record.repeats:
        if snapshot.repeat in repeats:
            raise ValueError(
                f"stability diagnostic has duplicate repeat {snapshot.repeat}: "
                f"{record.claim_id}"
            )
        repeats[snapshot.repeat] = snapshot
    return [
        repeats.get(repeat, RepeatSnapshot(repeat=repeat, valid=False))
        for repeat in range(_EXPECTED_REPEATS)
    ]


def _is_valid_final_repeat(repeat: RepeatSnapshot) -> bool:
    return (
        repeat.valid
        and repeat.status in _VALID_FINAL_STATUSES
        and repeat.verdict is not None
    )


def _ordered_categories(
    record: StabilityClaimDiagnostic,
    repeats: list[RepeatSnapshot],
) -> list[StabilityCategory]:
    if any(not _is_valid_final_repeat(repeat) for repeat in repeats):
        return [StabilityCategory.INCOMPLETE_OR_FAILED]
    categories = set(record.categories)
    return sorted(categories, key=_CATEGORY_ORDER.__getitem__)


def _is_consistent(record: StabilityClaimDiagnostic) -> bool:
    verdicts = {repeat.verdict for repeat in record.repeats}
    return (
        len(record.repeats) == 3
        and all(_is_valid_final_repeat(repeat) for repeat in record.repeats)
        and len(verdicts) == 1
    )


def project_stability_baseline(payload: Mapping[str, Any]) -> StabilityDiagnosticSummary:
    """Normalize a saved diagnostic without dropping incomplete repeat slots."""
    summary = StabilityDiagnosticSummary.model_validate(payload)
    records: list[StabilityClaimDiagnostic] = []
    claim_ids: set[str] = set()
    for record in sorted(summary.records, key=lambda item: item.claim_id):
        if record.claim_id in claim_ids:
            raise ValueError(f"stability diagnostic has duplicate claim ID: {record.claim_id}")
        claim_ids.add(record.claim_id)
        repeats = _complete_repeats(record)
        categories = _ordered_categories(record, repeats)
        differing_fields = (
            []
            if categories == [StabilityCategory.INCOMPLETE_OR_FAILED]
            else record.differing_fields
        )
        records.append(
            record.model_copy(
                update={
                    "repeats": repeats,
                    "differing_fields": differing_fields,
                    "categories": categories,
                    "primary_category": categories[0] if categories else None,
                }
            )
        )

    category_counts = {category.value: 0 for category in StabilityCategory}
    for record in records:
        for category in record.categories:
            category_counts[category.value] += 1

    repeat_schedule = {
        record.claim_id: list(range(_EXPECTED_REPEATS)) for record in records
    }
    if summary.repeat_schedule and summary.repeat_schedule != repeat_schedule:
        raise ValueError("stability diagnostic repeat schedule differs from repeat denominator")

    return summary.model_copy(
        update={
            "claim_count": len(records),
            "consistent_claim_count": sum(_is_consistent(record) for record in records),
            "category_counts": dict(sorted(category_counts.items())),
            "records": records,
            "repeat_schedule": repeat_schedule,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("stability diagnostic input must be a JSON object")
    summary = project_stability_baseline(raw)
    atomic_write_json(args.output, summary.model_dump(mode="json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "project_stability_baseline"]
