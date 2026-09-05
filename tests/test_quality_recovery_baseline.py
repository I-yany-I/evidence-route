from __future__ import annotations

import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from evidence_route.evaluation.stability_diagnostics import project_stability_baseline

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "evaluation"
INPUT_PATH = FIXTURE_DIR / "quality_recovery_baseline_input.json"
EXPERIMENT_DIR = FIXTURE_DIR / "quality-recovery-baseline-20260906"


def _input_payload() -> dict[str, object]:
    return json.loads(INPUT_PATH.read_text(encoding="utf-8"))


def _run_projection(output_path: Path) -> subprocess.CompletedProcess[str]:
    repository_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [str(repository_root / "src"), environment.get("PYTHONPATH")],
        )
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "evidence_route.evaluation.stability_diagnostics",
            "--input",
            str(INPUT_PATH),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        cwd=repository_root,
        env=environment,
        text=True,
    )


def test_quality_recovery_baseline_preserves_full_repeat_denominator(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "baseline.json"
    completed = _run_projection(output_path)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert set(payload) == {
        "activity_id",
        "campaign_id",
        "claim_count",
        "consistent_claim_count",
        "category_counts",
        "records",
        "repeat_schedule",
        "parent_activity_id",
    }
    assert payload["claim_count"] == 5
    assert payload["repeat_schedule"] == {
        claim_id: [0, 1, 2]
        for claim_id in [
            "claim-evidence",
            "claim-incomplete",
            "claim-provider",
            "claim-route",
            "claim-validation",
        ]
    }
    assert [record["claim_id"] for record in payload["records"]] == [
        "claim-evidence",
        "claim-incomplete",
        "claim-provider",
        "claim-route",
        "claim-validation",
    ]
    assert all(
        [repeat["repeat"] for repeat in record["repeats"]] == [0, 1, 2]
        for record in payload["records"]
    )
    assert sum(len(record["repeats"]) for record in payload["records"]) == 15

    statuses = [
        repeat["status"]
        for record in payload["records"]
        for repeat in record["repeats"]
    ]
    assert "partial" in statuses
    assert "failed" in statuses
    assert None in statuses
    assert payload["category_counts"] == {
        "evidence_or_citation_drift": 1,
        "incomplete_or_failed": 1,
        "provider_variance": 1,
        "route_drift": 1,
        "validation_or_status_drift": 1,
        "verdict_drift": 0,
    }


def test_incomplete_repeat_has_exclusive_category_priority() -> None:
    payload = _input_payload()
    evidence_record = next(
        record
        for record in payload["records"]  # type: ignore[index]
        if record["claim_id"] == "claim-evidence"
    )
    evidence_record["repeats"][1]["valid"] = False

    projected = project_stability_baseline(payload)
    record = next(record for record in projected.records if record.claim_id == "claim-evidence")

    assert [category.value for category in record.categories] == ["incomplete_or_failed"]
    assert record.differing_fields == []
    assert projected.category_counts["evidence_or_citation_drift"] == 0
    assert projected.category_counts["incomplete_or_failed"] == 2


@pytest.mark.parametrize("status", ["failed", "pending"])
def test_nonterminal_repeat_is_incomplete_and_not_consistent(status: str) -> None:
    payload = _input_payload()
    payload["records"] = [  # type: ignore[index]
        {
            "claim_id": "claim-status",
            "repeats": [
                {
                    "repeat": repeat,
                    "valid": True,
                    "status": status,
                    "verdict": "Supported",
                }
                for repeat in [2, 0, 1]
            ],
            "differing_fields": [],
            "categories": [],
            "primary_category": None,
        }
    ]

    projected = project_stability_baseline(payload)

    assert projected.consistent_claim_count == 0
    assert [category.value for category in projected.records[0].categories] == [
        "incomplete_or_failed"
    ]


def test_projection_rejects_repeat_schedule_that_disagrees_with_denominator() -> None:
    payload = _input_payload()
    payload["repeat_schedule"] = {  # type: ignore[index]
        record["claim_id"]: [0, 1, 2]
        for record in payload["records"]  # type: ignore[index]
    }
    payload["repeat_schedule"]["claim-incomplete"] = [0, 1]  # type: ignore[index]

    with pytest.raises(ValueError, match="repeat schedule"):
        project_stability_baseline(payload)


def test_frozen_diagnostic_bytes_and_experiment_hashes_are_reproducible(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "diagnostic.json"
    completed = _run_projection(output_path)
    assert completed.returncode == 0, completed.stderr

    frozen_output = EXPERIMENT_DIR / "diagnostic.json"
    identity = json.loads((EXPERIMENT_DIR / "experiment.json").read_text(encoding="utf-8"))
    assert output_path.read_bytes() == frozen_output.read_bytes()
    assert sha256(INPUT_PATH.read_bytes()).hexdigest() == identity["input_sha256"]
    assert sha256(frozen_output.read_bytes()).hexdigest() == identity["output_sha256"]
    assert identity["command"].startswith("python -m ")
    assert "C:\\Users" not in identity["command"]
