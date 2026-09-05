from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "evaluation"


def test_quality_recovery_baseline_preserves_full_repeat_denominator(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "baseline.json"
    repository_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [str(repository_root / "src"), environment.get("PYTHONPATH")],
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evidence_route.evaluation.stability_diagnostics",
            "--input",
            str(FIXTURE_DIR / "quality_recovery_baseline_input.json"),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        cwd=repository_root,
        env=environment,
        text=True,
    )

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
