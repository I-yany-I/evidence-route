"""Factories for deterministic evaluation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from evidence_route.contracts import ResultStatus, Usage, Verdict, VerificationResult
from evidence_route.evaluation.runtime_manifest import load_runtime_manifest

FIXTURE_ROOT = Path(__file__).parent


def _result(claim_id: str, verdict: Verdict, status: ResultStatus = ResultStatus.COMPLETED):
    if status is ResultStatus.FAILED:
        return VerificationResult(
            claim_id=claim_id,
            status=status,
            rationale="transport failed",
            initial_route="single",
            usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False),
        )
    return VerificationResult(
        claim_id=claim_id,
        status=status,
        verdict=verdict,
        confidence=0.8,
        rationale="structured result",
        initial_route="single",
        usage=Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True),
    )


@pytest.fixture
def runtime_claims():
    return load_runtime_manifest(
        FIXTURE_ROOT / "runtime.json", allowed_root=FIXTURE_ROOT
    ).items


@pytest.fixture
def completed_results():
    return {
        "dev-0": _result("dev-0", Verdict.SUPPORTED),
        "dev-1": _result("dev-1", Verdict.REFUTED),
    }
