"""Factories for deterministic evaluation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from evidence_route.contracts import (
    Citation,
    ClaimFeatures,
    ClaimUnit,
    ResultStatus,
    Usage,
    Verdict,
    VerificationResult,
)
from evidence_route.evaluation.calibration import CalibrationRuntimeCase, CalibrationScoredCase
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
    return load_runtime_manifest(FIXTURE_ROOT / "runtime.json", allowed_root=FIXTURE_ROOT).items


@pytest.fixture
def completed_results():
    return {
        "dev-0": _result("dev-0", Verdict.SUPPORTED),
        "dev-1": _result("dev-1", Verdict.REFUTED),
    }


@pytest.fixture
def calibration_case() -> CalibrationScoredCase:
    citation = Citation(
        evidence_id="e0",
        claim_unit_ids=["u0"],
        question="What happened?",
        answer="The source confirms the claim.",
        quote="The source confirms the claim.",
        stance="supports",
        source_url="https://example.org/source",
    )
    features = ClaimFeatures(
        claim_units=[ClaimUnit(unit_id="u0", text="Atomic claim")],
        atomic_clause_count=1,
        entity_count=1,
        numeric_count=0,
        time_scope_count=0,
        has_comparison=False,
        has_causal=False,
        has_contrast=False,
        probe_source_count=2,
        probe_score_spread=0.1,
        probe_conflict_hint=False,
    )
    usage = Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True)
    single = VerificationResult(
        claim_id="train-0",
        status=ResultStatus.COMPLETED,
        verdict=Verdict.SUPPORTED,
        confidence=0.9,
        rationale="single result",
        citations=[citation],
        available_evidence_ids=["e0"],
        initial_route="single",
        usage=usage,
    )
    multi = VerificationResult(
        claim_id="train-0",
        status=ResultStatus.COMPLETED,
        verdict=Verdict.SUPPORTED,
        confidence=0.95,
        rationale="multi result",
        citations=[citation],
        available_evidence_ids=["e0"],
        initial_route="multi",
        usage=Usage(input_tokens=16, output_tokens=4, total_tokens=20, complete=True),
    )
    runtime = CalibrationRuntimeCase(
        claim_id="train-0",
        case_id="a" * 64,
        router_run_id="b" * 64,
        single_run_id="c" * 64,
        multi_run_id="d" * 64,
        call_ids=["router-0", "single-0", "multi-0"],
        request_sha256_by_call_id={
            "router-0": "1" * 64,
            "single-0": "2" * 64,
            "multi-0": "3" * 64,
        },
        runtime_manifest_sha256="e" * 64,
        features=features,
        saved_llm_route="single",
        router_usage=Usage(input_tokens=4, output_tokens=1, total_tokens=5, complete=True),
        router_actual_cost_micro_cny=0,
        single_result=single,
        multi_result=multi,
        requested_alias="fixture-model",
        response_model_ids_raw=["fixture-model"],
        artifact_sha256="f" * 64,
    )
    return CalibrationScoredCase(runtime=runtime, gold_label=Verdict.SUPPORTED)
