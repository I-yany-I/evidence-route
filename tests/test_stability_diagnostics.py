from __future__ import annotations

import pytest
from pydantic import ValidationError

from evidence_route.contracts import Citation, ResultStatus, Usage, Verdict, VerificationResult
from evidence_route.evaluation.activity import LatencyBreakdown, RunArtifact
from evidence_route.evaluation.stability import (
    RepeatSnapshot,
    StabilityCategory,
    StabilityClaimDiagnostic,
    StabilityDiagnosticSummary,
    canonicalize_citation_url,
    citation_is_valid,
    citation_urls,
    normalize_result,
)


def _citation(*, evidence_id: str, url: str) -> Citation:
    return Citation(
        evidence_id=evidence_id,
        claim_unit_ids=["u0"],
        question="Which source supports the claim?",
        answer="The source supports the claim.",
        quote="The claim is supported.",
        stance="supports",
        source_url=url,
    )


def _result(
    *,
    citations: list[Citation] | None = None,
    available_evidence_ids: list[str] | None = None,
    errors: list[str] | None = None,
    status: ResultStatus = ResultStatus.COMPLETED,
) -> VerificationResult:
    return VerificationResult(
        claim_id="claim-0",
        status=status,
        verdict=None if status is ResultStatus.FAILED else Verdict.SUPPORTED,
        confidence=None if status is ResultStatus.FAILED else 0.9,
        rationale="fixture result",
        citations=[] if citations is None else citations,
        available_evidence_ids=(
            [" e2", "e1", "e1 "]
            if available_evidence_ids is None
            else available_evidence_ids
        ),
        initial_route=None if status is ResultStatus.FAILED else "single",
        failure_stage="pre_route" if status is ResultStatus.FAILED else None,
        usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5, complete=True),
        estimated_cost_micro_cny=42,
        cost_currency="CNY",
        price_config_id="c" * 64,
        latency_ms=17,
        errors=[] if errors is None else errors,
    )


def _artifact(result: VerificationResult | None = None) -> RunArtifact:
    result = result or _result()
    return RunArtifact(
        schema_version="1",
        activity_id="activity",
        campaign_id="campaign",
        phase="stability",
        run_id="a" * 64,
        claim_id=result.claim_id,
        strategy="adaptive",
        repeat=1,
        result=result,
        call_ids=["call-0"],
        usage=result.usage,
        usage_source="provider",
        actual_cost_micro_cny=42,
        known_actual_cost_micro_cny=42,
        committed_cost_micro_cny=42,
        cost_is_lower_bound=False,
        fresh_call_count=1,
        cache_hit_count=0,
        requested_alias="relay",
        response_model_ids_raw=["model-id"],
        billing_uncertain=False,
        latency=LatencyBreakdown(
            fresh_end_to_end_ms=31,
            model_active_ms=20,
            retry_ms=0,
            queue_ms=11,
            checkpoint_downtime_ms=0,
            total_elapsed_ms=31,
            interruption_count=0,
        ),
        artifact_sha256="b" * 64,
    )


def test_canonicalize_citation_url_removes_fragment_default_port_and_trailing_slash() -> None:
    assert canonicalize_citation_url(
        "HTTPS://Example.COM:443/fact/?b=2&a=1#quote"
    ) == "https://example.com/fact?a=1&b=2"


def test_canonicalize_citation_url_requires_absolute_url() -> None:
    with pytest.raises(ValueError):
        canonicalize_citation_url("/relative/fact")

    with pytest.raises(ValueError):
        canonicalize_citation_url("example.com/fact")


def test_empty_citations_are_valid_for_completed_results() -> None:
    result = _result(citations=[])

    assert citation_is_valid(result) is True


def test_duplicate_citations_are_invalid() -> None:
    citation = _citation(evidence_id="e1", url="https://example.com/source")
    result = _result(citations=[citation, citation])

    assert citation_is_valid(result) is False


def test_unknown_citation_is_invalid() -> None:
    citation = _citation(evidence_id="unknown", url="https://example.com/source")
    result = _result(citations=[citation])

    assert citation_is_valid(result) is False


def test_citation_urls_are_sorted_canonical_and_unique() -> None:
    result = _result(
        citations=[
            _citation(
                evidence_id="e1",
                url="HTTPS://Example.COM:443/fact/?b=2&a=1#quote",
            ),
            _citation(
                evidence_id="e2",
                url="https://example.com/fact?a=1&b=2",
            ),
            _citation(evidence_id="e3", url="https://other.example/source/"),
        ]
    )

    assert citation_urls(result) == [
        "https://example.com/fact?a=1&b=2",
        "https://other.example/source",
    ]


def test_normalize_result_uses_enum_values_and_sorted_unique_fields() -> None:
    result = _result(
        citations=[_citation(evidence_id="e1", url="https://example.com/source/")],
        errors=[" Z", "A", "A "],
    )

    assert normalize_result(result) == {
        "status": "completed",
        "verdict": "Supported",
        "route": "single",
        "route_source": None,
        "escalated": False,
        "errors": ["A", "Z"],
        "available_evidence_ids": ["e1", "e2"],
        "citation_urls": ["https://example.com/source"],
        "citation_valid": True,
    }


def test_normalize_result_none_returns_empty_comparison_shape() -> None:
    assert normalize_result(None) == {
        "status": None,
        "verdict": None,
        "route": None,
        "route_source": None,
        "escalated": None,
        "errors": [],
        "available_evidence_ids": [],
        "citation_urls": [],
        "citation_valid": None,
    }


def test_stability_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RepeatSnapshot(
            repeat=0,
            valid=True,
            unexpected="field",
        )

    with pytest.raises(ValidationError):
        StabilityClaimDiagnostic(
            claim_id="claim-0",
            repeats=[],
            differing_fields=[],
            categories=[],
            primary_category=StabilityCategory.VERDICT_DRIFT,
            unexpected="field",
        )

    with pytest.raises(ValidationError):
        StabilityDiagnosticSummary(
            activity_id="activity",
            campaign_id="campaign",
            claim_count=0,
            consistent_claim_count=0,
            category_counts={},
            records=[],
            unexpected="field",
        )


@pytest.mark.parametrize("repeat", [-1, 3])
def test_repeat_snapshot_rejects_repeat_out_of_bounds(repeat: int) -> None:
    with pytest.raises(ValidationError):
        RepeatSnapshot(repeat=repeat, valid=False)


def test_fixture_builds_real_valid_run_artifact() -> None:
    artifact = _artifact(
        _result(
            citations=[_citation(evidence_id="e1", url="https://example.com/source")]
        )
    )

    assert isinstance(artifact.result.citations[0], Citation)
    assert artifact.result.claim_id == artifact.claim_id
    assert artifact.result.usage == artifact.usage
