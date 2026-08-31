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
    build_stability_comparison,
    build_stability_diagnostics,
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


def _variant(artifact: RunArtifact, **updates: object) -> RunArtifact:
    result_updates = updates.pop("result", {})
    result = artifact.result.model_copy(update=result_updates)
    payload = {"result": result, "usage": result.usage}
    payload.update(updates)
    return artifact.model_copy(update=payload)


class _MetadataStore:
    def __init__(self, metadata: dict[str, dict[str, object] | None]) -> None:
        self.metadata = metadata
        self.calls: list[str] = []

    def get_call_metadata(self, call_id: str) -> dict[str, object] | None:
        self.calls.append(call_id)
        return self.metadata.get(call_id)


def _diagnostics(
    repeat_runs: dict[str, dict[int, RunArtifact | None]],
    run_store: object | None = None,
) -> StabilityDiagnosticSummary:
    return build_stability_diagnostics(
        activity_id="activity",
        campaign_id="campaign",
        repeat_runs=repeat_runs,
        run_store=run_store,  # type: ignore[arg-type]
    )


def test_canonicalize_citation_url_removes_fragment_default_port_and_trailing_slash() -> None:
    assert canonicalize_citation_url(
        "HTTPS://Example.COM:443/fact/?b=2&a=1#quote"
    ) == "https://example.com/fact?a=1&b=2"


def test_canonicalize_citation_url_preserves_blank_query_values() -> None:
    assert canonicalize_citation_url("https://example.com/fact?a=") == (
        "https://example.com/fact?a="
    )
    assert canonicalize_citation_url("https://example.com/fact") != (
        "https://example.com/fact?a="
    )


def test_canonicalize_citation_url_sorts_repeated_query_parameters() -> None:
    first = canonicalize_citation_url("https://example.com/fact?b=2&a=2&a=&a=1")
    second = canonicalize_citation_url("https://example.com/fact?a=1&b=2&a=&a=2")

    assert first == "https://example.com/fact?a=&a=1&a=2&b=2"
    assert second == first


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


def test_missing_repeat_is_visible_and_primary_incomplete() -> None:
    artifact = _artifact()

    summary = _diagnostics({"claim": {0: artifact, 1: None, 2: artifact}})

    record = summary.records[0]
    assert [snapshot.repeat for snapshot in record.repeats] == [0, 1, 2]
    assert record.repeats[1].valid is False
    assert record.primary_category is StabilityCategory.INCOMPLETE_OR_FAILED
    assert record.categories == [StabilityCategory.INCOMPLETE_OR_FAILED]


def test_stability_comparison_reports_improved_claims_and_category_deltas() -> None:
    artifact = _artifact()
    baseline = _diagnostics({"claim": {0: artifact, 1: None, 2: artifact}})
    experiment = _diagnostics({"claim": {0: artifact, 1: artifact, 2: artifact}})

    comparison = build_stability_comparison(baseline, experiment)

    assert comparison.improved_claims == ["claim"]
    assert comparison.regressed_claims == []
    assert comparison.unchanged_claims == []
    assert comparison.baseline_consistent_claim_count == 0
    assert comparison.experiment_consistent_claim_count == 1
    assert comparison.baseline_completion_rate == pytest.approx(2 / 3)
    assert comparison.experiment_completion_rate == 1.0
    assert comparison.baseline_diagnostic_sha256 != comparison.experiment_diagnostic_sha256


def test_stability_comparison_rejects_different_claim_sets() -> None:
    artifact = _artifact()
    baseline = _diagnostics({"claim-a": {0: artifact, 1: artifact, 2: artifact}})
    experiment = _diagnostics({"claim-b": {0: artifact, 1: artifact, 2: artifact}})

    with pytest.raises(ValueError, match="claim sets"):
        build_stability_comparison(baseline, experiment)


def test_failed_and_partial_incomplete_repeats_are_visible_and_incomplete() -> None:
    artifact = _artifact()
    failed = _artifact(
        _result(status=ResultStatus.FAILED, errors=["PROBE_RETRIEVAL_FAILED"])
    )
    incomplete_usage = Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False)
    partial_incomplete = _variant(
        artifact,
        result={
            "usage": incomplete_usage,
            "estimated_cost_micro_cny": None,
            "cost_currency": None,
            "price_config_id": None,
        },
        usage=incomplete_usage,
        usage_source="missing",
        actual_cost_micro_cny=None,
        known_actual_cost_micro_cny=0,
        committed_cost_micro_cny=42,
        cost_is_lower_bound=True,
        diagnostic_only=True,
        response_model_ids_raw=["model-id"],
    )

    summary = _diagnostics({"claim": {0: artifact, 1: failed, 2: partial_incomplete}})

    record = summary.records[0]
    assert all(snapshot.valid is False for snapshot in record.repeats[1:])
    assert record.repeats[1].status == "failed"
    assert record.primary_category is StabilityCategory.INCOMPLETE_OR_FAILED


def test_evidence_order_and_citation_change_is_primary_evidence_drift() -> None:
    artifact = _artifact(
        _result(
            citations=[_citation(evidence_id="e1", url="https://example.com/source")],
            available_evidence_ids=["e1", "e2"],
        )
    )
    changed = _variant(
        artifact,
        result={
            "available_evidence_ids": ["e2", "e1"],
            "citations": [
                _citation(evidence_id="e1", url="https://other.example/source")
            ],
        },
    )

    record = _diagnostics({"claim": {0: artifact, 1: changed, 2: artifact}}).records[0]

    assert record.primary_category is StabilityCategory.EVIDENCE_OR_CITATION_DRIFT
    assert StabilityCategory.EVIDENCE_OR_CITATION_DRIFT in record.categories
    assert record.repeats[1].selected_evidence_order == ["e2", "e1"]
    assert record.repeats[1].evidence_ids == ["e1"]


def test_citation_validity_change_is_evidence_primary_with_validation_contributor() -> None:
    citation = _citation(evidence_id="e1", url="https://example.com/source")
    artifact = _artifact(_result(citations=[citation], available_evidence_ids=["e1"]))
    invalid_citation = citation.model_copy(update={"claim_unit_ids": [""]})
    changed = _variant(artifact, result={"citations": [invalid_citation]})

    record = _diagnostics({"claim": {0: artifact, 1: changed, 2: artifact}}).records[0]

    assert record.primary_category is StabilityCategory.EVIDENCE_OR_CITATION_DRIFT
    assert record.categories == [
        StabilityCategory.EVIDENCE_OR_CITATION_DRIFT,
        StabilityCategory.VALIDATION_OR_STATUS_DRIFT,
    ]
    assert record.differing_fields.count("citations_valid") == 1


def test_route_source_escalation_and_worker_count_are_route_drift() -> None:
    artifact = _artifact()
    repeat_one = _variant(
        artifact,
        call_ids=["call-1"],
        result={"escalated": True},
    )
    store = _MetadataStore(
        {
            "call-0": {"node": "router", "transport_attempts": 1, "cache_hits": 0},
            "call-1": {"node": "worker", "transport_attempts": 3, "cache_hits": 2},
        }
    )

    record = _diagnostics(
        {"claim": {0: artifact, 1: repeat_one, 2: artifact}}, store
    ).records[0]

    assert record.primary_category is StabilityCategory.ROUTE_DRIFT
    assert record.repeats[0].route_source == "llm"
    assert record.repeats[1].route_source == "rule_or_fallback"
    assert record.repeats[1].worker_count == 1
    assert record.repeats[1].transport_attempts == 3
    assert record.repeats[1].cache_hits == 2

    fallback = _diagnostics({"claim": {0: artifact, 1: artifact, 2: artifact}}).records[0]
    assert fallback.repeats[0].route_source is None
    assert fallback.repeats[0].worker_count is None
    assert fallback.repeats[0].transport_attempts is None
    assert fallback.repeats[0].cache_hits == artifact.cache_hit_count


def test_same_verdict_with_status_error_and_citation_validity_is_validation_drift() -> None:
    artifact = _artifact()
    status_changed = _artifact(_result(status=ResultStatus.PARTIAL, errors=["validation-error"]))

    record = _diagnostics(
        {"claim": {0: artifact, 1: status_changed, 2: artifact}}
    ).records[0]

    assert record.primary_category is StabilityCategory.VALIDATION_OR_STATUS_DRIFT
    assert record.categories == [StabilityCategory.VALIDATION_OR_STATUS_DRIFT]


def test_verdict_change_after_equal_deterministic_fields_is_verdict_drift() -> None:
    artifact = _artifact()
    changed = _variant(artifact, result={"verdict": Verdict.REFUTED})

    record = _diagnostics({"claim": {0: artifact, 1: changed, 2: artifact}}).records[0]

    assert record.primary_category is StabilityCategory.VERDICT_DRIFT
    assert record.categories == [StabilityCategory.VERDICT_DRIFT]


def test_unexplained_verdict_change_is_provider_variance() -> None:
    artifact = _artifact()
    changed = _variant(
        artifact,
        response_model_ids_raw=["different-model"],
        result={
            "verdict": Verdict.REFUTED,
            "confidence": 0.8,
            "rationale": "provider returned a different explanation",
        },
    )

    record = _diagnostics({"claim": {0: artifact, 1: changed, 2: artifact}}).records[0]

    assert record.primary_category is StabilityCategory.PROVIDER_VARIANCE
    assert record.categories == [StabilityCategory.PROVIDER_VARIANCE]


def test_quantitative_only_difference_has_no_instability_category() -> None:
    artifact = _artifact()
    changed = _variant(
        artifact,
        latency=artifact.latency.model_copy(update={"fresh_end_to_end_ms": 99}),
        cache_hit_count=4,
    )

    record = _diagnostics({"claim": {0: artifact, 1: changed, 2: artifact}}).records[0]

    assert record.categories == []
    assert record.primary_category is None


def test_records_categories_and_category_counts_are_deterministically_sorted() -> None:
    artifact = _artifact()
    evidence_changed = _variant(
        artifact,
        result={"available_evidence_ids": ["e1", "e2"]},
    )
    route_changed = _variant(artifact, result={"initial_route": "multi"})
    store = _MetadataStore(
        {
            "call-0": {"node": "router", "transport_attempts": 1, "cache_hits": 0},
        }
    )

    summary = _diagnostics(
        {
            "z-claim": {0: artifact, 1: route_changed, 2: artifact},
            "a-claim": {0: artifact, 1: evidence_changed, 2: artifact},
            "stable": {0: artifact, 1: artifact, 2: artifact},
        },
        store,
    )

    assert [record.claim_id for record in summary.records] == ["a-claim", "stable", "z-claim"]
    assert list(summary.category_counts) == [
        StabilityCategory.EVIDENCE_OR_CITATION_DRIFT.value,
        StabilityCategory.ROUTE_DRIFT.value,
    ]
    assert summary.claim_count == 3
    assert summary.consistent_claim_count == 3
    assert summary.category_counts == {
        StabilityCategory.EVIDENCE_OR_CITATION_DRIFT.value: 1,
        StabilityCategory.ROUTE_DRIFT.value: 1,
    }


def test_run_store_metadata_failures_use_nullable_fallbacks_without_provider_calls() -> None:
    artifact = _artifact()

    class BrokenStore:
        def get_call_metadata(self, call_id: str) -> dict[str, object] | None:
            if call_id == "call-0":
                raise RuntimeError("metadata unavailable")
            return {"node": object(), "transport_attempts": "bad", "cache_hits": None}

    snapshot = _diagnostics(
        {"claim": {0: artifact, 1: artifact, 2: artifact}}, BrokenStore()
    ).records[0].repeats[0]

    assert snapshot.valid is True
    assert snapshot.route_source is None
    assert snapshot.worker_count is None
    assert snapshot.transport_attempts is None
    assert snapshot.cache_hits is None


def test_none_call_metadata_makes_all_ledger_derived_fields_unknown() -> None:
    artifact = _artifact()
    store = _MetadataStore({"call-0": None})

    snapshot = _diagnostics(
        {"claim": {0: artifact, 1: artifact, 2: artifact}}, store
    ).records[0].repeats[0]

    assert snapshot.route_source is None
    assert snapshot.worker_count is None
    assert snapshot.transport_attempts is None
    assert snapshot.cache_hits is None


def test_malformed_runtime_artifact_is_visible_with_malformed_error_code() -> None:
    snapshot = _diagnostics(
        {"claim": {0: object(), 1: None, 2: _artifact()}}
    ).records[0].repeats[0]

    assert snapshot.valid is False
    assert snapshot.error_codes == ["MALFORMED_ARTIFACT"]
    assert snapshot.run_id is None
    assert snapshot.status is None


def test_run_artifact_field_errors_are_not_silently_marked_invalid() -> None:
    malformed = _artifact().model_copy(update={"result": object()})

    with pytest.raises(AttributeError):
        _diagnostics({"claim": {0: malformed, 1: malformed, 2: malformed}})


def test_constructed_artifact_with_invalid_citation_url_does_not_hide_value_error() -> None:
    artifact = _artifact()
    malformed_citation = Citation.model_construct(
        evidence_id="e1",
        claim_unit_ids=["u0"],
        question="question",
        answer="answer",
        quote="quote",
        stance="supports",
        source_url=object(),
    )
    malformed_result = artifact.result.model_copy(update={"citations": [malformed_citation]})
    payload = artifact.__dict__.copy()
    payload["result"] = malformed_result
    malformed = RunArtifact.model_construct(**payload)

    with pytest.raises(ValueError, match="absolute"):
        _diagnostics({"claim": {0: artifact, 1: malformed, 2: artifact}})


def test_constructed_artifact_with_invalid_provider_fields_does_not_hide_type_error() -> None:
    artifact = _artifact()
    changed_result = artifact.result.model_copy(update={"verdict": Verdict.REFUTED})
    payload = artifact.__dict__.copy()
    payload["result"] = changed_result
    payload["response_model_ids_raw"] = object()
    malformed = RunArtifact.model_construct(**payload)

    with pytest.raises(TypeError):
        _diagnostics({"claim": {0: artifact, 1: malformed, 2: artifact}})
