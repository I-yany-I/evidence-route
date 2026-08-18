import pytest
from pydantic import ValidationError

from evidence_route.contracts import (
    Citation,
    DecompositionDraft,
    Evidence,
    ResultStatus,
    Usage,
    Verdict,
    VerificationResult,
    VerificationTask,
    WorkerResult,
)


def citation() -> Citation:
    return Citation(
        evidence_id="av:dev:0:0:4",
        claim_unit_ids=["u0"],
        question="Was the letter authentic?",
        answer="The letter was fabricated.",
        quote="Sean Connery never sent the letter.",
        stance="refutes",
        source_url="https://example.org/source",
    )


def complete_usage() -> Usage:
    return Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True)


def test_completed_result_requires_verdict() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="dev-0",
            status=ResultStatus.COMPLETED,
            confidence=0.8,
            rationale="Evidence is consistent.",
            citations=[citation()],
            initial_route="single",
            usage=complete_usage(),
        )


def test_failed_result_rejects_verdict() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="dev-0",
            status=ResultStatus.FAILED,
            verdict=Verdict.REFUTED,
            rationale="Provider unavailable.",
            initial_route="single",
            usage=complete_usage(),
        )


def test_running_is_not_a_final_result_status() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="dev-0", status="running", rationale="not final",
            initial_route="single", usage=complete_usage(),
        )


def test_result_requires_explicit_usage() -> None:
    with pytest.raises(ValidationError, match="usage"):
        VerificationResult(
            claim_id="dev-0", status=ResultStatus.FAILED,
            rationale="transport failed", initial_route="single",
        )


def test_incomplete_usage_forbids_exact_cost_fields() -> None:
    with pytest.raises(ValidationError, match="incomplete usage"):
        VerificationResult(
            claim_id="dev-0", status=ResultStatus.FAILED,
            rationale="provider omitted usage", initial_route="single",
            usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False),
            estimated_cost_micro_cny=0, cost_currency="CNY", price_config_id="a" * 64,
        )


def test_only_typed_pre_route_failure_allows_null_initial_route() -> None:
    result = VerificationResult(
        claim_id="dev-0", status=ResultStatus.FAILED,
        rationale="probe retrieval failed", initial_route=None,
        failure_stage="pre_route", usage=complete_usage(),
        errors=["PROBE_RETRIEVAL_FAILED"],
    )
    assert result.initial_route is None
    with pytest.raises(ValidationError, match="initial_route"):
        VerificationResult(
            claim_id="dev-0", status=ResultStatus.FAILED,
            rationale="single failed", initial_route=None,
            failure_stage="single", usage=complete_usage(),
        )


def test_failed_worker_rejects_verdict() -> None:
    with pytest.raises(ValidationError):
        WorkerResult(
            task_id="t0", claim_unit_ids=["u0"], status=ResultStatus.FAILED,
            verdict=Verdict.REFUTED, usage=complete_usage(),
        )


def test_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="av:dev:0:0:4",
            title="Source",
            source_url="https://example.org/source",
            text="Evidence text",
            provider="averitec_frozen",
            snapshot_sha256="a" * 64,
            ranking_score=1.0,
            gold_label="Refuted",
        )


def test_decomposition_rejects_duplicate_task_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        DecompositionDraft(
            tasks=[
                VerificationTask(task_id="t0", claim_unit_ids=["u0"], query="one"),
                VerificationTask(task_id="t0", claim_unit_ids=["u1"], query="two"),
            ]
        )


def test_decomposition_rejects_more_than_three_tasks() -> None:
    with pytest.raises(ValidationError):
        DecompositionDraft(
            tasks=[
                VerificationTask(task_id="t0", claim_unit_ids=["u0"], query="one"),
                VerificationTask(task_id="t1", claim_unit_ids=["u1"], query="two"),
                VerificationTask(task_id="t2", claim_unit_ids=["u2"], query="three"),
                VerificationTask(task_id="t0", claim_unit_ids=["u3"], query="four"),
            ]
        )
