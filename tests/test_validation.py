import pytest

from evidence_route.contracts import Citation, ResultStatus, Usage, Verdict, VerificationResult
from evidence_route.validation import (
    ResultValidator,
    ValidationAction,
    adjudicate_verification_results,
    normalize_verification_result,
)


def valid_result() -> VerificationResult:
    return VerificationResult(
        claim_id="dev-0",
        status=ResultStatus.COMPLETED,
        verdict=Verdict.SUPPORTED,
        confidence=0.91,
        rationale="Evidence supports the claim.",
        citations=[],
        available_evidence_ids=[],
        initial_route="single",
        usage=Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True),
    )


def test_validator_never_scans_rationale_for_label() -> None:
    result = VerificationResult(
        claim_id="legacy-third-case",
        status=ResultStatus.COMPLETED,
        verdict=Verdict.REFUTED,
        confidence=0.91,
        rationale="正文讨论了证据不足，但总体判定由结构化字段给出。",
        citations=[],
        initial_route="single",
        usage=Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True),
    )
    decision = ResultValidator(low_confidence=0.65, minimum_coverage=0.0).validate(
        result=result,
        claim_unit_ids=["u0"],
        evidence_ids=set(),
        escalation_count=1,
        strategy="adaptive",
    )
    assert decision.result.verdict is Verdict.REFUTED
    assert decision.action is ValidationAction.ACCEPT


def test_unknown_citation_escalates_once() -> None:
    citation = Citation(
        evidence_id="different-id",
        claim_unit_ids=["u0"],
        question="q",
        answer="a",
        quote="q",
        stance="supports",
        source_url="https://example.org/source",
    )
    result = valid_result().model_copy(update={"citations": [citation]})
    decision = ResultValidator(low_confidence=0.65, minimum_coverage=1.0).validate(
        result=result,
        claim_unit_ids=["u0"],
        evidence_ids={"different-id"},
        escalation_count=0,
        strategy="adaptive",
    )
    assert decision.action is ValidationAction.ESCALATE


def test_second_invalid_result_becomes_failed() -> None:
    result = valid_result()
    decision = ResultValidator(low_confidence=0.65, minimum_coverage=1.0).validate(
        result=result,
        claim_unit_ids=["u0"],
        evidence_ids=set(),
        escalation_count=1,
        strategy="adaptive",
    )
    assert decision.action is ValidationAction.FAIL
    assert decision.result.status is ResultStatus.FAILED
    assert decision.result.verdict is None


def test_not_enough_evidence_does_not_require_every_claim_unit_to_be_cited() -> None:
    citation = Citation(
        evidence_id="e1",
        claim_unit_ids=["u0"],
        question="What does the source establish?",
        answer="The source addresses only the first part.",
        quote="The source addresses only the first part.",
        stance="insufficient",
        source_url="https://example.org/source",
    )
    result = valid_result().model_copy(
        update={
            "verdict": Verdict.NOT_ENOUGH_EVIDENCE,
            "citations": [citation],
            "available_evidence_ids": ["e1"],
        }
    )

    decision = ResultValidator(low_confidence=0.65, minimum_coverage=1.0).validate(
        result=result,
        claim_unit_ids=["u0", "u1"],
        evidence_ids={"e1"},
        escalation_count=1,
        strategy="adaptive",
    )

    assert decision.action is ValidationAction.ACCEPT
    assert decision.result.verdict is Verdict.NOT_ENOUGH_EVIDENCE


def test_supported_result_still_requires_every_claim_unit_to_be_cited() -> None:
    citation = Citation(
        evidence_id="e1",
        claim_unit_ids=["u0"],
        question="What does the source establish?",
        answer="The source establishes the first part.",
        quote="The source establishes the first part.",
        stance="supports",
        source_url="https://example.org/source",
    )
    result = valid_result().model_copy(
        update={
            "verdict": Verdict.SUPPORTED,
            "citations": [citation],
            "available_evidence_ids": ["e1"],
        }
    )

    decision = ResultValidator(low_confidence=0.65, minimum_coverage=1.0).validate(
        result=result,
        claim_unit_ids=["u0", "u1"],
        evidence_ids={"e1"},
        escalation_count=1,
        strategy="adaptive",
    )

    assert decision.action is ValidationAction.FAIL
    assert "INSUFFICIENT_COVERAGE" in decision.errors


def test_partial_result_is_invalid_for_multi_recovery() -> None:
    result = valid_result().model_copy(update={"status": ResultStatus.PARTIAL})

    decision = ResultValidator(
        low_confidence=0.65, minimum_coverage=0.0, normalize_output=True
    ).validate(
        result=result,
        claim_unit_ids=["u0"],
        evidence_ids=set(),
        escalation_count=0,
        strategy="adaptive",
        draft_origin="multi",
    )

    assert decision.action is ValidationAction.FAIL
    assert decision.result.status is ResultStatus.FAILED
    assert "INCOMPLETE_COVERAGE" in decision.errors


def test_normalize_verification_result_cleans_structured_fields() -> None:
    result = valid_result().model_copy(
        update={
            "rationale": "  Evidence supports the claim.  ",
            "errors": [" LOW_CONFIDENCE ", "", "LOW_CONFIDENCE"],
            "available_evidence_ids": [" e2 ", "e1", "e2 ", ""],
            "verdict": "Supported",
        }
    )

    normalized = normalize_verification_result(result)

    assert normalized.rationale == "Evidence supports the claim."
    assert normalized.errors == ["LOW_CONFIDENCE"]
    assert normalized.available_evidence_ids == ["e1", "e2"]
    assert normalized.verdict is Verdict.SUPPORTED


def result_with(verdict: Verdict, evidence_id: str) -> VerificationResult:
    citation = Citation(
        evidence_id=evidence_id,
        claim_unit_ids=["u0"],
        question="q",
        answer="a",
        quote="quote",
        stance="supports" if verdict is Verdict.SUPPORTED else "insufficient",
        source_url=f"https://example.org/{evidence_id}",
    )
    return valid_result().model_copy(
        update={
            "verdict": verdict,
            "citations": [citation],
            "available_evidence_ids": [evidence_id],
            "rationale": f"candidate {evidence_id}",
        }
    )


def test_adjudication_is_order_independent_and_conservative_for_insufficient_evidence() -> None:
    supported = result_with(Verdict.SUPPORTED, "e2")
    insufficient = result_with(Verdict.NOT_ENOUGH_EVIDENCE, "e1")

    first = adjudicate_verification_results([supported, insufficient])
    second = adjudicate_verification_results([insufficient, supported])

    assert first == second
    assert first.verdict is Verdict.NOT_ENOUGH_EVIDENCE
    assert [citation.evidence_id for citation in first.citations] == ["e1"]
    assert first.errors == ["ADJUDICATED_DISAGREEMENT"]


def test_adjudication_maps_direct_support_refute_disagreement_to_conflicting() -> None:
    result = adjudicate_verification_results(
        [result_with(Verdict.SUPPORTED, "e2"), result_with(Verdict.REFUTED, "e1")]
    )

    assert result.verdict is Verdict.CONFLICTING


def test_adjudication_ignores_failed_candidates_but_requires_one_completed_result() -> None:
    failed = VerificationResult(
        claim_id="dev-0",
        status=ResultStatus.FAILED,
        rationale="failed",
        initial_route="single",
        failure_stage="validation",
        usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=True),
        errors=["FAILED"],
    )
    completed = result_with(Verdict.SUPPORTED, "e1")

    assert adjudicate_verification_results([failed, completed]).verdict is Verdict.SUPPORTED
    with pytest.raises(ValueError, match="completed candidate"):
        adjudicate_verification_results([failed])


def test_adjudication_preserves_distinct_claim_units_from_one_evidence_source() -> None:
    first = result_with(Verdict.SUPPORTED, "e1")
    second_citation = first.citations[0].model_copy(
        update={"claim_unit_ids": ["u1"], "quote": "second quote"}
    )
    candidate = first.model_copy(
        update={"citations": [first.citations[0], second_citation]}
    )

    adjudicated = adjudicate_verification_results([candidate])

    assert len(adjudicated.citations) == 2
    assert [citation.claim_unit_ids for citation in adjudicated.citations] == [["u0"], ["u1"]]


def test_validator_does_not_mark_distinct_claim_units_from_one_evidence_as_duplicate() -> None:
    first = result_with(Verdict.SUPPORTED, "e1")
    second = first.citations[0].model_copy(update={"claim_unit_ids": ["u1"]})
    result = first.model_copy(
        update={"citations": [first.citations[0], second]}
    )

    decision = ResultValidator(low_confidence=0.65, minimum_coverage=1.0).validate(
        result=result,
        claim_unit_ids=["u0", "u1"],
        evidence_ids={"e1"},
        escalation_count=0,
        strategy="adaptive",
    )

    assert decision.action is ValidationAction.ACCEPT

def test_normalize_verification_result_preserves_distinct_conflicting_verdict() -> None:
    result = valid_result().model_copy(
        update={"verdict": "Conflicting Evidence/Cherrypicking"}
    )

    assert normalize_verification_result(result).verdict is Verdict.CONFLICTING


def test_partial_result_gets_explicit_incomplete_coverage_error() -> None:
    result = valid_result().model_copy(update={"status": ResultStatus.PARTIAL, "errors": []})

    normalized = normalize_verification_result(result)

    assert normalized.status is ResultStatus.PARTIAL
    assert normalized.verdict is Verdict.SUPPORTED
    assert normalized.errors == ["INCOMPLETE_COVERAGE"]


def test_failed_result_cannot_retain_verdict_after_normalization() -> None:
    result = valid_result().model_copy(update={"status": ResultStatus.FAILED})

    normalized = normalize_verification_result(result)

    assert normalized.status is ResultStatus.FAILED
    assert normalized.verdict is None
    assert normalized.confidence is None


def test_adaptive_single_validation_error_still_escalates_with_typed_errors() -> None:
    citation = Citation(
        evidence_id="unknown",
        claim_unit_ids=["u9"],
        question="q",
        answer="a",
        quote="q",
        stance="supports",
        source_url="https://example.org/source",
    )
    result = valid_result().model_copy(
        update={"citations": [citation], "available_evidence_ids": ["unknown"]}
    )

    decision = ResultValidator(
        low_confidence=0.65, minimum_coverage=1.0, normalize_output=True
    ).validate(
        result=result,
        claim_unit_ids=["u0"],
        evidence_ids={"known"},
        escalation_count=0,
        strategy="adaptive",
    )

    assert decision.action is ValidationAction.ESCALATE
    assert decision.errors == (
        "UNKNOWN_EVIDENCE:unknown",
        "UNKNOWN_CLAIM_UNIT",
        "INSUFFICIENT_COVERAGE",
    )


def test_final_validation_failure_is_normalized_without_promoting_status() -> None:
    result = valid_result().model_copy(
        update={
            "rationale": "  draft  ",
            "available_evidence_ids": [" e2 ", "e1", "e2 "],
            "confidence": 0.1,
        }
    )

    decision = ResultValidator(
        low_confidence=0.65, minimum_coverage=1.0, normalize_output=True
    ).validate(
        result=result,
        claim_unit_ids=["u0"],
        evidence_ids=set(),
        escalation_count=1,
        strategy="adaptive",
    )

    assert decision.action is ValidationAction.FAIL
    assert decision.result.status is ResultStatus.FAILED
    assert decision.result.verdict is None
    assert decision.result.rationale == "INSUFFICIENT_COVERAGE; LOW_CONFIDENCE"
    assert decision.result.available_evidence_ids == ["e1", "e2"]


def test_validator_default_preserves_legacy_result_object_and_fields() -> None:
    result = valid_result().model_copy(
        update={
            "rationale": "  draft  ",
            "errors": [" LOW_CONFIDENCE ", "LOW_CONFIDENCE"],
            "available_evidence_ids": [" e2 ", "e1", "e2 "],
        }
    )

    decision = ResultValidator(low_confidence=0.65, minimum_coverage=0.0).validate(
        result=result,
        claim_unit_ids=["u0"],
        evidence_ids=set(),
        escalation_count=1,
        strategy="adaptive",
    )

    assert decision.action is ValidationAction.ACCEPT
    assert decision.result is result
    assert decision.result.rationale == "  draft  "
    assert decision.result.errors == [" LOW_CONFIDENCE ", "LOW_CONFIDENCE"]


def test_validator_normalizes_only_when_opted_in() -> None:
    result = valid_result().model_copy(update={"rationale": "  draft  "})

    decision = ResultValidator(
        low_confidence=0.65, minimum_coverage=0.0, normalize_output=True
    ).validate(
        result=result,
        claim_unit_ids=["u0"],
        evidence_ids=set(),
        escalation_count=1,
        strategy="adaptive",
    )

    assert decision.action is ValidationAction.ACCEPT
    assert decision.result is not result
    assert decision.result.rationale == "draft"


def test_recovery_result_cannot_escalate_again() -> None:
    result = valid_result().model_copy(update={"initial_route": "multi"})

    decision = ResultValidator(
        low_confidence=0.65, minimum_coverage=1.0, normalize_output=True
    ).validate(
        result=result,
        claim_unit_ids=["u0"],
        evidence_ids=set(),
        escalation_count=0,
        strategy="adaptive",
        draft_origin="single",
        fallback_used=True,
    )

    assert decision.action is ValidationAction.FAIL
