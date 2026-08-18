from evidence_route.contracts import Citation, ResultStatus, Usage, Verdict, VerificationResult
from evidence_route.validation import ResultValidator, ValidationAction


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
