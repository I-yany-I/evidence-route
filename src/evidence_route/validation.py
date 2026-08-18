from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from evidence_route.contracts import ResultStatus, Strategy, VerificationResult


class ValidationAction(StrEnum):
    ACCEPT = "accept"
    ESCALATE = "escalate"
    FAIL = "fail"


@dataclass(frozen=True)
class ValidationDecision:
    action: ValidationAction
    result: VerificationResult
    errors: tuple[str, ...] = ()


class ResultValidator:
    def __init__(self, *, low_confidence: float, minimum_coverage: float) -> None:
        self.low_confidence = low_confidence
        self.minimum_coverage = minimum_coverage

    def validate(
        self,
        *,
        result: VerificationResult,
        claim_unit_ids: Iterable[str],
        evidence_ids: set[str],
        escalation_count: int,
        strategy: Strategy | str,
        draft_origin: str | None = None,
    ) -> ValidationDecision:
        if result.status == ResultStatus.FAILED:
            return ValidationDecision(ValidationAction.FAIL, result, tuple(result.errors))
        units = set(claim_unit_ids)
        allowed = set(result.available_evidence_ids) & set(evidence_ids)
        errors: list[str] = []
        cited_units: set[str] = set()
        for citation in result.citations:
            if citation.evidence_id not in allowed:
                errors.append(f"UNKNOWN_EVIDENCE:{citation.evidence_id}")
            unknown_units = set(citation.claim_unit_ids) - units
            if unknown_units:
                errors.append("UNKNOWN_CLAIM_UNIT")
            cited_units.update(set(citation.claim_unit_ids) & units)
        coverage = len(cited_units) / len(units) if units else 1.0
        if coverage < self.minimum_coverage:
            errors.append("INSUFFICIENT_COVERAGE")
        if result.confidence is None or result.confidence < self.low_confidence:
            errors.append("LOW_CONFIDENCE")
        if not errors:
            return ValidationDecision(ValidationAction.ACCEPT, result)
        eligible = (
            str(strategy) == Strategy.ADAPTIVE.value
            and (draft_origin or result.initial_route) == "single"
            and escalation_count == 0
        )
        if eligible:
            return ValidationDecision(ValidationAction.ESCALATE, result, tuple(errors))
        failed = VerificationResult(
            claim_id=result.claim_id,
            status=ResultStatus.FAILED,
            rationale="; ".join(errors),
            citations=[],
            available_evidence_ids=result.available_evidence_ids,
            initial_route=result.initial_route,
            escalated=result.escalated,
            failure_stage="validation",
            usage=result.usage,
            latency_ms=result.latency_ms,
            errors=[*result.errors, *errors],
        )
        return ValidationDecision(ValidationAction.FAIL, failed, tuple(errors))
