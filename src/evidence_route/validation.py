from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from evidence_route.contracts import ResultStatus, Strategy, Verdict, VerificationResult
from evidence_route.evaluation.stability import canonicalize_citation_url


class ValidationAction(StrEnum):
    ACCEPT = "accept"
    ESCALATE = "escalate"
    FAIL = "fail"


@dataclass(frozen=True)
class ValidationDecision:
    action: ValidationAction
    result: VerificationResult
    errors: tuple[str, ...] = ()


def _normalize_strings(values: Iterable[str]) -> list[str]:
    return sorted({value.strip() for value in values if value.strip()})


def normalize_verification_result(result: VerificationResult) -> VerificationResult:
    """Return a fresh result with deterministic structured fields and status invariants."""
    payload = {
        field_name: getattr(result, field_name) for field_name in VerificationResult.model_fields
    }
    payload["rationale"] = str(result.rationale).strip()
    payload["errors"] = _normalize_strings(str(error) for error in result.errors)
    payload["available_evidence_ids"] = _normalize_strings(result.available_evidence_ids)
    payload["citations"] = [
        citation.model_copy(
            update={"source_url": canonicalize_citation_url(str(citation.source_url))}
        )
        for citation in result.citations
    ]

    status = ResultStatus(result.status)
    verdict = result.verdict
    if verdict is not None:
        verdict = Verdict(verdict)
    payload["status"] = status
    payload["verdict"] = verdict

    if status is ResultStatus.FAILED:
        payload["verdict"] = None
        payload["confidence"] = None
    elif status is ResultStatus.COMPLETED:
        if verdict is None or result.confidence is None:
            payload["status"] = ResultStatus.FAILED
            payload["verdict"] = None
            payload["confidence"] = None
            payload["errors"] = _normalize_strings(
                [*payload["errors"], "INVALID_COMPLETED_RESULT"]
            )
        elif not result.usage.complete:
            payload["status"] = ResultStatus.PARTIAL
            payload["errors"] = _normalize_strings([*payload["errors"], "INCOMPLETE_USAGE"])
    elif verdict is None or result.confidence is None:
        payload["status"] = ResultStatus.FAILED
        payload["verdict"] = None
        payload["confidence"] = None
        payload["errors"] = _normalize_strings([*payload["errors"], "INVALID_PARTIAL_RESULT"])
    elif not payload["errors"]:
        payload["errors"] = ["INCOMPLETE_COVERAGE"]

    return VerificationResult.model_validate(payload)


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
        normalized = normalize_verification_result(result)
        if normalized.status == ResultStatus.FAILED:
            return ValidationDecision(
                ValidationAction.FAIL, normalized, tuple(normalized.errors)
            )
        units = set(claim_unit_ids)
        allowed = set(normalized.available_evidence_ids) & set(evidence_ids)
        errors: list[str] = []
        cited_units: set[str] = set()
        seen_evidence_ids: set[str] = set()
        for citation in normalized.citations:
            if citation.evidence_id not in allowed:
                errors.append(f"UNKNOWN_EVIDENCE:{citation.evidence_id}")
            if citation.evidence_id in seen_evidence_ids:
                errors.append("DUPLICATE_CITATION")
            unknown_units = set(citation.claim_unit_ids) - units
            if unknown_units:
                errors.append("UNKNOWN_CLAIM_UNIT")
            cited_units.update(set(citation.claim_unit_ids) & units)
            seen_evidence_ids.add(citation.evidence_id)
        coverage = len(cited_units) / len(units) if units else 1.0
        if coverage < self.minimum_coverage:
            errors.append("INSUFFICIENT_COVERAGE")
        if normalized.confidence is None or normalized.confidence < self.low_confidence:
            errors.append("LOW_CONFIDENCE")
        if not errors:
            return ValidationDecision(ValidationAction.ACCEPT, normalized)
        eligible = (
            (strategy.value if isinstance(strategy, Strategy) else str(strategy))
            == Strategy.ADAPTIVE.value
            and (draft_origin or normalized.initial_route) == "single"
            and escalation_count == 0
        )
        if eligible:
            return ValidationDecision(ValidationAction.ESCALATE, normalized, tuple(errors))
        failed = VerificationResult(
            claim_id=normalized.claim_id,
            status=ResultStatus.FAILED,
            rationale="; ".join(errors),
            citations=[],
            available_evidence_ids=normalized.available_evidence_ids,
            initial_route=normalized.initial_route,
            escalated=normalized.escalated,
            failure_stage="validation",
            usage=normalized.usage,
            latency_ms=normalized.latency_ms,
            errors=[*normalized.errors, *errors],
        )
        return ValidationDecision(
            ValidationAction.FAIL, normalize_verification_result(failed), tuple(errors)
        )
