from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from evidence_route.contracts import ResultStatus, Strategy, Usage, Verdict, VerificationResult
from evidence_route.evaluation.stability import (
    canonicalize_citation_http_url,
    canonicalize_citation_url,
)
from evidence_route.verification import project_citations


def adjudicate_verification_results(
    candidates: Iterable[VerificationResult],
) -> VerificationResult:
    """Choose a conservative, order-independent result from valid completed candidates."""
    valid = [
        candidate
        for candidate in candidates
        if candidate.status is ResultStatus.COMPLETED
        and candidate.verdict is not None
        and candidate.confidence is not None
        and candidate.usage.complete
    ]
    if not valid:
        raise ValueError("adjudication requires at least one completed candidate")
    claim_ids = {candidate.claim_id for candidate in valid}
    if len(claim_ids) != 1:
        raise ValueError("adjudication candidates must have the same claim_id")

    def key(candidate: VerificationResult) -> tuple[object, ...]:
        evidence_ids = tuple(sorted({citation.evidence_id for citation in candidate.citations}))
        citation_urls = tuple(
            sorted(
                canonicalize_citation_url(str(citation.source_url))
                for citation in candidate.citations
            )
        )
        return evidence_ids, citation_urls, candidate.rationale.strip(), -candidate.confidence

    selected = min(valid, key=key)
    selected_citations = project_citations(
        selected.citations,
        selected.available_evidence_ids,
        max_citations=max(1, len(selected.citations)),
    )
    verdicts = {candidate.verdict for candidate in valid}
    if len(verdicts) == 1:
        verdict = selected.verdict
    elif verdicts == {Verdict.SUPPORTED, Verdict.REFUTED}:
        verdict = Verdict.CONFLICTING
    else:
        verdict = Verdict.NOT_ENOUGH_EVIDENCE
    usage = Usage(
        input_tokens=sum(candidate.usage.input_tokens for candidate in valid),
        output_tokens=sum(candidate.usage.output_tokens for candidate in valid),
        total_tokens=sum(candidate.usage.total_tokens for candidate in valid),
        complete=True,
    )
    errors = list(selected.errors)
    if len(verdicts) > 1 and "ADJUDICATED_DISAGREEMENT" not in errors:
        errors.append("ADJUDICATED_DISAGREEMENT")
    return selected.model_copy(
        update={
            "verdict": verdict,
            "confidence": min(candidate.confidence for candidate in valid),
            "citations": selected_citations,
            "available_evidence_ids": sorted(
                {
                    evidence_id
                    for candidate in valid
                    for evidence_id in candidate.available_evidence_ids
                }
            ),
            "usage": usage,
            "errors": errors,
            "estimated_cost_micro_cny": None,
            "cost_currency": None,
            "price_config_id": None,
        }
    )


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
            update={"source_url": canonicalize_citation_http_url(str(citation.source_url))}
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
    def __init__(
        self, *, low_confidence: float, minimum_coverage: float, normalize_output: bool = False
    ) -> None:
        self.low_confidence = low_confidence
        self.minimum_coverage = minimum_coverage
        self.normalize_output = normalize_output

    def validate(
        self,
        *,
        result: VerificationResult,
        claim_unit_ids: Iterable[str],
        evidence_ids: set[str],
        escalation_count: int,
        strategy: Strategy | str,
        draft_origin: str | None = None,
        fallback_used: bool = False,
    ) -> ValidationDecision:
        normalized = normalize_verification_result(result) if self.normalize_output else result
        if normalized.status == ResultStatus.FAILED:
            return ValidationDecision(
                ValidationAction.FAIL, normalized, tuple(normalized.errors)
            )
        units = set(claim_unit_ids)
        allowed = set(normalized.available_evidence_ids) & set(evidence_ids)
        errors: list[str] = []
        if normalized.status is ResultStatus.PARTIAL:
            errors.extend(normalized.errors or ["INCOMPLETE_COVERAGE"])
        cited_units: set[str] = set()
        seen_citations: set[tuple[str, frozenset[str]]] = set()
        for citation in normalized.citations:
            if citation.evidence_id not in allowed:
                errors.append(f"UNKNOWN_EVIDENCE:{citation.evidence_id}")
            citation_key = (
                canonicalize_citation_url(str(citation.source_url)),
                frozenset(citation.claim_unit_ids),
            )
            if citation_key in seen_citations:
                errors.append("DUPLICATE_CITATION")
            unknown_units = set(citation.claim_unit_ids) - units
            if unknown_units:
                errors.append("UNKNOWN_CLAIM_UNIT")
            cited_units.update(set(citation.claim_unit_ids) & units)
            seen_citations.add(citation_key)
        coverage = len(cited_units) / len(units) if units else 1.0
        requires_full_coverage = normalized.verdict is not Verdict.NOT_ENOUGH_EVIDENCE
        if requires_full_coverage and coverage < self.minimum_coverage:
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
            and not fallback_used
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
        final_result = normalize_verification_result(failed) if self.normalize_output else failed
        return ValidationDecision(ValidationAction.FAIL, final_result, tuple(errors))
