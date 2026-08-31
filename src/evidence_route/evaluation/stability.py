"""Provider-free contracts and normalization helpers for stability diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from pydantic import Field

from evidence_route.contracts import StrictModel, VerificationResult

if TYPE_CHECKING:
    from evidence_route.artifacts import SQLiteRunStore
    from evidence_route.evaluation.activity import RunArtifact


class StabilityCategory(StrEnum):
    INCOMPLETE_OR_FAILED = "incomplete_or_failed"
    EVIDENCE_OR_CITATION_DRIFT = "evidence_or_citation_drift"
    ROUTE_DRIFT = "route_drift"
    VALIDATION_OR_STATUS_DRIFT = "validation_or_status_drift"
    VERDICT_DRIFT = "verdict_drift"
    PROVIDER_VARIANCE = "provider_variance"


class RepeatSnapshot(StrictModel):
    repeat: int = Field(ge=0, le=2)
    run_id: str | None = None
    artifact_sha256: str | None = None
    valid: bool
    status: str | None = None
    verdict: str | None = None
    route: str | None = None
    route_source: str | None = None
    escalated: bool | None = None
    worker_count: int | None = Field(default=None, ge=0)
    error_codes: list[str] = Field(default_factory=list)
    citation_urls: list[str] = Field(default_factory=list)
    citations_valid: bool | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_coverage: list[str] = Field(default_factory=list)
    selected_evidence_order: list[str] = Field(default_factory=list)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    fresh_latency_ms: int | None = Field(default=None, ge=0)
    exact_cost_micro_cny: int | None = Field(default=None, ge=0)
    transport_attempts: int | None = Field(default=None, ge=0)
    cache_hits: int | None = Field(default=None, ge=0)


class StabilityClaimDiagnostic(StrictModel):
    claim_id: str
    repeats: list[RepeatSnapshot]
    differing_fields: list[str]
    categories: list[StabilityCategory]
    primary_category: StabilityCategory


class StabilityDiagnosticSummary(StrictModel):
    activity_id: str
    campaign_id: str
    claim_count: int = Field(ge=0)
    consistent_claim_count: int = Field(ge=0)
    category_counts: dict[str, Annotated[int, Field(ge=0)]]
    records: list[StabilityClaimDiagnostic]


def build_stability_diagnostics(
    *,
    activity_id: str,
    campaign_id: str,
    repeat_runs: Mapping[str, Mapping[int, RunArtifact | None]],
    run_store: SQLiteRunStore | None,
) -> StabilityDiagnosticSummary:
    del activity_id, campaign_id, repeat_runs, run_store
    raise NotImplementedError("implemented in Task 2")


def canonicalize_citation_url(value: str) -> str:
    """Return a deterministic absolute URL representation for a citation."""
    if not isinstance(value, str):
        raise ValueError("citation URL must be a string")

    parts = urlsplit(value.strip())
    if not parts.scheme or not parts.netloc:
        raise ValueError("citation URL must be absolute")

    try:
        hostname = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise ValueError("citation URL has an invalid authority") from exc
    if not hostname:
        raise ValueError("citation URL must include a hostname")

    hostname = hostname.lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if port is not None and not (
        (parts.scheme.lower() == "http" and port == 80)
        or (parts.scheme.lower() == "https" and port == 443)
    ):
        netloc = f"{netloc}:{port}"
    if parts.username is not None:
        userinfo = unquote(parts.netloc.rsplit("@", 1)[0])
        netloc = f"{userinfo}@{netloc}"

    path = parts.path or "/"
    path = path.rstrip("/") or "/"
    query_pairs = sorted(parse_qsl(parts.query, keep_blank_values=True))
    query = urlencode(query_pairs)
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def citation_urls(result: VerificationResult) -> list[str]:
    """Return sorted, unique canonical URLs from a verification result."""
    return sorted(
        {canonicalize_citation_url(str(citation.source_url)) for citation in result.citations}
    )


def citation_is_valid(result: VerificationResult) -> bool | None:
    """Apply the reporting layer's structural citation validity rule."""
    if result.status not in {"completed", "partial"}:
        return None
    if not result.citations:
        return True

    available = {value.strip() for value in result.available_evidence_ids if value.strip()}
    seen: set[str] = set()
    for citation in result.citations:
        if (
            citation.evidence_id not in available
            or citation.evidence_id in seen
            or not citation.claim_unit_ids
            or any(not claim_unit_id.strip() for claim_unit_id in citation.claim_unit_ids)
            or not citation.question.strip()
            or not citation.answer.strip()
            or not citation.quote.strip()
        ):
            return False
        try:
            canonicalize_citation_url(str(citation.source_url))
        except ValueError:
            return False
        seen.add(citation.evidence_id)
    return True


def _normalize_strings(values: list[str]) -> list[str]:
    return sorted({value.strip() for value in values if value.strip()})


def normalize_result(result: VerificationResult | None) -> dict[str, object]:
    """Return only deterministic result fields used for stability comparison."""
    if result is None:
        return {
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

    return {
        "status": result.status.value,
        "verdict": result.verdict.value if result.verdict is not None else None,
        "route": result.initial_route,
        "route_source": None,
        "escalated": result.escalated,
        "errors": _normalize_strings(result.errors),
        "available_evidence_ids": _normalize_strings(result.available_evidence_ids),
        "citation_urls": citation_urls(result),
        "citation_valid": citation_is_valid(result),
    }
