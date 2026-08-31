"""Provider-free contracts and normalization helpers for stability diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any
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
    primary_category: StabilityCategory | None


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
    records: list[StabilityClaimDiagnostic] = []
    category_counts: dict[str, int] = {}
    consistent_claim_count = 0

    for claim_id in sorted(repeat_runs):
        runs = repeat_runs[claim_id]
        snapshots = [
            _build_repeat_snapshot(runs.get(repeat), repeat, run_store)
            for repeat in range(3)
        ]
        categories, differing_fields = _classify_repeats(snapshots, runs)
        primary_category = categories[0] if categories else None
        if _is_consistent(snapshots):
            consistent_claim_count += 1

        for category in categories:
            category_counts[category.value] = category_counts.get(category.value, 0) + 1
        records.append(
            StabilityClaimDiagnostic(
                claim_id=claim_id,
                repeats=snapshots,
                differing_fields=differing_fields,
                categories=categories,
                primary_category=primary_category,
            )
        )

    return StabilityDiagnosticSummary(
        activity_id=activity_id,
        campaign_id=campaign_id,
        claim_count=len(records),
        consistent_claim_count=consistent_claim_count,
        category_counts={key: category_counts[key] for key in sorted(category_counts)},
        records=records,
    )


_EVIDENCE_FIELDS = (
    "evidence_ids",
    "evidence_coverage",
    "selected_evidence_order",
    "citation_urls",
    "citations_valid",
)
_ROUTE_FIELDS = ("route", "route_source", "escalated", "worker_count")
_VALIDATION_FIELDS = ("status", "error_codes", "citations_valid")
_COMPARISON_FIELDS = _EVIDENCE_FIELDS + _ROUTE_FIELDS + ("status", "error_codes", "verdict")


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _normalized_order(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            normalized.append(value.strip())
    return normalized


def _safe_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _metadata_value(metadata: Mapping[str, object], name: str) -> int:
    value = _safe_nonnegative_int(metadata.get(name))
    return value if value is not None else 0


def _strategy_route_source(strategy: object) -> str | None:
    value = str(_enum_value(strategy))
    if value == "adaptive":
        return "rule_or_fallback"
    if value in {"always_single", "always_multi"}:
        return "strategy"
    return None


def _run_store_accounting(
    artifact: Any,
    run_store: Any,
) -> tuple[str | None, int | None, int | None, int | None]:
    if run_store is None:
        return (
            _strategy_route_source(getattr(artifact, "strategy", None)),
            None,
            None,
            _safe_nonnegative_int(getattr(artifact, "cache_hit_count", None)),
        )

    metadata_rows: list[Mapping[str, object]] = []
    for call_id in getattr(artifact, "call_ids", []):
        try:
            metadata = run_store.get_call_metadata(call_id)
        except Exception:
            continue
        if isinstance(metadata, Mapping):
            metadata_rows.append(metadata)

    has_router = any(row.get("node") == "router" for row in metadata_rows)
    worker_count = sum(row.get("node") == "worker" for row in metadata_rows)
    transport_attempts = sum(_metadata_value(row, "transport_attempts") for row in metadata_rows)
    cache_hits = sum(_metadata_value(row, "cache_hits") for row in metadata_rows)
    route_source = "llm" if has_router else _strategy_route_source(artifact.strategy)
    return route_source, worker_count, transport_attempts, cache_hits


def _build_repeat_snapshot(artifact: Any, repeat: int, run_store: Any) -> RepeatSnapshot:
    if artifact is None:
        return RepeatSnapshot(repeat=repeat, valid=False)

    try:
        result = artifact.result
        status = _enum_value(result.status)
        verdict = _enum_value(result.verdict) if result.verdict is not None else None
        usage = artifact.usage
        route_source, worker_count, transport_attempts, cache_hits = _run_store_accounting(
            artifact, run_store
        )
        try:
            urls = citation_urls(result)
        except Exception:
            urls = []
        try:
            citations_valid = citation_is_valid(result)
        except Exception:
            citations_valid = None
        citation_evidence_ids = sorted(
            {
                citation.evidence_id.strip()
                for citation in result.citations
                if isinstance(getattr(citation, "evidence_id", None), str)
                and citation.evidence_id.strip()
            }
        )
        error_codes = _normalize_strings(
            _normalized_order(getattr(result, "errors", []))
            + _normalized_order([getattr(result, "failure_stage", None)])
        )
        valid = bool(
            not getattr(artifact, "diagnostic_only", False)
            and status in {"completed", "partial"}
            and verdict is not None
            and getattr(usage, "complete", False)
        )
        return RepeatSnapshot(
            repeat=repeat,
            run_id=getattr(artifact, "run_id", None),
            artifact_sha256=getattr(artifact, "artifact_sha256", None),
            valid=valid,
            status=status if isinstance(status, str) else None,
            verdict=verdict if isinstance(verdict, str) else None,
            route=getattr(result, "initial_route", None),
            route_source=route_source,
            escalated=getattr(result, "escalated", None),
            worker_count=worker_count,
            error_codes=error_codes,
            citation_urls=urls,
            citations_valid=citations_valid,
            evidence_ids=citation_evidence_ids,
            evidence_coverage=_normalized_order(
                getattr(result, "available_evidence_ids", [])
            ),
            selected_evidence_order=_normalized_order(
                getattr(result, "available_evidence_ids", [])
            ),
            input_tokens=_safe_nonnegative_int(getattr(usage, "input_tokens", None)),
            output_tokens=_safe_nonnegative_int(getattr(usage, "output_tokens", None)),
            total_tokens=_safe_nonnegative_int(getattr(usage, "total_tokens", None)),
            fresh_latency_ms=_safe_nonnegative_int(
                getattr(getattr(artifact, "latency", None), "fresh_end_to_end_ms", None)
            ),
            exact_cost_micro_cny=_safe_nonnegative_int(
                getattr(artifact, "actual_cost_micro_cny", None)
            ),
            transport_attempts=transport_attempts,
            cache_hits=cache_hits,
        )
    except Exception:
        return RepeatSnapshot(repeat=repeat, valid=False)


def _snapshot_values(snapshots: list[RepeatSnapshot], field: str) -> set[object]:
    return {str(getattr(snapshot, field)) for snapshot in snapshots}


def _provider_output_signature(artifact: Any) -> object:
    if artifact is None:
        return None
    try:
        result = artifact.result
        return (
            getattr(result, "rationale", None),
            getattr(result, "confidence", None),
            tuple(getattr(artifact, "response_model_ids_raw", [])),
        )
    except Exception:
        return None


def _classify_repeats(
    snapshots: list[RepeatSnapshot],
    artifacts: Mapping[int, Any],
) -> tuple[list[StabilityCategory], list[str]]:
    if any(
        not snapshot.valid or snapshot.status == "failed" for snapshot in snapshots
    ):
        return [StabilityCategory.INCOMPLETE_OR_FAILED], []

    differing_fields = [
        field for field in _COMPARISON_FIELDS if len(_snapshot_values(snapshots, field)) > 1
    ]
    evidence_drift = any(field in differing_fields for field in _EVIDENCE_FIELDS)
    route_drift = any(field in differing_fields for field in _ROUTE_FIELDS)
    same_verdict = len(_snapshot_values(snapshots, "verdict")) == 1
    validation_drift = same_verdict and any(
        field in differing_fields for field in _VALIDATION_FIELDS
    )
    verdict_drift = len(_snapshot_values(snapshots, "verdict")) > 1
    deterministic_explanation = evidence_drift or route_drift or validation_drift
    provider_variance = False
    if verdict_drift and not deterministic_explanation:
        provider_signatures = {
            _provider_output_signature(artifacts.get(repeat)) for repeat in range(3)
        }
        provider_variance = len(provider_signatures) > 1

    categories: list[StabilityCategory] = []
    if evidence_drift:
        categories.append(StabilityCategory.EVIDENCE_OR_CITATION_DRIFT)
    if route_drift:
        categories.append(StabilityCategory.ROUTE_DRIFT)
    if validation_drift:
        categories.append(StabilityCategory.VALIDATION_OR_STATUS_DRIFT)
    if verdict_drift and not deterministic_explanation and not provider_variance:
        categories.append(StabilityCategory.VERDICT_DRIFT)
    if provider_variance:
        categories.append(StabilityCategory.PROVIDER_VARIANCE)
    return categories, differing_fields


def _is_consistent(snapshots: list[RepeatSnapshot]) -> bool:
    return (
        len(snapshots) == 3
        and all(snapshot.valid and snapshot.verdict is not None for snapshot in snapshots)
        and len(_snapshot_values(snapshots, "verdict")) == 1
    )


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
