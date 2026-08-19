"""Adapters from persisted execution accounting to evaluation artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel, ValidationError

from evidence_route.budget import PriceConfig
from evidence_route.contracts import ResultStatus, Strategy, Usage, VerificationResult
from evidence_route.evaluation.activity import LatencyBreakdown, RunArtifact, artifact_fingerprint

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_SUMMARY_FIELDS = (
    "call_ids",
    "usage",
    "actual_cost_micro_cny",
    "known_actual_cost_micro_cny",
    "committed_cost_micro_cny",
    "cost_is_lower_bound",
    "cache_hit_count",
    "requested_aliases",
    "response_model_ids_raw",
    "billing_uncertain",
)


def load_price_config(path: Path) -> PriceConfig:
    """Load the strict CNY pricing file used for deterministic evaluation billing."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"strict CNY price configuration cannot be read: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("strict CNY price configuration root must be a mapping")
    try:
        pricing = PriceConfig.model_validate(raw, strict=True)
    except ValidationError as exc:
        raise ValueError(f"strict CNY price configuration is invalid: {exc}") from exc
    if pricing.currency != "CNY":
        raise ValueError("strict price configuration currency must be CNY")
    if not pricing.strict_evaluation:
        raise ValueError("strict evaluation price configuration is required")
    if (
        pricing.input_per_million is None
        or pricing.output_per_million is None
        or not pricing.price_source.strip()
    ):
        raise ValueError("strict evaluation requires input/output rates and price_source")
    return pricing


def build_run_artifact(
    activity_id: str,
    campaign_id: str,
    phase: Literal["calibration", "dev", "stability"],
    run_id: str,
    claim_id: str,
    strategy: Strategy,
    repeat: int,
    result: VerificationResult,
    summary: Mapping[str, object],
    requested_alias: str,
    price_config_id: str,
    latency: LatencyBreakdown,
) -> RunArtifact:
    """Bind a graph result to the accounting authoritatively persisted by ``SQLiteRunStore``."""
    values = _summary_mapping(summary)
    _require_text("activity_id", activity_id)
    _require_text("campaign_id", campaign_id)
    _require_text("claim_id", claim_id)
    _require_text("requested_alias", requested_alias)
    if not _SHA256_RE.fullmatch(run_id):
        raise ValueError("run_id must be a lowercase SHA-256")
    if not _SHA256_RE.fullmatch(price_config_id):
        raise ValueError("price_config_id must be a lowercase SHA-256")
    if result.claim_id != claim_id:
        raise ValueError("result claim_id must match artifact claim_id")

    for field in _REQUIRED_SUMMARY_FIELDS:
        if field not in values:
            raise ValueError(f"summary is missing required field: {field}")

    call_ids = _string_list("summary.call_ids", values["call_ids"])
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("summary.call_ids must be unique")
    usage = _usage(values["usage"])
    actual_cost = _optional_nonnegative_int(
        "summary.actual_cost_micro_cny", values["actual_cost_micro_cny"]
    )
    known_cost = _nonnegative_int(
        "summary.known_actual_cost_micro_cny", values["known_actual_cost_micro_cny"]
    )
    committed_cost = _nonnegative_int(
        "summary.committed_cost_micro_cny", values["committed_cost_micro_cny"]
    )
    lower_bound = _bool("summary.cost_is_lower_bound", values["cost_is_lower_bound"])
    cache_hits = _nonnegative_int("summary.cache_hit_count", values["cache_hit_count"])
    aliases = _string_list("summary.requested_aliases", values["requested_aliases"])
    model_ids = _string_list("summary.response_model_ids_raw", values["response_model_ids_raw"])
    billing_uncertain = _bool("summary.billing_uncertain", values["billing_uncertain"])

    if len(model_ids) != len(set(model_ids)):
        raise ValueError("summary.response_model_ids_raw must be unique")
    if any(alias != requested_alias for alias in aliases):
        raise ValueError("summary requested_aliases disagree with requested_alias")
    if call_ids and aliases != [requested_alias]:
        raise ValueError("summary requested_aliases must contain exactly one matching alias")
    if known_cost > committed_cost:
        raise ValueError("summary known actual cost cannot exceed committed cost")
    if actual_cost is None:
        if not lower_bound:
            raise ValueError("summary unknown actual cost must be marked lower bound")
    elif actual_cost != known_cost or lower_bound:
        raise ValueError("summary exact actual cost is inconsistent with known cost/lower bound")

    fresh_count = len(call_ids)
    if "fresh_call_count" in values:
        supplied_fresh_count = _nonnegative_int(
            "summary.fresh_call_count", values["fresh_call_count"]
        )
        if supplied_fresh_count != fresh_count:
            raise ValueError("summary fresh_call_count must equal call_ids length")

    usage_sources = _usage_sources(values.get("usage_sources"))
    if usage.complete and any(source != "provider" for source in usage_sources):
        raise ValueError("complete summary usage cannot have missing usage sources")
    if not usage.complete and usage_sources and "missing" not in usage_sources:
        raise ValueError("incomplete summary usage requires a missing usage source")

    if not call_ids:
        if (
            not usage.complete
            or usage.input_tokens != 0
            or usage.output_tokens != 0
            or usage.total_tokens != 0
            or actual_cost != 0
            or known_cost != 0
            or committed_cost != 0
            or lower_bound
            or aliases
            or model_ids
            or billing_uncertain
            or usage_sources
        ):
            raise ValueError("zero-call summary must have zero complete accounting")
        usage_source: Literal["provider", "missing", "not_applicable"] = "not_applicable"
    else:
        usage_source = "provider" if usage.complete else "missing"

    is_pre_route_failure = (
        result.status is ResultStatus.FAILED
        and result.failure_stage == "pre_route"
        and "PROBE_RETRIEVAL_FAILED" in result.errors
    )
    if is_pre_route_failure and call_ids:
        raise ValueError("pre-route failures cannot have persisted calls")
    if not call_ids and not is_pre_route_failure:
        raise ValueError("zero-call result requires typed pre-route failure")
    if result.status in {ResultStatus.COMPLETED, ResultStatus.PARTIAL} and not call_ids:
        raise ValueError("scored result requires at least one persisted call")

    diagnostic_only = billing_uncertain or not usage.complete or actual_cost is None
    updated_result = _result_with_persisted_accounting(
        result=result,
        usage=usage,
        actual_cost=actual_cost,
        price_config_id=price_config_id,
    )
    payload: dict[str, object] = {
        "schema_version": "1",
        "activity_id": activity_id,
        "campaign_id": campaign_id,
        "phase": phase,
        "run_id": run_id,
        "claim_id": claim_id,
        "strategy": strategy.value,
        "repeat": repeat,
        "result": updated_result.model_dump(mode="json"),
        "call_ids": call_ids,
        "usage": usage.model_dump(mode="json"),
        "usage_source": usage_source,
        "actual_cost_micro_cny": actual_cost,
        "known_actual_cost_micro_cny": known_cost,
        "committed_cost_micro_cny": committed_cost,
        "cost_is_lower_bound": lower_bound,
        "fresh_call_count": fresh_count,
        "cache_hit_count": cache_hits,
        "requested_alias": requested_alias,
        "response_model_ids_raw": model_ids,
        "identity_verified": False,
        "billing_uncertain": billing_uncertain,
        "diagnostic_only": diagnostic_only,
        "latency": latency.model_dump(mode="json"),
        "artifact_sha256": "0" * 64,
    }
    payload["artifact_sha256"] = artifact_fingerprint(payload)
    return RunArtifact.model_validate(payload)


def _summary_mapping(summary: Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(summary, Mapping):
        return summary
    if isinstance(summary, BaseModel):
        dumped = summary.model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dumped
    raise ValueError("summary must be a mapping or Pydantic summary model")


def _usage(value: object) -> Usage:
    try:
        return value if isinstance(value, Usage) else Usage.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"summary usage is invalid: {exc}") from exc


def _usage_sources(value: object | None) -> list[Literal["provider", "missing"]]:
    if value is None:
        return []
    sources = _string_list("summary.usage_sources", value)
    if any(source not in {"provider", "missing"} for source in sources):
        raise ValueError("summary.usage_sources must contain provider or missing")
    return cast(list[Literal["provider", "missing"]], sources)


def _string_list(name: str, value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list of strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    return list(value)


def _bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative_int(name: str, value: object) -> int | None:
    return None if value is None else _nonnegative_int(name, value)


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _result_with_persisted_accounting(
    *,
    result: VerificationResult,
    usage: Usage,
    actual_cost: int | None,
    price_config_id: str,
) -> VerificationResult:
    updates: dict[str, object] = {
        "usage": usage,
        "estimated_cost_micro_cny": None,
        "cost_currency": None,
        "price_config_id": None,
    }
    if usage.complete and actual_cost is not None:
        updates.update(
            estimated_cost_micro_cny=actual_cost,
            cost_currency="CNY",
            price_config_id=price_config_id,
        )
    copied = result.model_copy(update=updates)
    try:
        return VerificationResult.model_validate(copied.model_dump(mode="python"))
    except ValidationError as exc:
        raise ValueError(f"persisted accounting is incompatible with result: {exc}") from exc
