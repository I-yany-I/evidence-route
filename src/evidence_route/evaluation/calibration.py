"""Offline calibration by replaying immutable router, single and multi artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from evidence_route.config import RoutingSettings, stable_hash
from evidence_route.contracts import (
    ClaimFeatures,
    ResultStatus,
    Strategy,
    StrictModel,
    Usage,
    Verdict,
    VerificationResult,
)
from evidence_route.evaluation.metrics import score_full_manifest
from evidence_route.validation import ResultValidator, ValidationAction

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_SHA_PATTERN = r"^[0-9a-f]{40}$"


class CalibrationItemStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETE = "complete"
    STOPPED = "stopped"


class CalibrationWorkItem(StrictModel):
    order: int = Field(ge=0, le=31)
    claim_id: str = Field(min_length=1)
    case_id: str = Field(pattern=_SHA256_PATTERN)
    router_run_id: str = Field(pattern=_SHA256_PATTERN)
    single_run_id: str = Field(pattern=_SHA256_PATTERN)
    multi_run_id: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_distinct_ids(self) -> CalibrationWorkItem:
        run_ids = {self.router_run_id, self.single_run_id, self.multi_run_id}
        if len(run_ids) != 3:
            raise ValueError("calibration run IDs must be distinct")
        return self


class CalibrationPlan(StrictModel):
    schema_version: Literal["1"]
    activity_id: str = Field(min_length=1)
    manifest_freeze_git_sha: str = Field(pattern=_GIT_SHA_PATTERN)
    runtime_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_preparation_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    pricing_sha256: str = Field(pattern=_SHA256_PATTERN)
    endpoint_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    requirements_lock_sha256: str = Field(pattern=_SHA256_PATTERN)
    requested_alias: str = Field(min_length=1)
    seed: int
    cap_micro_cny: int = Field(gt=0)
    items: list[CalibrationWorkItem] = Field(min_length=32, max_length=32)
    plan_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_ordered_items(self) -> CalibrationPlan:
        if [item.order for item in self.items] != list(range(32)):
            raise ValueError("calibration plan items must have immutable order 0..31")
        claim_ids = [item.claim_id for item in self.items]
        case_ids = [item.case_id for item in self.items]
        if len(claim_ids) != len(set(claim_ids)) or len(case_ids) != len(set(case_ids)):
            raise ValueError("calibration plan claim and case IDs must be unique")
        return self


class CalibrationItemState(StrictModel):
    case_id: str = Field(pattern=_SHA256_PATTERN)
    status: CalibrationItemStatus = CalibrationItemStatus.PENDING
    artifact_relpath: str = Field(min_length=1)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_closed_artifact(self) -> CalibrationItemState:
        if self.status is CalibrationItemStatus.COMPLETE and self.artifact_sha256 is None:
            raise ValueError("complete calibration state requires an artifact SHA-256")
        if self.status is not CalibrationItemStatus.COMPLETE and self.artifact_sha256 is not None:
            raise ValueError("only complete calibration state may carry an artifact SHA-256")
        return self


class CalibrationState(StrictModel):
    schema_version: Literal["1"]
    activity_id: str = Field(min_length=1)
    plan_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    items: list[CalibrationItemState] = Field(min_length=32, max_length=32)
    state_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_unique_cases(self) -> CalibrationState:
        case_ids = [item.case_id for item in self.items]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("calibration state contains duplicate case IDs")
        return self


class CalibrationRuntimeCase(StrictModel):
    claim_id: str = Field(min_length=1)
    case_id: str = Field(pattern=_SHA256_PATTERN)
    router_run_id: str = Field(pattern=_SHA256_PATTERN)
    single_run_id: str = Field(pattern=_SHA256_PATTERN)
    multi_run_id: str = Field(pattern=_SHA256_PATTERN)
    call_ids: list[str] = Field(min_length=1)
    runtime_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    features: ClaimFeatures
    saved_llm_route: Literal["single", "multi"]
    router_usage: Usage
    router_actual_cost_micro_cny: int = Field(ge=0)
    single_result: VerificationResult
    multi_result: VerificationResult
    requested_alias: str = Field(min_length=1)
    response_model_ids_raw: list[str] = Field(min_length=1)
    identity_verified: Literal[False] = False
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_case_identity(self) -> CalibrationRuntimeCase:
        if (
            self.single_result.claim_id != self.claim_id
            or self.multi_result.claim_id != self.claim_id
        ):
            raise ValueError("saved calibration results must match the case claim ID")
        if self.single_result.initial_route != "single":
            raise ValueError("single calibration result must have initial_route=single")
        if self.multi_result.initial_route != "multi":
            raise ValueError("multi calibration result must have initial_route=multi")
        if len(self.call_ids) != len(set(self.call_ids)):
            raise ValueError("calibration call IDs must be unique")
        if len({self.router_run_id, self.single_run_id, self.multi_run_id}) != 3:
            raise ValueError("calibration run IDs must be distinct")
        return self


class CalibrationScoredCase(StrictModel):
    runtime: CalibrationRuntimeCase
    gold_label: Verdict


class ReplayDecision(StrictModel):
    claim_id: str
    route: Literal["single", "multi"]
    route_source: Literal["rule", "llm"]
    executed_path: Literal[
        "single", "single_failed", "single_escalated_multi", "multi", "multi_failed"
    ]
    escalated: bool
    final_status: ResultStatus
    final_verdict: Verdict | None
    total_tokens: int = Field(ge=0)
    simulated_cost_micro_cny: int = Field(ge=0)
    llm_router_called: bool


class ReplayOutcome(StrictModel):
    final_result: VerificationResult
    decision: ReplayDecision
    executed_path: str
    total_tokens: int = Field(ge=0)
    simulated_cost_micro_cny: int = Field(ge=0)
    llm_router_calls: int = Field(ge=0, le=1)


class CandidateOutcome(StrictModel):
    config_hash: str = Field(pattern=_SHA256_PATTERN)
    macro_f1: float = Field(ge=0, le=1)
    total_tokens: int = Field(ge=0)
    llm_router_calls: int = Field(ge=0)
    llm_router_rate: float = Field(ge=0, le=1)
    simulated_cost_micro_cny: int = Field(ge=0)
    settings: dict[str, object]
    clear_multi_clauses: int | None = Field(default=None, ge=2)
    clear_single_min_sources: int | None = Field(default=None, ge=1)
    low_confidence: float | None = Field(default=None, ge=0, le=1)
    minimum_coverage: float | None = Field(default=None, ge=0, le=1)
    manifest_count: int = Field(default=0, ge=0)
    completed_count: int = Field(default=0, ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    decisions: list[ReplayDecision] = Field(default_factory=list)


def derive_case_id(activity_id: str, claim_id: str) -> str:
    if not activity_id or not claim_id:
        raise ValueError("activity and claim IDs must be non-empty")
    payload = f"{activity_id}\0calibration\0{claim_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def derive_run_id(case_id: str, path: Literal["router", "single", "multi"]) -> str:
    if len(case_id) != 64 or any(char not in "0123456789abcdef" for char in case_id):
        raise ValueError("case ID must be a lowercase SHA-256")
    return hashlib.sha256(f"{case_id}\0{path}".encode()).hexdigest()


def plan_fingerprint(plan: CalibrationPlan | Mapping[str, object]) -> str:
    payload = plan.model_dump(mode="json") if isinstance(plan, CalibrationPlan) else dict(plan)
    payload.pop("plan_fingerprint", None)
    return stable_hash(payload)


def state_fingerprint(state: CalibrationState | Mapping[str, object]) -> str:
    payload = state.model_dump(mode="json") if isinstance(state, CalibrationState) else dict(state)
    payload.pop("state_sha256", None)
    return stable_hash(payload)


def runtime_case_fingerprint(case: CalibrationRuntimeCase | Mapping[str, object]) -> str:
    payload = (
        case.model_dump(mode="json")
        if isinstance(case, CalibrationRuntimeCase)
        else dict(case)
    )
    payload.pop("artifact_sha256", None)
    return stable_hash(payload)


def verify_plan_fingerprint(plan: CalibrationPlan) -> None:
    if plan.plan_fingerprint != plan_fingerprint(plan):
        raise ValueError("calibration plan fingerprint mismatch")


def verify_state_fingerprint(state: CalibrationState) -> None:
    if state.state_sha256 != state_fingerprint(state):
        raise ValueError("calibration state SHA-256 mismatch")


def verify_runtime_case_fingerprint(case: CalibrationRuntimeCase) -> None:
    if case.artifact_sha256 != runtime_case_fingerprint(case):
        raise ValueError("calibration case artifact SHA-256 mismatch")


def candidate_grid() -> list[RoutingSettings]:
    """Return the finite, pre-registered 54-candidate routing grid."""

    return [
        RoutingSettings(
            clear_multi_clauses=clauses,
            clear_single_min_sources=sources,
            low_confidence=confidence,
            minimum_coverage=coverage,
        )
        for clauses in (2, 3)
        for sources in (1, 2, 3)
        for confidence in (0.55, 0.65, 0.75)
        for coverage in (0.50, 0.75, 1.0)
    ]


def _clear_multi(features: ClaimFeatures, settings: RoutingSettings) -> bool:
    return features.atomic_clause_count >= settings.clear_multi_clauses or (
        features.atomic_clause_count >= 2
        and (
            features.has_comparison
            or features.time_scope_count >= 2
            or features.probe_conflict_hint
        )
    )


def _clear_single(features: ClaimFeatures, settings: RoutingSettings) -> bool:
    return (
        features.atomic_clause_count == 1
        and not features.has_comparison
        and not features.has_causal
        and not features.has_contrast
        and not features.probe_conflict_hint
        and features.probe_source_count >= settings.clear_single_min_sources
    )


def _validate_saved_result(
    result: VerificationResult,
    case: CalibrationRuntimeCase,
    settings: RoutingSettings,
    *,
    route: Literal["single", "multi"],
) -> tuple[VerificationResult, ValidationAction]:
    validator = ResultValidator(
        low_confidence=settings.low_confidence,
        minimum_coverage=settings.minimum_coverage,
    )
    decision = validator.validate(
        result=result,
        claim_unit_ids=[unit.unit_id for unit in case.features.claim_units],
        evidence_ids=set(result.available_evidence_ids),
        escalation_count=0 if route == "single" else 1,
        strategy=Strategy.ADAPTIVE,
        draft_origin=route,
    )
    return decision.result, decision.action


def _result_cost(result: VerificationResult) -> int:
    return result.estimated_cost_micro_cny or 0


def replay_candidate(
    case: CalibrationScoredCase,
    settings: RoutingSettings,
) -> ReplayOutcome:
    """Replay one candidate using only immutable saved results and pure validation."""

    runtime = case.runtime
    if _clear_multi(runtime.features, settings):
        route: Literal["single", "multi"] = "multi"
        route_source: Literal["rule", "llm"] = "rule"
    elif _clear_single(runtime.features, settings):
        route = "single"
        route_source = "rule"
    else:
        route = runtime.saved_llm_route
        route_source = "llm"

    router_tokens = runtime.router_usage.total_tokens if route_source == "llm" else 0
    router_cost = runtime.router_actual_cost_micro_cny if route_source == "llm" else 0
    escalated = False
    if route == "single":
        single_result, validation_action = _validate_saved_result(
            runtime.single_result,
            runtime,
            settings,
            route="single",
        )
        total_tokens = router_tokens + runtime.single_result.usage.total_tokens
        total_cost = router_cost + _result_cost(runtime.single_result)
        if validation_action is ValidationAction.ESCALATE:
            final_result = runtime.multi_result
            executed_path = "single_escalated_multi"
            escalated = True
            total_tokens += runtime.multi_result.usage.total_tokens
            total_cost += _result_cost(runtime.multi_result)
        elif validation_action is ValidationAction.FAIL:
            final_result = single_result
            executed_path = "single_failed"
        else:
            final_result = single_result
            executed_path = "single"
    else:
        final_result, validation_action = _validate_saved_result(
            runtime.multi_result,
            runtime,
            settings,
            route="multi",
        )
        total_tokens = router_tokens + runtime.multi_result.usage.total_tokens
        total_cost = router_cost + _result_cost(runtime.multi_result)
        executed_path = (
            "multi_failed" if validation_action is ValidationAction.FAIL else "multi"
        )

    decision = ReplayDecision(
        claim_id=runtime.claim_id,
        route=route,
        route_source=route_source,
        executed_path=executed_path,
        escalated=escalated,
        final_status=final_result.status,
        final_verdict=final_result.verdict,
        total_tokens=total_tokens,
        simulated_cost_micro_cny=total_cost,
        llm_router_called=route_source == "llm",
    )
    return ReplayOutcome(
        final_result=final_result,
        decision=decision,
        executed_path=executed_path,
        total_tokens=total_tokens,
        simulated_cost_micro_cny=total_cost,
        llm_router_calls=int(route_source == "llm"),
    )


def score_candidate(
    cases: Sequence[CalibrationScoredCase],
    settings: RoutingSettings,
) -> CandidateOutcome:
    if not cases:
        raise ValueError("candidate replay requires at least one scored case")
    claim_ids = [case.runtime.claim_id for case in cases]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("candidate replay cases must have unique claim IDs")
    replayed = [replay_candidate(case, settings) for case in cases]
    gold = {case.runtime.claim_id: case.gold_label for case in cases}
    results = {
        case.runtime.claim_id: outcome.final_result
        for case, outcome in zip(cases, replayed, strict=True)
    }
    metrics = score_full_manifest(gold, results)
    decisions = [outcome.decision for outcome in replayed]
    llm_router_calls = sum(outcome.llm_router_calls for outcome in replayed)
    settings_payload = settings.model_dump(mode="json")
    return CandidateOutcome(
        config_hash=stable_hash(settings_payload),
        macro_f1=metrics.macro_f1,
        total_tokens=sum(outcome.total_tokens for outcome in replayed),
        llm_router_calls=llm_router_calls,
        llm_router_rate=llm_router_calls / len(cases),
        simulated_cost_micro_cny=sum(
            outcome.simulated_cost_micro_cny for outcome in replayed
        ),
        settings=settings_payload,
        clear_multi_clauses=settings.clear_multi_clauses,
        clear_single_min_sources=settings.clear_single_min_sources,
        low_confidence=settings.low_confidence,
        minimum_coverage=settings.minimum_coverage,
        manifest_count=len(cases),
        completed_count=metrics.completed_count,
        status_counts=dict(Counter(decision.final_status.value for decision in decisions)),
        decisions=decisions,
    )


def score_candidate_grid(
    cases: Sequence[CalibrationScoredCase],
) -> list[CandidateOutcome]:
    return [score_candidate(cases, settings) for settings in candidate_grid()]


def choose_candidate(
    outcomes: Sequence[CandidateOutcome],
    *,
    tolerance: float = 0.03,
) -> CandidateOutcome:
    if not outcomes:
        raise ValueError("candidate selection requires at least one outcome")
    if tolerance < 0 or tolerance > 1:
        raise ValueError("candidate quality tolerance must be between zero and one")
    best_macro_f1 = max(item.macro_f1 for item in outcomes)
    eligible = [item for item in outcomes if item.macro_f1 >= best_macro_f1 - tolerance]
    return min(
        eligible,
        key=lambda item: (item.total_tokens, item.llm_router_calls, item.config_hash),
    )


def _aligned_identity(aligned: object) -> tuple[object, object]:
    if isinstance(aligned, tuple) and len(aligned) == 2:
        return aligned
    runtime = getattr(aligned, "runtime", None)
    gold = getattr(aligned, "gold", None)
    if runtime is None or gold is None:
        raise ValueError("aligned calibration claims require runtime and gold values")
    return runtime, gold


def attach_calibration_gold(
    runtime_cases: Sequence[CalibrationRuntimeCase],
    aligned_claims: Sequence[object],
) -> list[CalibrationScoredCase]:
    """Attach scorer-only gold after validating the frozen 32-case train cohort."""

    if len(runtime_cases) != 32 or len(aligned_claims) != 32:
        raise ValueError("calibration gold attachment requires exactly 32 cases")
    case_ids = [case.claim_id for case in runtime_cases]
    if len(case_ids) != len(set(case_ids)) or any(
        not claim_id.startswith("train-") for claim_id in case_ids
    ):
        raise ValueError("calibration cases require 32 unique train claim IDs")
    manifest_hashes = {case.runtime_manifest_sha256 for case in runtime_cases}
    if len(manifest_hashes) != 1:
        raise ValueError("calibration cases reference different runtime manifests")

    scored: list[CalibrationScoredCase] = []
    labels: Counter[Verdict] = Counter()
    for case, aligned in zip(runtime_cases, aligned_claims, strict=True):
        runtime, gold = _aligned_identity(aligned)
        runtime_claim_id = getattr(runtime, "claim_id", None)
        gold_claim_id = getattr(gold, "claim_id", runtime_claim_id)
        if case.claim_id != runtime_claim_id or case.claim_id != gold_claim_id:
            raise ValueError("calibration runtime/gold claim order differs from saved cases")
        raw_label = gold if isinstance(gold, Verdict) else getattr(gold, "label", None)
        try:
            label = raw_label if isinstance(raw_label, Verdict) else Verdict(raw_label)
        except (TypeError, ValueError) as exc:
            raise ValueError("calibration gold row has an invalid label") from exc
        labels[label] += 1
        scored.append(CalibrationScoredCase(runtime=case, gold_label=label))
    if any(labels[verdict] != 8 for verdict in Verdict):
        raise ValueError("calibration gold must contain exactly eight cases of each verdict")
    return scored


def write_canonical_json(path: Path | str, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, StrictModel):
        value: object = payload.model_dump(mode="json")
    else:
        value = payload
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


__all__ = [
    "CalibrationItemState",
    "CalibrationItemStatus",
    "CalibrationPlan",
    "CalibrationRuntimeCase",
    "CalibrationScoredCase",
    "CalibrationState",
    "CalibrationWorkItem",
    "CandidateOutcome",
    "ReplayDecision",
    "ReplayOutcome",
    "attach_calibration_gold",
    "candidate_grid",
    "choose_candidate",
    "derive_case_id",
    "derive_run_id",
    "plan_fingerprint",
    "replay_candidate",
    "runtime_case_fingerprint",
    "score_candidate",
    "score_candidate_grid",
    "state_fingerprint",
    "verify_plan_fingerprint",
    "verify_runtime_case_fingerprint",
    "verify_state_fingerprint",
    "write_canonical_json",
]
