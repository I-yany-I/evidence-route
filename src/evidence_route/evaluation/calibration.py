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

import yaml
from pydantic import Field, model_validator

from evidence_route.config import (
    BudgetSettings,
    EvidenceSettings,
    GenerationSettings,
    HardeningSettings,
    LLMSettings,
    RoutingSettings,
    stable_hash,
)
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
        for item in self.items:
            if item.case_id != derive_case_id(self.activity_id, item.claim_id):
                raise ValueError("calibration plan case IDs must be deterministic")
            if item.router_run_id != derive_run_id(item.case_id, "router"):
                raise ValueError("calibration plan router run IDs must be deterministic")
            if item.single_run_id != derive_run_id(item.case_id, "single"):
                raise ValueError("calibration plan single run IDs must be deterministic")
            if item.multi_run_id != derive_run_id(item.case_id, "multi"):
                raise ValueError("calibration plan multi run IDs must be deterministic")
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
    request_sha256_by_call_id: dict[str, str] = Field(min_length=1)
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
        if set(self.request_sha256_by_call_id) != set(self.call_ids):
            raise ValueError("calibration request fingerprints must match call IDs")
        if any(
            len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
            for digest in self.request_sha256_by_call_id.values()
        ):
            raise ValueError("calibration request fingerprints must be lowercase SHA-256")
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


class CalibrationReplay(StrictModel):
    schema_version: Literal["1"] = "1"
    scope: Literal["train_calibration_only"] = "train_calibration_only"
    warning: Literal["Train-only policy selection; this is not final dev performance."] = (
        "Train-only policy selection; this is not final dev performance."
    )
    activity_id: str = Field(min_length=1)
    runtime_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    code_git_sha: str = Field(pattern=_GIT_SHA_PATTERN)
    candidate_count: Literal[54] = 54
    quality_tolerance: float = Field(ge=0, le=1)
    best_candidate_macro_f1: float = Field(ge=0, le=1)
    quality_floor_macro_f1: float = Field(ge=0, le=1)
    candidates: list[CandidateOutcome] = Field(min_length=54, max_length=54)
    baselines: dict[str, CandidateOutcome]
    selected: CandidateOutcome
    collection_accounting: dict[str, object]
    simulated_selected_accounting: dict[str, int | float]
    calibrated_config_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_selection(self) -> CalibrationReplay:
        if set(self.baselines) != {
            Strategy.ALWAYS_SINGLE.value,
            Strategy.ALWAYS_MULTI.value,
        }:
            raise ValueError("calibration report requires fixed single and multi baselines")
        hashes = [candidate.config_hash for candidate in self.candidates]
        if len(hashes) != len(set(hashes)):
            raise ValueError("calibration report candidate hashes must be unique")
        if self.selected.config_hash not in set(hashes):
            raise ValueError("selected calibration candidate is absent from the grid")
        expected_best = max(candidate.macro_f1 for candidate in self.candidates)
        if self.best_candidate_macro_f1 != expected_best:
            raise ValueError("calibration report best-candidate quality is inconsistent")
        expected_floor = max(0.0, expected_best - self.quality_tolerance)
        if self.quality_floor_macro_f1 != expected_floor:
            raise ValueError("calibration report quality floor is inconsistent")
        if self.selected.macro_f1 < self.quality_floor_macro_f1:
            raise ValueError("selected calibration candidate is below the quality floor")
        expected_selected = choose_candidate(self.candidates, tolerance=self.quality_tolerance)
        if self.selected != expected_selected:
            raise ValueError(
                "selected calibration candidate differs from the pre-registered tie-break"
            )
        return self


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
        case.model_dump(mode="json") if isinstance(case, CalibrationRuntimeCase) else dict(case)
    )
    # The marker was added after the frozen Gate A cases were written. An explicit false
    # is equivalent to the legacy omitted field; a true recovery marker remains auditable.
    for result_name in ("single_result", "multi_result"):
        result = payload.get(result_name)
        if isinstance(result, Mapping) and result.get("fallback_used") is False:
            normalized_result = dict(result)
            normalized_result.pop("fallback_used", None)
            payload[result_name] = normalized_result
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


def _claim_id(claim: object) -> str:
    if isinstance(claim, str):
        value = claim
        split = "train" if claim.startswith("train-") else None
    elif isinstance(claim, Mapping):
        value = claim.get("claim_id")
        split = claim.get("split")
    else:
        value = getattr(claim, "claim_id", None)
        split = getattr(claim, "split", None)
    if not isinstance(value, str) or not value:
        raise ValueError("calibration runtime claim must expose a non-empty claim_id")
    if split is not None and str(split) != "train":
        raise ValueError("calibration runtime claims must come from the train split")
    if not value.startswith("train-"):
        raise ValueError("calibration runtime claim IDs must use the train- prefix")
    return value


def build_calibration_plan(
    runtime_claims: Sequence[object],
    *,
    activity_id: str,
    manifest_freeze_git_sha: str,
    runtime_manifest_sha256: str,
    corpus_preparation_receipt_sha256: str,
    prompt_bundle_sha256: str,
    config_sha256: str,
    pricing_sha256: str,
    endpoint_config_sha256: str,
    requirements_lock_sha256: str,
    requested_alias: str,
    seed: int,
    cap_micro_cny: int,
) -> CalibrationPlan:
    """Freeze the exact 32-row train cohort before any provider call."""

    claims = list(runtime_claims)
    if len(claims) != 32:
        raise ValueError("calibration plan requires exactly 32 runtime claims")
    claim_ids = [_claim_id(claim) for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("calibration plan claim IDs must be unique")
    items: list[CalibrationWorkItem] = []
    for order, claim_id in enumerate(claim_ids):
        case_id = derive_case_id(activity_id, claim_id)
        items.append(
            CalibrationWorkItem(
                order=order,
                claim_id=claim_id,
                case_id=case_id,
                router_run_id=derive_run_id(case_id, "router"),
                single_run_id=derive_run_id(case_id, "single"),
                multi_run_id=derive_run_id(case_id, "multi"),
            )
        )
    payload: dict[str, object] = {
        "schema_version": "1",
        "activity_id": activity_id,
        "manifest_freeze_git_sha": manifest_freeze_git_sha,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "corpus_preparation_receipt_sha256": corpus_preparation_receipt_sha256,
        "prompt_bundle_sha256": prompt_bundle_sha256,
        "config_sha256": config_sha256,
        "pricing_sha256": pricing_sha256,
        "endpoint_config_sha256": endpoint_config_sha256,
        "requirements_lock_sha256": requirements_lock_sha256,
        "requested_alias": requested_alias,
        "seed": seed,
        "cap_micro_cny": cap_micro_cny,
        "items": [item.model_dump(mode="json") for item in items],
        "plan_fingerprint": "0" * 64,
    }
    payload["plan_fingerprint"] = plan_fingerprint(payload)
    return CalibrationPlan.model_validate(payload)


def build_calibration_state(plan: CalibrationPlan) -> CalibrationState:
    """Create the initial self-hashed journal in immutable plan order."""

    verify_plan_fingerprint(plan)
    payload: dict[str, object] = {
        "schema_version": "1",
        "activity_id": plan.activity_id,
        "plan_fingerprint": plan.plan_fingerprint,
        "items": [
            {
                "case_id": item.case_id,
                "status": CalibrationItemStatus.PENDING.value,
                "artifact_relpath": f"calibration/cases/{item.case_id}.json",
                "artifact_sha256": None,
                "errors": [],
            }
            for item in plan.items
        ],
        "state_sha256": "0" * 64,
    }
    payload["state_sha256"] = state_fingerprint(payload)
    return CalibrationState.model_validate(payload)


def build_calibration_runtime_case(
    *,
    plan: CalibrationPlan,
    work: CalibrationWorkItem,
    call_ids: Sequence[str],
    request_sha256_by_call_id: Mapping[str, str],
    features: ClaimFeatures,
    saved_llm_route: Literal["single", "multi"],
    router_usage: Usage,
    router_actual_cost_micro_cny: int,
    single_result: VerificationResult,
    multi_result: VerificationResult,
    response_model_ids_raw: Sequence[str],
) -> CalibrationRuntimeCase:
    """Seal one fully accounted router + single + one-to-three-worker multi collection."""

    verify_plan_fingerprint(plan)
    _index, expected_work = _work_for_case(plan, work.case_id)
    if expected_work != work:
        raise ValueError("calibration work item differs from the immutable plan")
    calls = list(call_ids)
    model_ids = list(response_model_ids_raw)
    # The base graph has router, single, decomposer, one-to-three workers, and judge
    # (five to seven calls). Repair attempts add calls, so do not infer worker count
    # from the aggregate call list; slot-level validation belongs to the persisted ledger.
    if len(calls) < 5:
        raise ValueError("complete calibration collection requires one to three workers")
    if len(model_ids) != len(calls):
        raise ValueError("calibration response model IDs must align with call IDs")
    if any(not value.strip() for value in calls + model_ids):
        raise ValueError("calibration call and response model IDs must be non-empty")
    if len(set(model_ids)) != 1:
        raise ValueError("calibration case observed response model drift")
    fingerprints = dict(request_sha256_by_call_id)
    if set(fingerprints) != set(calls):
        raise ValueError("calibration request fingerprints must match call IDs")
    payload: dict[str, object] = {
        "claim_id": work.claim_id,
        "case_id": work.case_id,
        "router_run_id": work.router_run_id,
        "single_run_id": work.single_run_id,
        "multi_run_id": work.multi_run_id,
        "call_ids": calls,
        "request_sha256_by_call_id": fingerprints,
        "runtime_manifest_sha256": plan.runtime_manifest_sha256,
        "features": features.model_dump(mode="json"),
        "saved_llm_route": saved_llm_route,
        "router_usage": router_usage.model_dump(mode="json"),
        "router_actual_cost_micro_cny": router_actual_cost_micro_cny,
        "single_result": single_result.model_dump(mode="json"),
        "multi_result": multi_result.model_dump(mode="json"),
        "requested_alias": plan.requested_alias,
        "response_model_ids_raw": model_ids,
        "identity_verified": False,
        "artifact_sha256": "0" * 64,
    }
    payload["artifact_sha256"] = runtime_case_fingerprint(payload)
    case = CalibrationRuntimeCase.model_validate(payload)
    _verify_case_against_work(case, work, plan)
    return case


def _read_model(path: Path, model_type: type[StrictModel], *, label: str) -> StrictModel:
    try:
        encoded = path.read_bytes()
        text = encoded.decode("utf-8")
        value = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    try:
        return model_type.model_validate(value)
    except Exception as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def _validate_plan_state(plan: CalibrationPlan, state: CalibrationState) -> None:
    verify_plan_fingerprint(plan)
    verify_state_fingerprint(state)
    if state.activity_id != plan.activity_id:
        raise ValueError("calibration state activity ID differs from the plan")
    if state.plan_fingerprint != plan.plan_fingerprint:
        raise ValueError("calibration state references a different plan")
    if [item.case_id for item in state.items] != [item.case_id for item in plan.items]:
        raise ValueError("calibration state case order differs from the plan")
    for work, item in zip(plan.items, state.items, strict=True):
        expected = f"calibration/cases/{work.case_id}.json"
        if Path(item.artifact_relpath).as_posix() != expected:
            raise ValueError("calibration state artifact path differs from the plan")


def persist_calibration_plan(
    path: Path | str,
    plan: CalibrationPlan,
    *,
    fresh: bool = False,
) -> None:
    """Atomically freeze a plan, permitting only byte-equivalent resume reuse."""

    destination = Path(path)
    verify_plan_fingerprint(plan)
    if destination.exists():
        existing = load_calibration_plan(destination)
        if fresh:
            raise FileExistsError("calibration plan already exists; use resume")
        if existing != plan:
            raise ValueError("persisted calibration plan differs from requested plan")
        return
    write_canonical_json(destination, plan)


def load_calibration_plan(path: Path | str) -> CalibrationPlan:
    plan = _read_model(Path(path), CalibrationPlan, label="calibration plan")
    assert isinstance(plan, CalibrationPlan)
    verify_plan_fingerprint(plan)
    return plan


def persist_calibration_state(path: Path | str, state: CalibrationState) -> None:
    """Reseal and atomically replace the mutable calibration journal."""

    state.state_sha256 = state_fingerprint(state)
    sealed = CalibrationState.model_validate(state.model_dump(mode="json"))
    verify_state_fingerprint(sealed)
    write_canonical_json(path, sealed)


def load_calibration_state(
    path: Path | str,
    *,
    expected_plan: CalibrationPlan | None = None,
) -> CalibrationState:
    state = _read_model(Path(path), CalibrationState, label="calibration state")
    assert isinstance(state, CalibrationState)
    verify_state_fingerprint(state)
    if expected_plan is not None:
        _validate_plan_state(expected_plan, state)
    return state


def _work_for_case(plan: CalibrationPlan, case_id: str) -> tuple[int, CalibrationWorkItem]:
    for index, work in enumerate(plan.items):
        if work.case_id == case_id:
            return index, work
    raise ValueError("calibration case is absent from the immutable plan")


def _verify_case_against_work(
    case: CalibrationRuntimeCase,
    work: CalibrationWorkItem,
    plan: CalibrationPlan,
) -> None:
    verify_runtime_case_fingerprint(case)
    expected = (
        work.claim_id,
        work.case_id,
        work.router_run_id,
        work.single_run_id,
        work.multi_run_id,
    )
    actual = (
        case.claim_id,
        case.case_id,
        case.router_run_id,
        case.single_run_id,
        case.multi_run_id,
    )
    if actual != expected:
        raise ValueError("calibration case identity differs from the plan")
    if case.runtime_manifest_sha256 != plan.runtime_manifest_sha256:
        raise ValueError("calibration case references a different runtime manifest")
    if case.requested_alias != plan.requested_alias:
        raise ValueError("calibration case requested alias differs from the plan")


def _case_path(root: Path | str, relative_path: str) -> Path:
    base = Path(root).resolve()
    destination = (base / relative_path).resolve()
    try:
        destination.relative_to(base)
    except ValueError as exc:
        raise ValueError("calibration case path escapes the activity directory") from exc
    return destination


def begin_calibration_case(
    plan: CalibrationPlan,
    state: CalibrationState,
    case_id: str,
    *,
    state_path: Path | str | None = None,
) -> CalibrationWorkItem:
    """Persist the single RUNNING journal entry before executing its first call."""

    _validate_plan_state(plan, state)
    index, work = _work_for_case(plan, case_id)
    running = [item.case_id for item in state.items if item.status is CalibrationItemStatus.RUNNING]
    if running and running != [case_id]:
        raise ValueError(f"calibration case {running[0]} is already RUNNING")
    item = state.items[index]
    if item.status is CalibrationItemStatus.COMPLETE:
        raise ValueError("completed calibration cases cannot be started again")
    if item.status is CalibrationItemStatus.STOPPED:
        raise ValueError("stopped calibration cases require an explicit recovery decision")
    item.status = CalibrationItemStatus.RUNNING
    item.artifact_sha256 = None
    state.state_sha256 = state_fingerprint(state)
    if state_path is not None:
        persist_calibration_state(state_path, state)
    return work


def persist_calibration_case(
    activity_root: Path | str,
    plan: CalibrationPlan,
    state: CalibrationState,
    case: CalibrationRuntimeCase,
    *,
    state_path: Path | str | None = None,
) -> str:
    """Close one self-hashed case and its journal entry without unsafe overwrite."""

    _validate_plan_state(plan, state)
    index, work = _work_for_case(plan, case.case_id)
    _verify_case_against_work(case, work, plan)
    item = state.items[index]
    destination = _case_path(activity_root, item.artifact_relpath)
    if destination.exists():
        existing = load_calibration_case(destination, work=work, plan=plan)
        if existing != case:
            raise ValueError("persisted calibration case differs from the completed result")
    else:
        write_canonical_json(destination, case)
    item.status = CalibrationItemStatus.COMPLETE
    item.artifact_sha256 = case.artifact_sha256
    item.errors = []
    state.state_sha256 = state_fingerprint(state)
    if state_path is not None:
        persist_calibration_state(state_path, state)
    return case.artifact_sha256


def load_calibration_case(
    path: Path | str,
    *,
    work: CalibrationWorkItem | None = None,
    plan: CalibrationPlan | None = None,
) -> CalibrationRuntimeCase:
    case = _read_model(Path(path), CalibrationRuntimeCase, label="calibration runtime case")
    assert isinstance(case, CalibrationRuntimeCase)
    verify_runtime_case_fingerprint(case)
    if (work is None) != (plan is None):
        raise ValueError("work and plan must be supplied together")
    if work is not None and plan is not None:
        _verify_case_against_work(case, work, plan)
    return case


def load_calibration_cases(
    activity_root: Path | str,
    plan: CalibrationPlan,
    state: CalibrationState,
    *,
    require_complete: bool = True,
) -> list[CalibrationRuntimeCase]:
    """Load verified cases once, in immutable plan order."""

    _validate_plan_state(plan, state)
    if require_complete and any(
        item.status is not CalibrationItemStatus.COMPLETE for item in state.items
    ):
        raise ValueError("calibration replay requires 32 completed case artifacts")
    cases: list[CalibrationRuntimeCase] = []
    for work, item in zip(plan.items, state.items, strict=True):
        if item.status is not CalibrationItemStatus.COMPLETE:
            continue
        if item.artifact_sha256 is None:
            raise ValueError("complete calibration state is missing an artifact SHA-256")
        path = _case_path(activity_root, item.artifact_relpath)
        if not path.is_file():
            raise ValueError(f"calibration case artifact is missing: {work.case_id}")
        case = load_calibration_case(path, work=work, plan=plan)
        if case.artifact_sha256 != item.artifact_sha256:
            raise ValueError("calibration state artifact SHA-256 differs from the case")
        cases.append(case)
    if require_complete and len(cases) != 32:
        raise ValueError("calibration replay requires 32 completed case artifacts")
    return cases


def reconcile_calibration_state(
    activity_root: Path | str,
    plan: CalibrationPlan,
    state: CalibrationState,
    *,
    state_path: Path | str | None = None,
) -> CalibrationState:
    """Reconcile a resume journal against already closed immutable case files."""

    _validate_plan_state(plan, state)
    running = [item for item in state.items if item.status is CalibrationItemStatus.RUNNING]
    if len(running) > 1:
        raise ValueError("calibration state contains multiple RUNNING items")
    for work, item in zip(plan.items, state.items, strict=True):
        destination = _case_path(activity_root, item.artifact_relpath)
        if destination.is_file():
            case = load_calibration_case(destination, work=work, plan=plan)
            if item.artifact_sha256 is not None and item.artifact_sha256 != case.artifact_sha256:
                raise ValueError("calibration state artifact SHA-256 differs from the case")
            item.status = CalibrationItemStatus.COMPLETE
            item.artifact_sha256 = case.artifact_sha256
            item.errors = []
            continue
        if item.status is CalibrationItemStatus.COMPLETE:
            raise ValueError(f"calibration case artifact is missing: {work.case_id}")
        if item.status is CalibrationItemStatus.RUNNING:
            item.status = CalibrationItemStatus.INTERRUPTED
            item.errors = [*item.errors, "PROCESS_INTERRUPTION"]
    state.state_sha256 = state_fingerprint(state)
    if state_path is not None:
        persist_calibration_state(state_path, state)
    return state


def write_calibration_replay_view(
    path: Path | str,
    cases: Sequence[CalibrationRuntimeCase],
    *,
    plan: CalibrationPlan | None = None,
) -> None:
    """Atomically regenerate JSONL from verified cases without append semantics."""

    if len(cases) != 32:
        raise ValueError("calibration replay view requires exactly 32 cases")
    if plan is not None:
        verify_plan_fingerprint(plan)
        for work, case in zip(plan.items, cases, strict=True):
            _verify_case_against_work(case, work, plan)
    else:
        for case in cases:
            verify_runtime_case_fingerprint(case)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(
                json.dumps(
                    case.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


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
    strategy: Strategy = Strategy.ADAPTIVE,
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
        strategy=strategy,
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
            # The saved multi path still has to pass the candidate's thresholds.  An
            # escalation consumes the one allowed retry, so validation can only accept
            # or fail here; it must never silently return an unvalidated result.
            final_result, _multi_validation_action = _validate_saved_result(
                runtime.multi_result,
                runtime,
                settings,
                route="multi",
            )
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
        executed_path = "multi_failed" if validation_action is ValidationAction.FAIL else "multi"

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
        simulated_cost_micro_cny=sum(outcome.simulated_cost_micro_cny for outcome in replayed),
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


def _score_fixed_baseline(
    cases: Sequence[CalibrationScoredCase],
    *,
    strategy: Literal["always_single", "always_multi"],
    routing: RoutingSettings,
) -> CandidateOutcome:
    if not cases:
        raise ValueError("fixed calibration baseline requires scored cases")
    decisions: list[ReplayDecision] = []
    results: dict[str, VerificationResult] = {}
    gold: dict[str, Verdict] = {}
    total_tokens = total_cost = 0
    for case in cases:
        runtime = case.runtime
        route: Literal["single", "multi"] = (
            "single" if strategy == Strategy.ALWAYS_SINGLE.value else "multi"
        )
        result, action = _validate_saved_result(
            runtime.single_result if route == "single" else runtime.multi_result,
            runtime,
            routing,
            route=route,
            strategy=(
                Strategy.ALWAYS_SINGLE
                if strategy == Strategy.ALWAYS_SINGLE.value
                else Strategy.ALWAYS_MULTI
            ),
        )
        source_result = runtime.single_result if route == "single" else runtime.multi_result
        tokens = source_result.usage.total_tokens
        cost = _result_cost(source_result)
        executed_path = f"{route}_failed" if action is ValidationAction.FAIL else route
        decisions.append(
            ReplayDecision(
                claim_id=runtime.claim_id,
                route=route,
                route_source="rule",
                executed_path=executed_path,
                escalated=False,
                final_status=result.status,
                final_verdict=result.verdict,
                total_tokens=tokens,
                simulated_cost_micro_cny=cost,
                llm_router_called=False,
            )
        )
        results[runtime.claim_id] = result
        gold[runtime.claim_id] = case.gold_label
        total_tokens += tokens
        total_cost += cost
    metrics = score_full_manifest(gold, results)
    settings: dict[str, object] = {
        "strategy": strategy,
        "routing": routing.model_dump(mode="json"),
    }
    return CandidateOutcome(
        config_hash=stable_hash(settings),
        macro_f1=metrics.macro_f1,
        total_tokens=total_tokens,
        llm_router_calls=0,
        llm_router_rate=0.0,
        simulated_cost_micro_cny=total_cost,
        settings=settings,
        manifest_count=len(cases),
        completed_count=metrics.completed_count,
        status_counts=dict(Counter(decision.final_status.value for decision in decisions)),
        decisions=decisions,
    )


def build_calibration_replay(
    cases: Sequence[CalibrationScoredCase | object],
    *,
    plan: CalibrationPlan,
    code_git_sha: str,
    collection_accounting: Mapping[str, object] | None = None,
    tolerance: float = 0.03,
) -> CalibrationReplay:
    """Replay the frozen grid and build the complete auditable train-only report."""

    verify_plan_fingerprint(plan)
    if len(cases) != 32:
        raise ValueError("calibration replay requires exactly 32 scored cases")
    scored = [
        item
        if isinstance(item, CalibrationScoredCase)
        else CalibrationScoredCase(
            runtime=getattr(item, "runtime", None),
            gold_label=getattr(item, "gold_label", None),
        )
        for item in cases
    ]
    for work, case in zip(plan.items, scored, strict=True):
        _verify_case_against_work(case.runtime, work, plan)
    labels = Counter(case.gold_label for case in scored)
    if any(labels[verdict] != 8 for verdict in Verdict):
        raise ValueError("calibration gold must contain exactly eight cases of each verdict")
    outcomes = score_candidate_grid(scored)
    selected = choose_candidate(outcomes, tolerance=tolerance)
    selected_routing = RoutingSettings.model_validate(selected.settings)
    best_macro_f1 = max(item.macro_f1 for item in outcomes)
    return CalibrationReplay(
        activity_id=plan.activity_id,
        runtime_manifest_sha256=plan.runtime_manifest_sha256,
        prompt_bundle_sha256=plan.prompt_bundle_sha256,
        code_git_sha=code_git_sha,
        quality_tolerance=tolerance,
        best_candidate_macro_f1=best_macro_f1,
        quality_floor_macro_f1=max(0.0, best_macro_f1 - tolerance),
        candidates=outcomes,
        baselines={
            Strategy.ALWAYS_SINGLE.value: _score_fixed_baseline(
                scored,
                strategy=Strategy.ALWAYS_SINGLE.value,
                routing=selected_routing,
            ),
            Strategy.ALWAYS_MULTI.value: _score_fixed_baseline(
                scored,
                strategy=Strategy.ALWAYS_MULTI.value,
                routing=selected_routing,
            ),
        },
        selected=selected,
        collection_accounting=dict(collection_accounting or {}),
        simulated_selected_accounting={
            "total_tokens": selected.total_tokens,
            "llm_router_calls": selected.llm_router_calls,
            "llm_router_rate": selected.llm_router_rate,
            "simulated_cost_micro_cny": selected.simulated_cost_micro_cny,
        },
    )


def _load_and_validate_file_config(path: Path | str) -> dict[str, object]:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"configuration is not valid UTF-8 YAML: {source}") from exc
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    allowed = {"llm", "routing", "hardening", "evidence", "generation", "budget"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"configuration contains unknown sections: {sorted(unknown)}")
    llm = raw.get("llm") or {}
    if not isinstance(llm, dict):
        raise ValueError("llm configuration must be a mapping")
    environment_only = {"base_url", "api_key", "requested_alias"}
    if environment_only & set(llm):
        raise ValueError("provider endpoint and credentials must not be written to config files")
    LLMSettings.model_validate(
        {
            **llm,
            "base_url": "https://invalid.local/v1",
            "api_key": "validation-only",
            "requested_alias": "validation-only",
        }
    )
    RoutingSettings.model_validate(raw.get("routing") or {})
    HardeningSettings.model_validate(raw.get("hardening") or {})
    EvidenceSettings.model_validate(raw.get("evidence") or {})
    GenerationSettings.model_validate(raw.get("generation") or {})
    BudgetSettings.model_validate(raw.get("budget") or {})
    return raw


def _write_yaml_atomic(path: Path | str, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = yaml.safe_dump(
        dict(payload),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def write_calibration_outputs(
    *,
    source_config: Path | str,
    output_config: Path | str,
    output_report: Path | str,
    replay: CalibrationReplay,
) -> CalibrationReplay:
    """Write calibrated YAML and report while preserving every non-routing value."""

    original = _load_and_validate_file_config(source_config)
    selected_routing = RoutingSettings.model_validate(replay.selected.settings)
    calibrated = dict(original)
    calibrated["routing"] = selected_routing.model_dump(mode="json")
    _write_yaml_atomic(output_config, calibrated)
    reloaded = _load_and_validate_file_config(output_config)
    original_nonrouting = {key: value for key, value in original.items() if key != "routing"}
    reloaded_nonrouting = {key: value for key, value in reloaded.items() if key != "routing"}
    if reloaded_nonrouting != original_nonrouting:
        raise ValueError("calibrated config changed a non-routing value")
    if reloaded.get("routing") != selected_routing.model_dump(mode="json"):
        raise ValueError("calibrated config routing values differ from the selected candidate")
    replay.calibrated_config_sha256 = hashlib.sha256(Path(output_config).read_bytes()).hexdigest()
    sealed = CalibrationReplay.model_validate(replay.model_dump(mode="json"))
    write_canonical_json(output_report, sealed)
    return sealed


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
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "CalibrationItemState",
    "CalibrationItemStatus",
    "CalibrationPlan",
    "CalibrationRuntimeCase",
    "CalibrationReplay",
    "CalibrationScoredCase",
    "CalibrationState",
    "CalibrationWorkItem",
    "CandidateOutcome",
    "ReplayDecision",
    "ReplayOutcome",
    "attach_calibration_gold",
    "begin_calibration_case",
    "build_calibration_plan",
    "build_calibration_replay",
    "build_calibration_runtime_case",
    "build_calibration_state",
    "candidate_grid",
    "choose_candidate",
    "derive_case_id",
    "derive_run_id",
    "load_calibration_case",
    "load_calibration_cases",
    "load_calibration_plan",
    "load_calibration_state",
    "persist_calibration_case",
    "persist_calibration_plan",
    "persist_calibration_state",
    "plan_fingerprint",
    "reconcile_calibration_state",
    "replay_candidate",
    "runtime_case_fingerprint",
    "score_candidate",
    "score_candidate_grid",
    "state_fingerprint",
    "verify_plan_fingerprint",
    "verify_runtime_case_fingerprint",
    "verify_state_fingerprint",
    "write_calibration_replay_view",
    "write_calibration_outputs",
    "write_canonical_json",
]
