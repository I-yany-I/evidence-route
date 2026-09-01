"""Immutable campaign/activity contracts used by the Gate A evaluator.

The campaign runner owns persistence and execution.  This module deliberately contains only
serialisable records, deterministic identity helpers, and validation that can be performed
without opening a run store.  Checks that need files or SQLite are exposed as ``verify_*``
functions and are called by the runner/reporting layer.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from evidence_route.config import stable_hash
from evidence_route.contracts import ResultStatus, Strategy, StrictModel, Usage, VerificationResult

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_SHA_PATTERN = r"^[0-9a-f]{40}$"


class WorkStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    STOPPED = "stopped"
    NOT_RUN_BUDGET = "not_run_budget"
    CANCELLED = "cancelled"


class CampaignStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETE = "complete"
    INCOMPLETE_BUDGET = "incomplete_budget"
    INCOMPLETE_MODEL_DRIFT = "incomplete_model_drift"
    INCOMPLETE_USAGE = "incomplete_usage"
    INCOMPLETE_COST_UNCERTAIN = "incomplete_cost_uncertain"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CampaignStopReason(StrEnum):
    PROCESS_INTERRUPTION = "process_interruption"
    BUDGET = "budget"
    MODEL_DRIFT = "model_drift"
    USAGE_MISSING = "usage_missing"
    BILLING_UNCERTAIN = "billing_uncertain"
    USER_CANCELLED = "user_cancelled"
    INTERNAL_ERROR = "internal_error"
    USER_PAUSED = "user_paused"


class FreezeMismatch(ValueError):
    """Raised when a resumed campaign no longer matches its immutable freeze."""

    def __init__(
        self,
        field: str,
        expected: object | None = None,
        actual: object | None = None,
    ) -> None:
        self.field = field
        self.expected = expected
        self.actual = actual
        if expected is None and actual is None:
            message = f"frozen field changed: {field}"
        else:
            message = f"frozen field changed: {field} (expected={expected!r}, actual={actual!r})"
        super().__init__(message)


class CallProfile(StrictModel):
    """Upper-bound call counts by graph node."""

    router: int = Field(ge=0)
    single: int = Field(ge=0)
    decomposer: int = Field(ge=0)
    worker: int = Field(ge=0)
    judge: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.router + self.single + self.decomposer + self.worker + self.judge

    def __add__(self, other: object) -> CallProfile:
        if not isinstance(other, CallProfile):
            return NotImplemented
        return CallProfile(
            router=self.router + other.router,
            single=self.single + other.single,
            decomposer=self.decomposer + other.decomposer,
            worker=self.worker + other.worker,
            judge=self.judge + other.judge,
        )


class CallBounds(StrictModel):
    """Integer call and reservation limits persisted with a campaign plan."""

    base_call_upper_bound: int = Field(ge=0)
    repair_upper_bound: int = Field(ge=0)
    fault_upper_bound: int = Field(ge=0)
    base_cost_micro_cny: int = Field(ge=0)
    repair_cost_micro_cny: int = Field(ge=0)
    fault_cost_micro_cny: int = Field(ge=0)
    reserve_basis_points: int = Field(ge=0, le=10_000)
    startup_required_micro_cny: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_monotonic_bounds(self) -> CallBounds:
        if self.repair_upper_bound < self.base_call_upper_bound:
            raise ValueError("repair call bound cannot be below base call bound")
        if self.fault_upper_bound < self.repair_upper_bound:
            raise ValueError("fault call bound cannot be below repair call bound")
        if self.repair_cost_micro_cny < self.base_cost_micro_cny:
            raise ValueError("repair cost bound cannot be below base cost bound")
        if self.fault_cost_micro_cny < self.repair_cost_micro_cny:
            raise ValueError("fault cost bound cannot be below repair cost bound")
        return self


class FreezeIdentity(StrictModel):
    manifest_freeze_git_sha: str = Field(pattern=_GIT_SHA_PATTERN)
    dev_protocol_git_sha: str = Field(pattern=_GIT_SHA_PATTERN)
    calibration_runtime_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    dev_runtime_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    stability_runtime_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_preparation_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    pricing_sha256: str = Field(pattern=_SHA256_PATTERN)
    endpoint_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    requirements_lock_sha256: str = Field(pattern=_SHA256_PATTERN)
    requested_alias: str = Field(min_length=1)
    seed: int


class LatencyBreakdown(StrictModel):
    fresh_end_to_end_ms: int = Field(ge=0)
    model_active_ms: int = Field(ge=0)
    retry_ms: int = Field(ge=0)
    queue_ms: int = Field(ge=0)
    checkpoint_downtime_ms: int = Field(ge=0)
    total_elapsed_ms: int = Field(ge=0)
    interruption_count: int = Field(ge=0)


class CampaignWorkItem(StrictModel):
    order: int = Field(ge=0)
    phase: Literal["dev", "stability"]
    claim_id: str = Field(min_length=1)
    strategy: Strategy
    repeat: int = Field(ge=0, le=2)
    run_id: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_phase_repeat(self) -> CampaignWorkItem:
        if self.phase == "dev" and self.repeat != 0:
            raise ValueError("dev work items must use repeat=0")
        if self.phase == "stability":
            if self.strategy is not Strategy.ADAPTIVE:
                raise ValueError("stability work items must use the adaptive strategy")
            if self.repeat not in (1, 2):
                raise ValueError("stability work items must use repeat=1 or repeat=2")
        return self


class StabilityBaselineLink(StrictModel):
    claim_id: str = Field(min_length=1)
    dev_adaptive_run_id: str = Field(pattern=_SHA256_PATTERN)


class CampaignItemState(StrictModel):
    run_id: str = Field(pattern=_SHA256_PATTERN)
    status: WorkStatus = WorkStatus.PENDING
    artifact_relpath: str = Field(min_length=1)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    stop_reason: CampaignStopReason | None = None
    interrupted_at: str | None = None
    resumed_at: str | None = None
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_artifact_state(self) -> CampaignItemState:
        terminal_with_artifact = {
            WorkStatus.COMPLETED,
            WorkStatus.PARTIAL,
            WorkStatus.FAILED,
        }
        no_artifact = {
            WorkStatus.PENDING,
            WorkStatus.RUNNING,
            WorkStatus.NOT_RUN_BUDGET,
            WorkStatus.CANCELLED,
        }
        if self.status in terminal_with_artifact and self.artifact_sha256 is None:
            raise ValueError(f"{self.status.value} item requires an artifact SHA-256")
        if self.status is WorkStatus.STOPPED and self.artifact_sha256 is None:
            raise ValueError("stopped item requires a diagnostic artifact SHA-256")
        if self.status in no_artifact and self.artifact_sha256 is not None:
            raise ValueError(f"{self.status.value} item cannot carry an artifact SHA-256")
        if self.status is WorkStatus.STOPPED and self.stop_reason is None:
            raise ValueError("stopped item requires a typed stop reason")
        if self.status is WorkStatus.INTERRUPTED and self.stop_reason not in {
            None,
            CampaignStopReason.PROCESS_INTERRUPTION,
        }:
            raise ValueError("interrupted item may only carry process_interruption")
        if self.status is WorkStatus.NOT_RUN_BUDGET and self.stop_reason not in {
            None,
            CampaignStopReason.BUDGET,
        }:
            raise ValueError("budget-skipped item may only carry budget")
        if self.status is WorkStatus.CANCELLED and self.stop_reason not in {
            None,
            CampaignStopReason.USER_CANCELLED,
        }:
            raise ValueError("cancelled item may only carry user_cancelled")
        if (
            self.status
            not in {
                WorkStatus.STOPPED,
                WorkStatus.INTERRUPTED,
                WorkStatus.NOT_RUN_BUDGET,
                WorkStatus.CANCELLED,
            }
            and self.stop_reason is not None
        ):
            raise ValueError("stop reason is only valid for a stopped/unfinished item")
        return self


class CampaignPlan(StrictModel):
    schema_version: Literal["1"]
    activity_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    freeze: FreezeIdentity
    cap_micro_cny: int = Field(gt=0)
    call_bounds: CallBounds
    schedule: list[CampaignWorkItem] = Field(min_length=1)
    stability_repeat_zero_links: list[StabilityBaselineLink] = Field(min_length=20, max_length=20)
    campaign_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_schedule(self) -> CampaignPlan:
        orders = [item.order for item in self.schedule]
        if orders != list(range(len(self.schedule))):
            raise ValueError("campaign schedule orders must be contiguous and immutable")
        run_ids = [item.run_id for item in self.schedule]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("campaign schedule run IDs must be unique")
        expected_ids = [
            derive_run_id(
                self.campaign_id,
                item.phase,
                item.claim_id,
                item.strategy,
                item.repeat,
            )
            for item in self.schedule
        ]
        if run_ids != expected_ids:
            raise ValueError("campaign schedule contains a non-deterministic run ID")
        phases = [item.phase for item in self.schedule]
        first_stability = next(
            (index for index, phase in enumerate(phases) if phase == "stability"),
            len(phases),
        )
        if any(phase == "dev" for phase in phases[first_stability:]):
            raise ValueError("campaign schedule must keep a dev/stability phase boundary")
        link_claims = [link.claim_id for link in self.stability_repeat_zero_links]
        if len(link_claims) != len(set(link_claims)):
            raise ValueError("stability baseline links must have unique claim IDs")
        adaptive_dev_ids = {
            item.claim_id: item.run_id
            for item in self.schedule
            if item.phase == "dev" and item.strategy is Strategy.ADAPTIVE and item.repeat == 0
        }
        for link in self.stability_repeat_zero_links:
            if adaptive_dev_ids.get(link.claim_id) != link.dev_adaptive_run_id:
                raise ValueError(
                    "stability baseline link must reference the matching dev adaptive run"
                )
        stability_repeats: dict[str, set[int]] = {}
        for item in self.schedule:
            if item.phase == "stability":
                stability_repeats.setdefault(item.claim_id, set()).add(item.repeat)
        if set(stability_repeats) != set(link_claims) or any(
            repeats != {1, 2} for repeats in stability_repeats.values()
        ):
            raise ValueError(
                "each stability baseline must have exactly scheduled repeats one and two"
            )
        scheduled_stability_claims: list[str] = []
        for item in self.schedule[first_stability:]:
            if not scheduled_stability_claims or scheduled_stability_claims[-1] != item.claim_id:
                scheduled_stability_claims.append(item.claim_id)
        if scheduled_stability_claims != link_claims:
            raise ValueError("stability baseline links must match scheduled manifest order")
        for item in self.schedule[first_stability:]:
            if item.strategy is not Strategy.ADAPTIVE or item.repeat not in {1, 2}:
                raise ValueError("stability schedule must contain adaptive repeats one and two")
        return self


class RunArtifact(StrictModel):
    schema_version: Literal["1"]
    activity_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    phase: Literal["calibration", "dev", "stability"]
    run_id: str = Field(pattern=_SHA256_PATTERN)
    claim_id: str = Field(min_length=1)
    strategy: Strategy
    repeat: int = Field(ge=0, le=2)
    result: VerificationResult
    call_ids: list[str]
    usage: Usage
    usage_source: Literal["provider", "missing", "not_applicable"]
    actual_cost_micro_cny: int | None = Field(default=None, ge=0)
    known_actual_cost_micro_cny: int = Field(ge=0)
    committed_cost_micro_cny: int = Field(ge=0)
    cost_is_lower_bound: bool
    fresh_call_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    requested_alias: str = Field(min_length=1)
    response_model_ids_raw: list[str]
    identity_verified: Literal[False] = False
    billing_uncertain: bool
    diagnostic_only: bool = False
    latency: LatencyBreakdown
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_accounting(self) -> RunArtifact:
        if self.result.claim_id != self.claim_id:
            raise ValueError("run artifact claim ID differs from result claim ID")
        if self.result.usage != self.usage:
            raise ValueError("run artifact result usage differs from artifact usage")
        if self.actual_cost_micro_cny is not None and (
            self.result.estimated_cost_micro_cny != self.actual_cost_micro_cny
            or self.result.cost_currency != "CNY"
            or not self.result.price_config_id
            or not self.result.price_config_id.strip()
        ):
            raise ValueError("run artifact result cost differs from exact artifact cost")
        if len(self.call_ids) != len(set(self.call_ids)):
            raise ValueError("run artifact call IDs must be unique")
        if self.fresh_call_count != len(self.call_ids):
            raise ValueError("fresh call count must equal persisted call IDs")
        if len(self.response_model_ids_raw) != len(set(self.response_model_ids_raw)):
            raise ValueError("run artifact raw model IDs must be unique")
        if self.known_actual_cost_micro_cny > self.committed_cost_micro_cny:
            raise ValueError("known actual cost cannot exceed committed cost")
        if self.actual_cost_micro_cny is not None and (
            self.actual_cost_micro_cny != self.known_actual_cost_micro_cny
        ):
            raise ValueError("exact actual cost must equal known actual cost")
        if self.actual_cost_micro_cny is not None and self.cost_is_lower_bound:
            raise ValueError("exact actual cost cannot be marked as a lower bound")
        if self.actual_cost_micro_cny is None and not self.cost_is_lower_bound:
            raise ValueError("unknown actual cost must be marked as a lower bound")
        if self.usage_source == "provider" and not self.usage.complete:
            raise ValueError("provider usage source requires complete usage")
        if self.usage_source == "missing" and self.usage.complete:
            raise ValueError("missing usage source requires incomplete usage")
        if self.usage_source == "not_applicable" and (
            not self.usage.complete
            or self.usage.input_tokens != 0
            or self.usage.output_tokens != 0
            or self.usage.total_tokens != 0
        ):
            raise ValueError("not_applicable usage must be complete and zero")
        if self.phase == "calibration" and self.repeat != 0:
            raise ValueError("calibration artifacts must use repeat=0")
        if self.phase == "stability" and self.strategy is not Strategy.ADAPTIVE:
            raise ValueError("stability artifacts must use the adaptive strategy")

        scored_failure = (
            self.result.status is ResultStatus.FAILED
            and not self.call_ids
            and self.result.failure_stage == "pre_route"
            and "PROBE_RETRIEVAL_FAILED" in self.result.errors
        )
        if scored_failure:
            if (
                self.usage_source != "not_applicable"
                or not self.usage.complete
                or self.usage.total_tokens != 0
                or self.actual_cost_micro_cny != 0
                or self.known_actual_cost_micro_cny != 0
                or self.committed_cost_micro_cny != 0
                or self.response_model_ids_raw
                or self.diagnostic_only
            ):
                raise ValueError(
                    "pre-route failed artifact must be a zero-call diagnostic-free result"
                )
        elif not self.diagnostic_only:
            if self.result.status in {ResultStatus.COMPLETED, ResultStatus.PARTIAL}:
                if (
                    self.usage_source != "provider"
                    or not self.usage.complete
                    or self.actual_cost_micro_cny is None
                    or not self.call_ids
                    or not any(model_id.strip() for model_id in self.response_model_ids_raw)
                ):
                    raise ValueError("scored artifact requires complete provider accounting")
            elif self.result.status is ResultStatus.FAILED and self.call_ids:
                if self.usage_source != "provider" or not self.usage.complete:
                    raise ValueError("called failed artifact requires complete provider usage")
        else:
            if not (
                self.billing_uncertain or not self.usage.complete or self.response_model_ids_raw
            ):
                # A diagnostic artifact is allowed for model drift, which is represented by
                # a changed/non-empty response ID, or for one of the two accounting stops.
                raise ValueError("diagnostic artifact lacks a typed accounting/model diagnostic")
        if self.billing_uncertain and not self.diagnostic_only:
            raise ValueError("billing-uncertain artifacts must be diagnostic_only")
        return self


class RunSummary(StrictModel):
    run_ids: list[str]
    call_ids: list[str]
    usage: Usage
    usage_sources: list[Literal["provider", "missing"]]
    actual_cost_micro_cny: int | None = Field(default=None, ge=0)
    known_actual_cost_micro_cny: int = Field(ge=0)
    committed_cost_micro_cny: int = Field(ge=0)
    cost_is_lower_bound: bool
    fresh_call_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    transport_attempts: int = Field(ge=0)
    requested_aliases: list[str]
    response_model_ids_raw: list[str]
    identity_verified: Literal[False] = False
    billing_uncertain: bool

    @model_validator(mode="after")
    def validate_summary_accounting(self) -> RunSummary:
        if len(self.run_ids) != len(set(self.run_ids)):
            raise ValueError("run summary run IDs must be unique")
        if len(self.call_ids) != len(set(self.call_ids)):
            raise ValueError("run summary call IDs must be unique")
        if self.fresh_call_count != len(self.call_ids):
            raise ValueError("fresh call count must equal persisted call IDs")
        if self.known_actual_cost_micro_cny > self.committed_cost_micro_cny:
            raise ValueError("known actual cost cannot exceed committed cost")
        if self.actual_cost_micro_cny is not None and (
            self.actual_cost_micro_cny != self.known_actual_cost_micro_cny
        ):
            raise ValueError("exact actual cost must equal known actual cost")
        if self.actual_cost_micro_cny is not None and self.cost_is_lower_bound:
            raise ValueError("exact actual cost cannot be marked as a lower bound")
        if self.actual_cost_micro_cny is None and not self.cost_is_lower_bound:
            raise ValueError("unknown actual cost must be marked as a lower bound")
        if not self.usage.complete and self.actual_cost_micro_cny is not None:
            raise ValueError("incomplete usage cannot carry exact actual cost")
        if self.billing_uncertain and not self.cost_is_lower_bound:
            raise ValueError("billing uncertainty must mark cost as a lower bound")
        return self


class CampaignState(StrictModel):
    schema_version: Literal["1"]
    activity_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    status: CampaignStatus
    stop_reason: CampaignStopReason | None = None
    items: list[CampaignItemState]
    stability_repeat_zero_artifact_sha256s: dict[str, str]
    observed_response_model_ids_raw: list[str]
    billing_uncertain: bool = False
    summary: RunSummary | None = None

    @model_validator(mode="after")
    def validate_state(self) -> CampaignState:
        run_ids = [item.run_id for item in self.items]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("campaign state run IDs must be unique")
        for claim_id, digest in self.stability_repeat_zero_artifact_sha256s.items():
            if not claim_id:
                raise ValueError("stability baseline claim IDs must be non-empty")
            if not _is_sha256(digest):
                raise ValueError("stability baseline artifact hashes must be lowercase SHA-256")
        if self.billing_uncertain != (self.stop_reason is CampaignStopReason.BILLING_UNCERTAIN):
            raise ValueError(
                "billing_uncertain must match a billing_uncertain campaign stop reason"
            )
        if self.stop_reason is not None:
            expected = _status_for_stop_reason(self.stop_reason)
            if self.status is not expected:
                raise ValueError("campaign stop reason does not match campaign status")
        return self


class CalibrationArtifactLink(StrictModel):
    case_id: str = Field(pattern=_SHA256_PATTERN)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)


class ActivityRecord(StrictModel):
    schema_version: Literal["1"]
    activity_id: str = Field(min_length=1)
    status: CampaignStatus
    calibration_status: CampaignStatus
    dev_status: CampaignStatus
    stability_status: CampaignStatus
    calibration_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    calibration_state_sha256: str = Field(pattern=_SHA256_PATTERN)
    calibration_artifacts: list[CalibrationArtifactLink] = Field(min_length=32, max_length=32)
    freeze: FreezeIdentity | None = None
    campaign_plan_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    campaign_state_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    observed_response_model_ids_raw: list[str]
    identity_verified: Literal[False] = False
    billing_uncertain: bool = False
    stop_reason: CampaignStopReason | None = None
    summary: RunSummary | None = None

    @model_validator(mode="after")
    def validate_activity(self) -> ActivityRecord:
        case_ids = [link.case_id for link in self.calibration_artifacts]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("calibration artifact links must have unique case IDs")
        if self.calibration_status is CampaignStatus.COMPLETE and any(
            link.artifact_sha256 is None for link in self.calibration_artifacts
        ):
            raise ValueError("complete calibration status requires all artifact links")
        if (self.campaign_plan_sha256 is None) != (self.campaign_state_sha256 is None):
            raise ValueError("campaign plan and state hashes must be present together")
        if self.billing_uncertain != (self.stop_reason is CampaignStopReason.BILLING_UNCERTAIN):
            raise ValueError(
                "billing_uncertain must match a billing_uncertain activity stop reason"
            )
        if self.stop_reason is not None:
            expected = _status_for_stop_reason(self.stop_reason)
            if self.status is not expected:
                raise ValueError("activity stop reason does not match activity status")
        if self.status is CampaignStatus.COMPLETE:
            if not (
                self.calibration_status is CampaignStatus.COMPLETE
                and self.dev_status is CampaignStatus.COMPLETE
                and self.stability_status is CampaignStatus.COMPLETE
                and self.freeze is not None
                and self.campaign_plan_sha256 is not None
                and self.campaign_state_sha256 is not None
                and not self.billing_uncertain
                and len(self.observed_response_model_ids_raw) == 1
                and self.summary is not None
                and self.summary.usage.complete
                and not self.summary.billing_uncertain
                and len(self.summary.requested_aliases) == 1
            ):
                raise ValueError("complete activity does not contain all publication invariants")
        return self


_STOP_PRECEDENCE: tuple[CampaignStopReason, ...] = (
    CampaignStopReason.BILLING_UNCERTAIN,
    CampaignStopReason.USAGE_MISSING,
    CampaignStopReason.MODEL_DRIFT,
    CampaignStopReason.BUDGET,
    CampaignStopReason.INTERNAL_ERROR,
    CampaignStopReason.USER_CANCELLED,
    CampaignStopReason.PROCESS_INTERRUPTION,
    CampaignStopReason.USER_PAUSED,
)

_STOP_STATUS: dict[CampaignStopReason, CampaignStatus] = {
    CampaignStopReason.BILLING_UNCERTAIN: CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
    CampaignStopReason.USAGE_MISSING: CampaignStatus.INCOMPLETE_USAGE,
    CampaignStopReason.MODEL_DRIFT: CampaignStatus.INCOMPLETE_MODEL_DRIFT,
    CampaignStopReason.BUDGET: CampaignStatus.INCOMPLETE_BUDGET,
    CampaignStopReason.INTERNAL_ERROR: CampaignStatus.FAILED,
    CampaignStopReason.USER_CANCELLED: CampaignStatus.CANCELLED,
    CampaignStopReason.PROCESS_INTERRUPTION: CampaignStatus.INTERRUPTED,
    CampaignStopReason.USER_PAUSED: CampaignStatus.PAUSED,
}

_STATUS_STOP: dict[CampaignStatus, CampaignStopReason] = {
    status: reason for reason, status in _STOP_STATUS.items()
}


def _status_for_stop_reason(reason: CampaignStopReason) -> CampaignStatus:
    return _STOP_STATUS[reason]


def highest_stop_reason(
    reasons: Iterable[CampaignStopReason | str | None],
) -> CampaignStopReason | None:
    """Return the most safety-critical reason using the frozen protocol order."""

    present = {_coerce_stop_reason(reason) for reason in reasons if reason is not None}
    return next((reason for reason in _STOP_PRECEDENCE if reason in present), None)


def result_status_to_work_status(status: ResultStatus | str) -> WorkStatus:
    """Map a terminal graph result to the corresponding campaign work state."""

    try:
        result = status if isinstance(status, ResultStatus) else ResultStatus(status)
    except ValueError as exc:
        raise ValueError(f"unsupported result status: {status!r}") from exc
    return {
        ResultStatus.COMPLETED: WorkStatus.COMPLETED,
        ResultStatus.PARTIAL: WorkStatus.PARTIAL,
        ResultStatus.FAILED: WorkStatus.FAILED,
    }[result]


def _coerce_work_status(
    item: CampaignItemState | WorkStatus | str | Mapping[str, Any],
) -> WorkStatus:
    if isinstance(item, CampaignItemState):
        return item.status
    if isinstance(item, Mapping):
        return WorkStatus(item["status"])
    return item if isinstance(item, WorkStatus) else WorkStatus(item)


def _coerce_stop_reason(value: CampaignStopReason | str) -> CampaignStopReason:
    return value if isinstance(value, CampaignStopReason) else CampaignStopReason(value)


def _reason_from_item(item: object) -> CampaignStopReason | None:
    if isinstance(item, CampaignItemState):
        return item.stop_reason
    if isinstance(item, Mapping):
        value = item.get("stop_reason")
        return None if value is None else _coerce_stop_reason(value)
    return None


def derive_campaign_status(
    items: Sequence[CampaignItemState | WorkStatus | str | Mapping[str, Any]],
    stop_reason: CampaignStopReason | str | None = None,
    *,
    stop_reasons: Iterable[CampaignStopReason | str] | None = None,
) -> CampaignStatus:
    """Derive campaign status with the fixed safety-stop precedence.

    Partial and failed *predictions* are terminal quality outcomes and therefore still permit a
    ``COMPLETE`` campaign.  Safety stops and unfinished work never do.
    """

    reasons: list[CampaignStopReason] = []
    if stop_reason is not None:
        reasons.append(_coerce_stop_reason(stop_reason))
    if stop_reasons is not None:
        reasons.extend(_coerce_stop_reason(reason) for reason in stop_reasons)
    reasons.extend(reason for item in items if (reason := _reason_from_item(item)) is not None)
    if reason := highest_stop_reason(reasons):
        return _status_for_stop_reason(reason)

    statuses = [_coerce_work_status(item) for item in items]
    if not statuses:
        return CampaignStatus.PLANNED
    if all(
        status in {WorkStatus.COMPLETED, WorkStatus.PARTIAL, WorkStatus.FAILED}
        for status in statuses
    ):
        return CampaignStatus.COMPLETE
    if WorkStatus.NOT_RUN_BUDGET in statuses:
        return CampaignStatus.INCOMPLETE_BUDGET
    if WorkStatus.CANCELLED in statuses:
        return CampaignStatus.CANCELLED
    if WorkStatus.STOPPED in statuses:
        return CampaignStatus.FAILED
    if WorkStatus.INTERRUPTED in statuses:
        return CampaignStatus.INTERRUPTED
    if WorkStatus.RUNNING in statuses:
        return CampaignStatus.RUNNING
    return CampaignStatus.PLANNED


def derive_activity_status(
    calibration_status: CampaignStatus | str,
    dev_status: CampaignStatus | str,
    stability_status: CampaignStatus | str,
    *,
    stop_reason: CampaignStopReason | str | None = None,
    billing_uncertain: bool = False,
) -> CampaignStatus:
    """Derive the publication-level status from the three phase statuses."""

    statuses = [
        value if isinstance(value, CampaignStatus) else CampaignStatus(value)
        for value in (calibration_status, dev_status, stability_status)
    ]
    reasons: list[CampaignStopReason | str | None] = [stop_reason]
    reasons.extend(_STATUS_STOP.get(status) for status in statuses)
    if billing_uncertain:
        reasons.append(CampaignStopReason.BILLING_UNCERTAIN)
    if reason := highest_stop_reason(reasons):
        return _status_for_stop_reason(reason)
    if all(status is CampaignStatus.COMPLETE for status in statuses):
        return CampaignStatus.COMPLETE
    if CampaignStatus.RUNNING in statuses:
        return CampaignStatus.RUNNING
    if all(status is CampaignStatus.PLANNED for status in statuses):
        return CampaignStatus.PLANNED
    # A phase can be complete while a later phase has not been created yet.
    return CampaignStatus.PLANNED


def derive_run_id(
    campaign_id: str,
    phase: str,
    claim_id: str,
    strategy: Strategy | str,
    repeat: int = 0,
) -> str:
    """Derive the stable run ID defined by the Gate A protocol."""

    if not campaign_id or not phase or not claim_id:
        raise ValueError("campaign, phase, and claim IDs must be non-empty")
    if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 0:
        raise ValueError("repeat must be a non-negative integer")
    if phase not in {"dev", "stability"}:
        raise ValueError("campaign phase must be dev or stability")
    if repeat > 2:
        raise ValueError("repeat must be between zero and two")
    try:
        strategy_value = (
            strategy.value if isinstance(strategy, Strategy) else Strategy(strategy).value
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported strategy: {strategy!r}") from exc
    payload = f"{campaign_id}\0{phase}\0{claim_id}\0{strategy_value}\0{repeat}".encode()
    return hashlib.sha256(payload).hexdigest()


def derive_case_id(activity_id: str, claim_id: str) -> str:
    """Derive the calibration case identity used by the saved train artifacts."""

    if not activity_id or not claim_id:
        raise ValueError("activity and claim IDs must be non-empty")
    return hashlib.sha256(f"{activity_id}\0calibration\0{claim_id}".encode()).hexdigest()


def compute_gate_a_call_profile(
    calibration_claims: int = 32,
    dev_claims: int = 80,
    stability_claims: int = 20,
    stability_extra_repeats: int = 2,
    *,
    include_multi_recovery: bool = False,
) -> CallProfile:
    """Return the calculated Gate A worst-case node call profile."""

    values = (calibration_claims, dev_claims, stability_claims, stability_extra_repeats)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError("call profile counts must be non-negative integers")
    if not isinstance(include_multi_recovery, bool):
        raise ValueError("include_multi_recovery must be a boolean")
    recovery_single = int(include_multi_recovery)
    calibration = CallProfile(
        router=calibration_claims,
        single=calibration_claims,
        decomposer=calibration_claims,
        worker=3 * calibration_claims,
        judge=calibration_claims,
    )
    dev = CallProfile(
        router=dev_claims,
        single=(2 + recovery_single) * dev_claims,
        decomposer=2 * dev_claims,
        worker=6 * dev_claims,
        judge=2 * dev_claims,
    )
    stability = CallProfile(
        router=stability_claims * stability_extra_repeats,
        single=(1 + recovery_single) * stability_claims * stability_extra_repeats,
        decomposer=stability_claims * stability_extra_repeats,
        worker=3 * stability_claims * stability_extra_repeats,
        judge=stability_claims * stability_extra_repeats,
    )
    return calibration + dev + stability


def _jsonable(value: object) -> object:
    if isinstance(value, StrictModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def _without_digest(value: object, field: str) -> object:
    payload = _jsonable(value)
    if not isinstance(payload, dict):
        raise TypeError("fingerprint input must be a model or mapping")
    payload = copy.deepcopy(payload)
    payload.pop(field, None)
    return payload


def campaign_fingerprint(plan: CampaignPlan | Mapping[str, object]) -> str:
    """Hash the complete persisted campaign plan, excluding its self-referential digest."""

    return stable_hash(_without_digest(plan, "campaign_fingerprint"))


def plan_fingerprint(plan: CampaignPlan | Mapping[str, object]) -> str:
    """Alias used by callers that treat the campaign plan as the activity plan."""

    return campaign_fingerprint(plan)


def verify_campaign_fingerprint(plan: CampaignPlan) -> None:
    expected = campaign_fingerprint(plan)
    if plan.campaign_fingerprint != expected:
        raise FreezeMismatch(
            "campaign_fingerprint", expected=expected, actual=plan.campaign_fingerprint
        )


def state_fingerprint(state: CampaignState | Mapping[str, object]) -> str:
    return stable_hash(_without_digest(state, "state_sha256"))


def verify_state_fingerprint(
    state: CampaignState | Mapping[str, object], expected: str | None = None
) -> None:
    actual = state_fingerprint(state)
    if expected is None:
        expected = getattr(state, "state_sha256", None)
    if expected is None or expected != actual:
        raise FreezeMismatch("state_sha256", expected=actual, actual=expected)


def artifact_fingerprint(artifact: RunArtifact | Mapping[str, object]) -> str:
    return stable_hash(_without_digest(artifact, "artifact_sha256"))


def verify_artifact_fingerprint(artifact: RunArtifact | Mapping[str, object]) -> None:
    expected = artifact_fingerprint(artifact)
    actual = getattr(artifact, "artifact_sha256", None)
    if actual is None and isinstance(artifact, Mapping):
        actual = artifact.get("artifact_sha256")
    if actual != expected:
        raise FreezeMismatch("artifact_sha256", expected=expected, actual=actual)


def compare_freeze_identity(expected: FreezeIdentity, actual: FreezeIdentity) -> None:
    """Raise ``FreezeMismatch`` for the first field that differs in declaration order."""

    for field in FreezeIdentity.model_fields:
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if expected_value != actual_value:
            raise FreezeMismatch(field, expected=expected_value, actual=actual_value)


def verify_freeze_identity(expected: FreezeIdentity, actual: FreezeIdentity) -> None:
    compare_freeze_identity(expected, actual)


def summarize_run_artifacts(artifacts: Sequence[RunArtifact]) -> RunSummary:
    """Aggregate immutable artifact accounting without trusting an incremental counter."""

    if not artifacts:
        raise ValueError("run summary requires at least one artifact")
    run_ids: list[str] = []
    call_ids: list[str] = []
    usage_sources: list[Literal["provider", "missing"]] = []
    aliases: list[str] = []
    model_ids: list[str] = []
    usage_input = usage_output = usage_total = 0
    usage_complete = True
    actual_exact = True
    actual_total = known_total = committed_total = 0
    fresh_calls = cache_hits = transport_attempts = 0
    billing_uncertain = False
    for artifact in artifacts:
        run_ids.append(artifact.run_id)
        call_ids.extend(artifact.call_ids)
        if artifact.usage_source in {"provider", "missing"}:
            usage_sources.append(artifact.usage_source)
        usage_input += artifact.usage.input_tokens
        usage_output += artifact.usage.output_tokens
        usage_total += artifact.usage.total_tokens
        usage_complete = usage_complete and artifact.usage.complete
        known_total += artifact.known_actual_cost_micro_cny
        committed_total += artifact.committed_cost_micro_cny
        if artifact.actual_cost_micro_cny is None:
            actual_exact = False
        else:
            actual_total += artifact.actual_cost_micro_cny
        fresh_calls += artifact.fresh_call_count
        cache_hits += artifact.cache_hit_count
        billing_uncertain = billing_uncertain or artifact.billing_uncertain
        if artifact.requested_alias not in aliases:
            aliases.append(artifact.requested_alias)
        for model_id in artifact.response_model_ids_raw:
            if model_id and model_id not in model_ids:
                model_ids.append(model_id)
    usage = Usage(
        input_tokens=usage_input,
        output_tokens=usage_output,
        total_tokens=usage_total,
        complete=usage_complete,
    )
    return RunSummary(
        run_ids=run_ids,
        call_ids=call_ids,
        usage=usage,
        usage_sources=usage_sources,
        actual_cost_micro_cny=actual_total if actual_exact else None,
        known_actual_cost_micro_cny=known_total,
        committed_cost_micro_cny=committed_total,
        cost_is_lower_bound=not actual_exact,
        fresh_call_count=fresh_calls,
        cache_hit_count=cache_hits,
        transport_attempts=transport_attempts,
        requested_aliases=aliases,
        response_model_ids_raw=model_ids,
        billing_uncertain=billing_uncertain,
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "ActivityRecord",
    "CallBounds",
    "CallProfile",
    "CalibrationArtifactLink",
    "CampaignItemState",
    "CampaignPlan",
    "CampaignState",
    "CampaignStatus",
    "CampaignStopReason",
    "CampaignWorkItem",
    "FreezeIdentity",
    "FreezeMismatch",
    "LatencyBreakdown",
    "RunArtifact",
    "RunSummary",
    "StabilityBaselineLink",
    "WorkStatus",
    "artifact_fingerprint",
    "campaign_fingerprint",
    "compare_freeze_identity",
    "compute_gate_a_call_profile",
    "derive_activity_status",
    "derive_case_id",
    "derive_campaign_status",
    "derive_run_id",
    "highest_stop_reason",
    "plan_fingerprint",
    "result_status_to_work_status",
    "state_fingerprint",
    "summarize_run_artifacts",
    "verify_artifact_fingerprint",
    "verify_campaign_fingerprint",
    "verify_freeze_identity",
    "verify_state_fingerprint",
]
