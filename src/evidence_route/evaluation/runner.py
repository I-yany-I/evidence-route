"""Deterministic, resumable execution for Gate A evaluation campaigns.

The runner intentionally keeps transport concerns outside this module.  An injected async
executor receives one immutable :class:`CampaignWorkItem` and returns a validated
``RunArtifact``.  The runner owns ordering, persistence, identity checks and safety stops.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from evidence_route.artifacts import (
    BillingStateError,
    CallState,
    SQLiteRunStore,
    atomic_write_json,
)
from evidence_route.budget import BudgetExceeded, PriceConfig, UsageUnavailable
from evidence_route.config import BudgetSettings, GenerationSettings, HardeningSettings
from evidence_route.contracts import ResultStatus, Strategy, Usage, VerificationResult
from evidence_route.evaluation.activity import (
    CampaignItemState,
    CampaignPlan,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    CampaignWorkItem,
    FreezeIdentity,
    FreezeMismatch,
    RunArtifact,
    WorkStatus,
    artifact_fingerprint,
    compare_freeze_identity,
    derive_campaign_status,
    derive_run_id,
    highest_stop_reason,
    result_status_to_work_status,
    verify_artifact_fingerprint,
    verify_campaign_fingerprint,
)
from evidence_route.llm import BillingUncertain

Executor = Callable[[CampaignWorkItem], Awaitable[RunArtifact | Mapping[str, Any]]]
StartupEstimator = Callable[[Sequence[CampaignWorkItem]], int]


class CampaignProcessInterruption(RuntimeError):
    """Typed signal for a process-level interruption that is safe to resume."""


def compute_gate_a_call_profile(
    calibration_claims: int = 32,
    dev_claims: int = 80,
    stability_claims: int = 20,
    stability_extra_repeats: int = 2,
    *,
    include_multi_recovery: bool = False,
):
    """Compute the Gate A worst-case node calls.

    The canonical model lives in ``activity``; this forwarding function keeps the runner's
    public API stable for callers that do not otherwise import activity contracts.
    """

    from evidence_route.evaluation.activity import compute_gate_a_call_profile as _compute

    return _compute(
        calibration_claims,
        dev_claims,
        stability_claims,
        stability_extra_repeats,
        include_multi_recovery=include_multi_recovery,
    )


def estimate_call_bounds(
    profile,
    generation: GenerationSettings,
    pricing: PriceConfig,
    *,
    reserve_ratio: float = 0.2,
):
    """Estimate integer worst-case costs and retry/fault call bounds.

    Costs are calculated from the configured per-node token caps, using the same ceiling rule as
    ``SQLiteRunStore``.  This makes the estimate suitable for a startup cap check, not merely a
    display approximation.
    """

    from evidence_route.evaluation.activity import CallBounds

    if pricing.currency != "CNY":
        raise ValueError("Gate A pricing currency must be CNY")
    if (
        not pricing.strict_evaluation
        or pricing.input_per_million is None
        or pricing.output_per_million is None
    ):
        raise ValueError("strict evaluation pricing with input/output rates is required")
    if not isinstance(reserve_ratio, (int, float)) or not 0 <= reserve_ratio <= 1:
        raise ValueError("reserve_ratio must be between zero and one")
    reserve_basis_points = int(Decimal(str(reserve_ratio)) * Decimal(10_000))
    limits = {
        "router": generation.router,
        "single": generation.single,
        "decomposer": generation.decomposer,
        "worker": generation.worker,
        "judge": generation.judge,
    }
    counts = profile.model_dump()
    base_cost = 0
    for node, count in counts.items():
        limit = limits[node]
        usage = Usage(
            input_tokens=limit.max_input_tokens,
            output_tokens=limit.max_output_tokens,
            total_tokens=limit.max_input_tokens + limit.max_output_tokens,
            complete=True,
        )
        base_cost += count * pricing.estimate_micro_cny(usage)
    repair = base_cost * 2
    fault = repair * 3
    startup = (base_cost * (10_000 + reserve_basis_points) + 9_999) // 10_000
    return CallBounds(
        base_call_upper_bound=profile.total,
        repair_upper_bound=profile.total * 2,
        fault_upper_bound=profile.total * 2 * 3,
        base_cost_micro_cny=base_cost,
        repair_cost_micro_cny=repair,
        fault_cost_micro_cny=fault,
        reserve_basis_points=reserve_basis_points,
        startup_required_micro_cny=startup,
    )


def build_campaign_budget_preview(
    *,
    generation: GenerationSettings,
    hardening: HardeningSettings,
    budget: BudgetSettings,
    pricing: PriceConfig,
) -> dict[str, object]:
    """Build the deterministic paid-campaign startup preview."""

    bounds = estimate_call_bounds(
        compute_gate_a_call_profile(
            include_multi_recovery=hardening.multi_single_recovery
        ),
        generation,
        pricing,
        reserve_ratio=budget.reserve_ratio,
    )
    return {
        **bounds.model_dump(mode="json"),
        "cap_micro_cny": int(budget.estimated_cost_cap_cny * 1_000_000),
        "paid_execution_started": False,
    }


def _claim_id(claim: object) -> str:
    if isinstance(claim, str):
        return claim
    if isinstance(claim, Mapping):
        value = claim.get("claim_id")
    else:
        value = getattr(claim, "claim_id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("runtime claim must expose a non-empty claim_id")
    return value


def build_dev_schedule(
    runtime_claims: Iterable[object],
    *,
    seed: int,
    campaign_id: str = "campaign",
) -> list[CampaignWorkItem]:
    """Build the deterministic three-strategy cyclic dev schedule."""

    claims = list(runtime_claims)
    strategies = list(Strategy)
    base = sorted(
        strategies,
        key=lambda strategy: hashlib.sha256(f"{seed}\0{strategy.value}".encode()).digest(),
    )
    schedule: list[CampaignWorkItem] = []
    for claim_index, claim in enumerate(claims):
        claim_id = _claim_id(claim)
        rotated = base[claim_index % 3 :] + base[: claim_index % 3]
        for strategy in rotated:
            order = len(schedule)
            schedule.append(
                CampaignWorkItem(
                    order=order,
                    phase="dev",
                    claim_id=claim_id,
                    strategy=strategy,
                    repeat=0,
                    run_id=derive_run_id(campaign_id, "dev", claim_id, strategy, 0),
                )
            )
    return schedule


def build_campaign_schedule(
    runtime_claims: Iterable[object],
    *,
    stability_runtime_claims: Iterable[object],
    seed: int,
    campaign_id: str = "campaign",
) -> tuple[list[CampaignWorkItem], list[dict[str, str]]]:
    """Build dev items plus adaptive stability repeats and their immutable links."""

    claims = list(runtime_claims)
    stability_claims = list(stability_runtime_claims)
    if len(stability_claims) != 20:
        raise ValueError("stability runtime manifest must contain exactly 20 claims")
    dev_claim_ids = {_claim_id(claim) for claim in claims}
    stability_claim_ids = [_claim_id(claim) for claim in stability_claims]
    if len(stability_claim_ids) != len(set(stability_claim_ids)):
        raise ValueError("stability runtime manifest claim IDs must be unique")
    if not set(stability_claim_ids).issubset(dev_claim_ids):
        raise ValueError("stability runtime manifest must be a subset of the dev manifest")

    dev = build_dev_schedule(claims, seed=seed, campaign_id=campaign_id)
    links: list[dict[str, str]] = []
    schedule = list(dev)
    for claim_id in stability_claim_ids:
        adaptive = next(
            item for item in dev if item.claim_id == claim_id and item.strategy is Strategy.ADAPTIVE
        )
        links.append({"claim_id": claim_id, "dev_adaptive_run_id": adaptive.run_id})
        for repeat in (1, 2):
            schedule.append(
                CampaignWorkItem(
                    order=len(schedule),
                    phase="stability",
                    claim_id=claim_id,
                    strategy=Strategy.ADAPTIVE,
                    repeat=repeat,
                    run_id=derive_run_id(
                        campaign_id, "stability", claim_id, Strategy.ADAPTIVE, repeat
                    ),
                )
            )
    return schedule, links


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _model_ids(artifact: RunArtifact) -> set[str]:
    return {value.strip() for value in artifact.response_model_ids_raw if value.strip()}


class CampaignRunner:
    """Run and resume an immutable campaign plan."""

    def __init__(
        self,
        root: Path,
        executor: Executor,
        *,
        run_store: SQLiteRunStore | None = None,
        startup_estimator: StartupEstimator | None = None,
    ) -> None:
        self.root = Path(root)
        self.executor = executor
        self.run_store = run_store or getattr(executor, "run_store", None)
        self.startup_estimator = startup_estimator
        self.plan_path = self.root / "plan.json"
        self.state_path = self.root / "campaign.json"
        self.artifact_dir = self.root / "artifacts"

    def _write_plan(self, plan: CampaignPlan) -> None:
        verify_campaign_fingerprint(plan)
        atomic_write_json(self.plan_path, plan.model_dump(mode="json"))

    def _new_state(
        self,
        plan: CampaignPlan,
        *,
        initial_model_ids: Iterable[str] = (),
    ) -> CampaignState:
        model_ids = sorted({value.strip() for value in initial_model_ids if value.strip()})
        return CampaignState(
            schema_version="1",
            activity_id=plan.activity_id,
            campaign_id=plan.campaign_id,
            status=CampaignStatus.PLANNED,
            stop_reason=None,
            items=[
                CampaignItemState(
                    run_id=item.run_id,
                    artifact_relpath=f"artifacts/{item.run_id}.json",
                )
                for item in plan.schedule
            ],
            stability_repeat_zero_artifact_sha256s={},
            observed_response_model_ids_raw=model_ids,
            billing_uncertain=False,
            summary=None,
        )

    def _write_state(self, state: CampaignState) -> None:
        atomic_write_json(self.state_path, state.model_dump(mode="json"))

    @staticmethod
    def _validate_max_items(max_items: int | None) -> None:
        if max_items is not None and (
            not isinstance(max_items, int) or isinstance(max_items, bool) or max_items <= 0
        ):
            raise ValueError("max_items must be a positive integer")

    def _verify_state_items(self, plan: CampaignPlan, state: CampaignState) -> None:
        if len(state.items) != len(plan.schedule):
            raise ValueError("campaign state items do not match persisted plan length")
        for index, (work, item) in enumerate(zip(plan.schedule, state.items, strict=True)):
            expected_relpath = f"artifacts/{work.run_id}.json"
            if item.run_id != work.run_id or item.artifact_relpath != expected_relpath:
                raise ValueError(
                    f"campaign state items do not match persisted plan at index {index}"
                )

    def _verify_run_store_identity(self, plan: CampaignPlan) -> None:
        if self.run_store is None:
            return
        if self.run_store.activity_id != plan.activity_id:
            raise FreezeMismatch(
                "run_store.activity_id",
                expected=plan.activity_id,
                actual=self.run_store.activity_id,
            )
        if self.run_store.cap_micro_cny != plan.cap_micro_cny:
            raise FreezeMismatch(
                "run_store.cap_micro_cny",
                expected=plan.cap_micro_cny,
                actual=self.run_store.cap_micro_cny,
            )

    @staticmethod
    def _pending_work_items(
        plan: CampaignPlan, state: CampaignState
    ) -> list[CampaignWorkItem]:
        terminal = {
            WorkStatus.COMPLETED,
            WorkStatus.PARTIAL,
            WorkStatus.FAILED,
            WorkStatus.NOT_RUN_BUDGET,
            WorkStatus.CANCELLED,
            WorkStatus.STOPPED,
        }
        return [
            work
            for work, item in zip(plan.schedule, state.items, strict=True)
            if item.status not in terminal
        ]

    def _validate_startup_reservation(
        self,
        plan: CampaignPlan,
        state: CampaignState,
        *,
        max_items: int | None,
    ) -> None:
        pending = self._pending_work_items(plan, state)
        if not pending:
            return
        selected = pending if max_items is None else pending[:max_items]
        if self.startup_estimator is None:
            required = plan.call_bounds.startup_required_micro_cny
        else:
            required = self.startup_estimator(selected)
            if not isinstance(required, int) or isinstance(required, bool) or required < 0:
                raise ValueError("startup estimator must return a non-negative integer")
        committed = (
            self.run_store.summarize_activity().committed_cost_micro_cny
            if self.run_store is not None
            else 0
        )
        if committed + required > plan.cap_micro_cny:
            raise BudgetExceeded("startup worst-case batch reservation exceeds campaign cap")

    @staticmethod
    def _verify_artifact_identity(
        plan: CampaignPlan,
        work: CampaignWorkItem,
        artifact: RunArtifact,
    ) -> None:
        expected = (
            plan.activity_id,
            plan.campaign_id,
            work.phase,
            work.run_id,
            work.claim_id,
            work.strategy,
            work.repeat,
            plan.freeze.requested_alias,
        )
        actual = (
            artifact.activity_id,
            artifact.campaign_id,
            artifact.phase,
            artifact.run_id,
            artifact.claim_id,
            artifact.strategy,
            artifact.repeat,
            artifact.requested_alias,
        )
        if actual != expected:
            raise ValueError(f"artifact identity does not match campaign plan for {work.run_id}")

    @staticmethod
    def _require_accounting_match(field: str, expected: object, actual: object) -> None:
        if actual != expected:
            raise FreezeMismatch(
                f"run_store.{field}",
                expected=expected,
                actual=actual,
            )

    def _verify_run_store_accounting(self, artifact: RunArtifact) -> None:
        if self.run_store is None:
            return
        summary = self.run_store.summarize_run(artifact.run_id)
        unresolved = self.run_store.unresolved_call_states(artifact.run_id)
        if unresolved and not artifact.diagnostic_only:
            if any(
                state in {CallState.SENT, CallState.BILLING_UNCERTAIN}
                for state in unresolved.values()
            ):
                raise BillingStateError(f"run {artifact.run_id} has unresolved transmitted calls")
            raise FreezeMismatch(
                "run_store.unresolved_call_state",
                expected="all calls completed",
                actual={call_id: state.value for call_id, state in unresolved.items()},
            )
        if not summary.call_ids:
            if artifact.call_ids:
                self._require_accounting_match("call_ids", [], artifact.call_ids)
            zero_accounting = (
                artifact.usage.input_tokens == 0
                and artifact.usage.output_tokens == 0
                and artifact.usage.total_tokens == 0
                and artifact.actual_cost_micro_cny in {None, 0}
                and artifact.known_actual_cost_micro_cny == 0
                and artifact.committed_cost_micro_cny == 0
                and artifact.fresh_call_count == 0
            )
            if not zero_accounting:
                raise FreezeMismatch(
                    "run_store.zero_call_accounting",
                    expected="zero accounting",
                    actual=artifact.model_dump(mode="json"),
                )
            return
        expected_usage = summary.usage
        if "missing" in summary.usage_sources or not summary.usage.complete:
            expected_usage_source = "missing"
        else:
            expected_usage_source = "provider"

        comparisons = {
            "call_ids": (summary.call_ids, artifact.call_ids),
            "usage": (
                expected_usage.model_dump(mode="json"),
                artifact.usage.model_dump(mode="json"),
            ),
            "usage_source": (expected_usage_source, artifact.usage_source),
            "actual_cost_micro_cny": (
                summary.actual_cost_micro_cny,
                artifact.actual_cost_micro_cny,
            ),
            "known_actual_cost_micro_cny": (
                summary.known_actual_cost_micro_cny,
                artifact.known_actual_cost_micro_cny,
            ),
            "committed_cost_micro_cny": (
                summary.committed_cost_micro_cny,
                artifact.committed_cost_micro_cny,
            ),
            "cost_is_lower_bound": (
                summary.cost_is_lower_bound,
                artifact.cost_is_lower_bound,
            ),
            "fresh_call_count": (summary.fresh_call_count, artifact.fresh_call_count),
            "cache_hit_count": (summary.cache_hit_count, artifact.cache_hit_count),
            "response_model_ids_raw": (
                summary.response_model_ids_raw,
                artifact.response_model_ids_raw,
            ),
            "billing_uncertain": (
                summary.billing_uncertain,
                artifact.billing_uncertain,
            ),
        }
        for field, (expected, actual) in comparisons.items():
            self._require_accounting_match(field, expected, actual)
        if summary.requested_aliases:
            self._require_accounting_match(
                "requested_aliases", [artifact.requested_alias], summary.requested_aliases
            )

    def _verify_persisted_artifacts(
        self,
        plan: CampaignPlan,
        state: CampaignState,
        *,
        allow_authorized_billing_recovery: set[str] | None = None,
    ) -> None:
        for work, item in zip(plan.schedule, state.items, strict=True):
            path = self.root / item.artifact_relpath
            if item.artifact_sha256 is None:
                if path.is_file():
                    raise FreezeMismatch(
                        "unlinked_artifact",
                        expected=None,
                        actual=str(path),
                    )
                continue
            if not path.is_file():
                raise FileNotFoundError(f"missing artifact for run {work.run_id}: {path}")
            artifact = RunArtifact.model_validate_json(path.read_text(encoding="utf-8"))
            verify_artifact_fingerprint(artifact)
            if artifact.artifact_sha256 != item.artifact_sha256:
                raise FreezeMismatch(
                    "artifact_sha256",
                    expected=artifact.artifact_sha256,
                    actual=item.artifact_sha256,
                )
            self._verify_artifact_identity(plan, work, artifact)
            if not (
                allow_authorized_billing_recovery
                and item.stop_reason is CampaignStopReason.BILLING_UNCERTAIN
                and artifact.diagnostic_only
            ):
                self._verify_run_store_accounting(artifact)

    def _refresh_summary(self, state: CampaignState) -> None:
        """Recompute accounting from immutable artifact files, never from counters."""

        from evidence_route.evaluation.activity import summarize_run_artifacts

        artifacts: list[RunArtifact] = []
        for item in state.items:
            if item.artifact_sha256 is None:
                continue
            path = self.root / item.artifact_relpath
            if path.is_file():
                artifact = RunArtifact.model_validate_json(path.read_text(encoding="utf-8"))
                verify_artifact_fingerprint(artifact)
                artifacts.append(artifact)
        if artifacts:
            state.summary = summarize_run_artifacts(artifacts)

    def _load_plan(self) -> CampaignPlan:
        if not self.plan_path.is_file():
            raise FileNotFoundError(self.plan_path)
        plan = CampaignPlan.model_validate_json(self.plan_path.read_text(encoding="utf-8"))
        verify_campaign_fingerprint(plan)
        return plan

    def _load_state(self) -> CampaignState:
        if not self.state_path.is_file():
            raise FileNotFoundError(self.state_path)
        return CampaignState.model_validate_json(self.state_path.read_text(encoding="utf-8"))

    async def run(
        self,
        plan: CampaignPlan,
        *,
        initial_model_ids: Iterable[str] = (),
        max_items: int | None = None,
    ) -> CampaignState:
        self._validate_max_items(max_items)
        plan_exists = self.plan_path.exists()
        state_exists = self.state_path.exists()
        if plan_exists != state_exists:
            raise FileExistsError("campaign directory contains an incomplete campaign journal")
        if plan_exists:
            raise FileExistsError("campaign directory already contains a persisted plan")
        self._verify_run_store_identity(plan)
        state = self._new_state(plan, initial_model_ids=initial_model_ids)
        self._validate_startup_reservation(plan, state, max_items=max_items)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._write_plan(plan)
        self._write_state(state)
        return await self._execute(plan, state, max_items=max_items)

    async def resume(
        self,
        *,
        expected_identity: FreezeIdentity | None = None,
        max_items: int | None = None,
        authorized_call_ids: set[str] | None = None,
        recovery_run_ids: set[str] | None = None,
    ) -> CampaignState:
        self._validate_max_items(max_items)
        plan = self._load_plan()
        self._verify_run_store_identity(plan)
        if expected_identity is not None:
            compare_freeze_identity(plan.freeze, expected_identity)
        state = self._load_state()
        if state.activity_id != plan.activity_id or state.campaign_id != plan.campaign_id:
            raise ValueError("campaign state does not match persisted plan")
        self._verify_state_items(plan, state)
        persisted_reason = highest_stop_reason(
            [state.stop_reason, *(item.stop_reason for item in state.items)]
        )
        recovery_run_ids = set(recovery_run_ids or set())
        recovery_requested = (
            persisted_reason
            in {
                CampaignStopReason.BILLING_UNCERTAIN,
                CampaignStopReason.USAGE_MISSING,
                CampaignStopReason.INTERNAL_ERROR,
            }
            and bool(authorized_call_ids or recovery_run_ids)
        )
        self._verify_persisted_artifacts(
            plan,
            state,
            allow_authorized_billing_recovery=(
                authorized_call_ids if recovery_requested and authorized_call_ids else None
            ),
        )
        requeued_billing_recovery = (
            recovery_requested
            and self._requeue_authorized_billing_recovery(
                plan,
                state,
                authorized_call_ids or set(),
                recovery_run_ids,
            )
        )
        if requeued_billing_recovery:
            self._validate_startup_reservation(plan, state, max_items=max_items)
            return await self._execute(plan, state, max_items=max_items)
        if (
            persisted_reason is not None
            and persisted_reason
            not in {CampaignStopReason.PROCESS_INTERRUPTION, CampaignStopReason.USER_PAUSED}
        ):
            normalized_status = derive_campaign_status(state.items, stop_reason=persisted_reason)
            normalized_billing = persisted_reason is CampaignStopReason.BILLING_UNCERTAIN
            if (
                state.stop_reason is not persisted_reason
                or state.status is not normalized_status
                or state.billing_uncertain != normalized_billing
            ):
                state.stop_reason = persisted_reason
                state.status = normalized_status
                state.billing_uncertain = normalized_billing
                self._write_state(state)
            return state
        running_items = [item for item in state.items if item.status is WorkStatus.RUNNING]
        if len(running_items) > 1:
            raise ValueError("campaign state is corrupt: multiple running items")
        # A process can die after marking RUNNING.  Normalize exactly one such item and retain
        # interruption timestamps as audit history before allowing another transport call.
        changed = False
        running_item = next(
            (item for item in state.items if item.status is WorkStatus.RUNNING),
            None,
        )
        if running_item is not None:
            running_item.status = WorkStatus.INTERRUPTED
            running_item.stop_reason = CampaignStopReason.PROCESS_INTERRUPTION
            running_item.interrupted_at = running_item.interrupted_at or _now()
            changed = True
        elif state.stop_reason in {
            CampaignStopReason.PROCESS_INTERRUPTION,
            CampaignStopReason.USER_PAUSED,
        }:
            if state.stop_reason is CampaignStopReason.USER_PAUSED:
                state.stop_reason = None
                state.status = CampaignStatus.RUNNING
                self._write_state(state)
                self._validate_startup_reservation(plan, state, max_items=max_items)
                return await self._execute(plan, state, max_items=max_items)
            resumable = [
                item
                for item in state.items
                if item.status is WorkStatus.INTERRUPTED
                and item.stop_reason is CampaignStopReason.PROCESS_INTERRUPTION
            ]
            if not resumable:
                raise ValueError("campaign state has process interruption without a resumable item")
            if len(resumable) > 1:
                timestamps: list[datetime] = []
                for item in resumable:
                    if item.interrupted_at is None:
                        raise ValueError("campaign state has multiple resumable interruptions")
                    try:
                        timestamp = datetime.fromisoformat(item.interrupted_at)
                    except ValueError as exc:
                        raise ValueError(
                            "campaign state has an invalid interruption timestamp"
                        ) from exc
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=UTC)
                    timestamps.append(timestamp)
                latest = max(timestamps)
                latest_items = [
                    item
                    for item, timestamp in zip(resumable, timestamps, strict=True)
                    if timestamp == latest
                ]
                if len(latest_items) != 1:
                    raise ValueError("campaign state has multiple resumable interruptions")
                running_item = latest_items[0]
            else:
                running_item = resumable[0]
        if changed:
            state.stop_reason = CampaignStopReason.PROCESS_INTERRUPTION
            state.status = CampaignStatus.INTERRUPTED
            self._write_state(state)
        if running_item is not None and self.run_store is not None:
            unresolved = self.run_store.unresolved_call_states(running_item.run_id)
            for call_id, call_state in unresolved.items():
                if call_state is CallState.SENT:
                    self.run_store.mark_billing_uncertain(call_id)
            unresolved = self.run_store.unresolved_call_states(running_item.run_id)
            unresolved_states = set(unresolved.values())
            reason: CampaignStopReason | None = None
            if unresolved_states & {CallState.SENT, CallState.BILLING_UNCERTAIN}:
                reason = CampaignStopReason.BILLING_UNCERTAIN
            elif CallState.USAGE_MISSING in unresolved_states:
                reason = CampaignStopReason.USAGE_MISSING
            if reason is not None:
                artifact = self._diagnostic_artifact(
                    plan,
                    plan.schedule[state.items.index(running_item)],
                    reason,
                    "interrupted run has unresolved persisted call accounting",
                )
                return self._stop_with_artifact(
                    plan,
                    state,
                    state.items.index(running_item),
                    artifact,
                    reason,
                )
        if running_item is not None:
            running_item.status = WorkStatus.PENDING
            running_item.stop_reason = None
            running_item.resumed_at = _now()
            state.stop_reason = None
            state.status = CampaignStatus.RUNNING
            self._write_state(state)
        self._validate_startup_reservation(plan, state, max_items=max_items)
        return await self._execute(plan, state, max_items=max_items)

    async def resume_after_billing_recovery(
        self,
        *,
        expected_identity: FreezeIdentity | None = None,
        max_items: int | None = None,
        authorized_call_ids: set[str],
        recovery_run_ids: set[str] | None = None,
    ) -> CampaignState:
        """Resume only after explicit authorization for every unresolved paid call."""

        if not authorized_call_ids and not recovery_run_ids:
            raise ValueError("billing recovery requires an authorized call or recovery run")
        return await self.resume(
            expected_identity=expected_identity,
            max_items=max_items,
            authorized_call_ids=authorized_call_ids,
            recovery_run_ids=recovery_run_ids,
        )

    def _requeue_authorized_billing_recovery(
        self,
        plan: CampaignPlan,
        state: CampaignState,
        authorized_call_ids: set[str],
        recovery_run_ids: set[str],
    ) -> bool:
        if self.run_store is None:
            return False
        recovery_items = [
            (index, item)
            for index, item in enumerate(state.items)
            if item.stop_reason
            in {
                CampaignStopReason.BILLING_UNCERTAIN,
                CampaignStopReason.USAGE_MISSING,
                CampaignStopReason.INTERNAL_ERROR,
            }
        ]
        if not recovery_items:
            return False
        for _index, item in recovery_items:
            unresolved = self.run_store.unresolved_call_states(item.run_id)
            if unresolved:
                if any(
                    call_state is not CallState.RESERVED
                    or call_id not in authorized_call_ids
                    or self.run_store.get_billing_recovery_event(call_id) is None
                    for call_id, call_state in unresolved.items()
                ):
                    return False
            elif (
                item.stop_reason is not CampaignStopReason.INTERNAL_ERROR
                or item.run_id not in recovery_run_ids
            ):
                return False
            item.status = WorkStatus.PENDING
            item.stop_reason = None
            item.artifact_sha256 = None
            item.resumed_at = _now()
        state.stop_reason = None
        state.billing_uncertain = False
        state.status = CampaignStatus.RUNNING
        self._write_state(state)
        return True

    async def _invoke(self, work: CampaignWorkItem) -> RunArtifact:
        result = self.executor(work)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, RunArtifact) else RunArtifact.model_validate(result)

    def _persist_artifact(
        self,
        plan: CampaignPlan,
        artifact: RunArtifact,
        work: CampaignWorkItem,
    ) -> str:
        if (
            artifact.run_id != work.run_id
            or artifact.claim_id != work.claim_id
            or artifact.strategy is not work.strategy
            or artifact.repeat != work.repeat
            or artifact.phase != work.phase
        ):
            raise ValueError("executor artifact does not match scheduled work item")
        self._verify_artifact_identity(plan, work, artifact)
        verify_artifact_fingerprint(artifact)
        self._verify_run_store_accounting(artifact)
        path = self.root / f"artifacts/{artifact.run_id}.json"
        atomic_write_json(path, artifact.model_dump(mode="json"))
        return artifact.artifact_sha256

    def _diagnostic_artifact(
        self,
        plan: CampaignPlan,
        work: CampaignWorkItem,
        reason: CampaignStopReason,
        detail: str,
    ) -> RunArtifact:
        call_ids: list[str] = []
        known_cost = committed_cost = cache_hits = 0
        requested_alias = plan.freeze.requested_alias
        response_ids: list[str] = []
        usage_source = "missing"
        actual_cost: int | None = None
        cost_is_lower_bound = True
        billing_uncertain = reason is CampaignStopReason.BILLING_UNCERTAIN
        if self.run_store is not None:
            summary = self.run_store.summarize_run(work.run_id)
            if summary.call_ids:
                call_ids = summary.call_ids
                known_cost = summary.known_actual_cost_micro_cny
                committed_cost = summary.committed_cost_micro_cny
                cache_hits = summary.cache_hit_count
                response_ids = summary.response_model_ids_raw
                usage = summary.usage
                usage_source = "provider" if summary.usage.complete else "missing"
                actual_cost = summary.actual_cost_micro_cny
                cost_is_lower_bound = summary.cost_is_lower_bound
                billing_uncertain = billing_uncertain or summary.billing_uncertain
            else:
                usage = Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False)
        else:
            usage = Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False)
        result = VerificationResult(
            claim_id=work.claim_id,
            status=ResultStatus.FAILED,
            verdict=None,
            confidence=None,
            rationale="campaign stopped before accounting could be closed",
            citations=[],
            available_evidence_ids=[],
            initial_route="single",
            failure_stage="single",
            usage=usage,
            estimated_cost_micro_cny=actual_cost,
            cost_currency="CNY" if actual_cost is not None else None,
            price_config_id=(
                self.run_store.pricing.config_id
                if actual_cost is not None and self.run_store is not None
                else None
            ),
            errors=[reason.value.upper(), detail],
        )
        payload: dict[str, object] = {
            "schema_version": "1",
            "activity_id": plan.activity_id,
            "campaign_id": plan.campaign_id,
            "phase": work.phase,
            "run_id": work.run_id,
            "claim_id": work.claim_id,
            "strategy": work.strategy.value,
            "repeat": work.repeat,
            "result": result.model_dump(mode="json"),
            "call_ids": call_ids,
            "usage": usage.model_dump(mode="json"),
            "usage_source": usage_source,
            "actual_cost_micro_cny": actual_cost,
            "known_actual_cost_micro_cny": known_cost,
            "committed_cost_micro_cny": committed_cost,
            "cost_is_lower_bound": cost_is_lower_bound,
            "fresh_call_count": len(call_ids),
            "cache_hit_count": cache_hits,
            "requested_alias": requested_alias,
            "response_model_ids_raw": response_ids,
            "identity_verified": False,
            "billing_uncertain": billing_uncertain,
            "diagnostic_only": True,
            "latency": {
                "fresh_end_to_end_ms": 0,
                "model_active_ms": 0,
                "retry_ms": 0,
                "queue_ms": 0,
                "checkpoint_downtime_ms": 0,
                "total_elapsed_ms": 0,
                "interruption_count": 0,
            },
            "artifact_sha256": "0" * 64,
        }
        payload["artifact_sha256"] = artifact_fingerprint(payload)
        return RunArtifact.model_validate(payload)

    def _mark_budget(self, state: CampaignState, index: int) -> CampaignState:
        for item in state.items[index:]:
            if item.status in {WorkStatus.PENDING, WorkStatus.RUNNING, WorkStatus.INTERRUPTED}:
                item.status = WorkStatus.NOT_RUN_BUDGET
                item.stop_reason = CampaignStopReason.BUDGET
        state.stop_reason = CampaignStopReason.BUDGET
        state.status = CampaignStatus.INCOMPLETE_BUDGET
        self._write_state(state)
        return state

    def _verify_stability_baselines(self, plan: CampaignPlan, state: CampaignState) -> None:
        item_by_run_id = {item.run_id: item for item in state.items}
        for link in plan.stability_repeat_zero_links:
            dev_item = item_by_run_id[link.dev_adaptive_run_id]
            expected = dev_item.artifact_sha256
            actual = state.stability_repeat_zero_artifact_sha256s.get(link.claim_id)
            if expected is None or actual != expected:
                raise FreezeMismatch(
                    "stability_repeat_zero_artifact_sha256s",
                    expected=expected,
                    actual=actual,
                )
            artifact_path = self.root / dev_item.artifact_relpath
            artifact = RunArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
            verify_artifact_fingerprint(artifact)
            if artifact.artifact_sha256 != actual:
                raise FreezeMismatch(
                    "stability_repeat_zero_artifact_sha256s",
                    expected=artifact.artifact_sha256,
                    actual=actual,
                )

    def _stop_with_artifact(
        self,
        plan: CampaignPlan,
        state: CampaignState,
        index: int,
        artifact: RunArtifact,
        reason: CampaignStopReason,
    ) -> CampaignState:
        digest = self._persist_artifact(plan, artifact, plan.schedule[index])
        item = state.items[index]
        item.status = WorkStatus.STOPPED
        item.stop_reason = reason
        item.artifact_sha256 = digest
        state.stop_reason = reason
        state.billing_uncertain = reason is CampaignStopReason.BILLING_UNCERTAIN
        state.status = derive_campaign_status(state.items, stop_reason=reason)
        self._write_state(state)
        return state

    async def _execute(
        self,
        plan: CampaignPlan,
        state: CampaignState,
        *,
        max_items: int | None = None,
    ) -> CampaignState:
        state.status = CampaignStatus.RUNNING
        self._write_state(state)
        executed_items = 0
        baseline = set(state.observed_response_model_ids_raw)
        linked_claims = {link.claim_id for link in plan.stability_repeat_zero_links}
        stability_verified = False
        for index, work in enumerate(plan.schedule):
            item = state.items[index]
            if item.status in {
                WorkStatus.COMPLETED,
                WorkStatus.PARTIAL,
                WorkStatus.FAILED,
                WorkStatus.NOT_RUN_BUDGET,
                WorkStatus.CANCELLED,
                WorkStatus.STOPPED,
            }:
                continue
            if work.phase == "stability" and not stability_verified:
                self._verify_stability_baselines(plan, state)
                stability_verified = True
            item.status = WorkStatus.RUNNING
            item.stop_reason = None
            self._write_state(state)
            try:
                artifact = await self._invoke(work)
            except BudgetExceeded as exc:
                # A provider response may already have been committed to the run store before
                # the store surfaces an over-budget signal.  Preserve that accounting on the
                # current item; only work items that were never started are budget-skipped.
                if (
                    self.run_store is not None
                    and self.run_store.summarize_run(work.run_id).call_ids
                ):
                    artifact = self._diagnostic_artifact(
                        plan, work, CampaignStopReason.BUDGET, str(exc)
                    )
                    self._stop_with_artifact(
                        plan, state, index, artifact, CampaignStopReason.BUDGET
                    )
                    return self._mark_budget(state, index + 1)
                return self._mark_budget(state, index)
            except BillingStateError as exc:
                # An unresolved sent/billing-uncertain row cannot be safely retried.  Leave the
                # reservation committed and persist an explicit diagnostic artifact.
                artifact = self._diagnostic_artifact(
                    plan, work, CampaignStopReason.BILLING_UNCERTAIN, str(exc)
                )
                return self._stop_with_artifact(
                    plan, state, index, artifact, CampaignStopReason.BILLING_UNCERTAIN
                )
            except BillingUncertain as exc:
                artifact = self._diagnostic_artifact(
                    plan, work, CampaignStopReason.BILLING_UNCERTAIN, str(exc)
                )
                return self._stop_with_artifact(
                    plan, state, index, artifact, CampaignStopReason.BILLING_UNCERTAIN
                )
            except UsageUnavailable as exc:
                artifact = self._diagnostic_artifact(
                    plan, work, CampaignStopReason.USAGE_MISSING, str(exc)
                )
                return self._stop_with_artifact(
                    plan, state, index, artifact, CampaignStopReason.USAGE_MISSING
                )
            except (KeyboardInterrupt, CampaignProcessInterruption):
                item.status = WorkStatus.INTERRUPTED
                item.stop_reason = CampaignStopReason.PROCESS_INTERRUPTION
                item.interrupted_at = _now()
                state.stop_reason = CampaignStopReason.PROCESS_INTERRUPTION
                state.status = CampaignStatus.INTERRUPTED
                self._write_state(state)
                raise
            except Exception as exc:
                artifact = self._diagnostic_artifact(
                    plan, work, CampaignStopReason.INTERNAL_ERROR, str(exc)
                )
                self._stop_with_artifact(
                    plan, state, index, artifact, CampaignStopReason.INTERNAL_ERROR
                )
                raise

            # Identity is part of the frozen protocol, not an executor error that can be
            # converted into a diagnostic result.  A mismatch leaves the journal RUNNING so a
            # subsequent resume performs the same immutable audit before any retry.
            self._verify_artifact_identity(plan, work, artifact)
            if artifact.billing_uncertain:
                self._stop_with_artifact(
                    plan, state, index, artifact, CampaignStopReason.BILLING_UNCERTAIN
                )
                return state
            if artifact.usage_source == "missing" or not artifact.usage.complete:
                self._stop_with_artifact(
                    plan, state, index, artifact, CampaignStopReason.USAGE_MISSING
                )
                return state
            ids = _model_ids(artifact)
            if not baseline and ids:
                baseline = set(ids)
                state.observed_response_model_ids_raw = sorted(ids)
            elif len(ids) > 1 or (ids and ids != baseline):
                diagnostic_payload = artifact.model_dump(mode="json")
                diagnostic_payload["diagnostic_only"] = True
                diagnostic_payload["artifact_sha256"] = artifact_fingerprint(diagnostic_payload)
                diagnostic = RunArtifact.model_validate(diagnostic_payload)
                self._stop_with_artifact(
                    plan, state, index, diagnostic, CampaignStopReason.MODEL_DRIFT
                )
                return state
            digest = self._persist_artifact(plan, artifact, work)
            item.artifact_sha256 = digest
            item.status = result_status_to_work_status(artifact.result.status)
            if (
                work.phase == "dev"
                and work.strategy is Strategy.ADAPTIVE
                and work.claim_id in linked_claims
            ):
                state.stability_repeat_zero_artifact_sha256s[work.claim_id] = digest
            state.status = derive_campaign_status(state.items)
            self._refresh_summary(state)
            self._write_state(state)
            executed_items += 1
            if max_items is not None and executed_items >= max_items:
                remaining = any(item.status is WorkStatus.PENDING for item in state.items)
                if remaining:
                    state.stop_reason = CampaignStopReason.USER_PAUSED
                    state.status = CampaignStatus.PAUSED
                    self._write_state(state)
                    return state
        state.status = derive_campaign_status(state.items)
        self._refresh_summary(state)
        self._write_state(state)
        return state


__all__ = [
    "CampaignRunner",
    "CampaignProcessInterruption",
    "build_campaign_schedule",
    "build_campaign_budget_preview",
    "build_dev_schedule",
    "compute_gate_a_call_profile",
    "estimate_call_bounds",
]
