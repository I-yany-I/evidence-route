"""Paid Gate A campaign orchestration after calibration replay.

This module owns the boundary between the frozen calibration activity and the interleaved
dev/stability campaign.  It creates the campaign journal before constructing a transport and
uses the same SQLite call ledger as calibration.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evidence_route.artifacts import SQLiteRunStore
from evidence_route.budget import PriceConfig
from evidence_route.config import AppConfig, load_app_config, stable_hash
from evidence_route.contracts import Usage
from evidence_route.evaluation.activity import (
    ActivityRecord,
    CampaignPlan,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    CampaignWorkItem,
    FreezeIdentity,
    RunSummary,
    WorkStatus,
    compare_freeze_identity,
    derive_activity_status,
    derive_campaign_status,
    verify_campaign_fingerprint,
)
from evidence_route.evaluation.calibration import (
    CalibrationItemStatus,
    CalibrationRuntimeCase,
    load_calibration_cases,
    load_calibration_plan,
    load_calibration_state,
)
from evidence_route.evaluation.experiment import (
    create_experiment_identity,
    load_experiment_identity,
    materialize_calibration_assets,
    validate_experiment_target,
    validate_parent_baseline,
    verify_repeat_zero_reuse,
)
from evidence_route.evaluation.lifecycle import (
    build_freeze_identity,
    git_head,
    load_activity,
    persist_activity,
    resume_activity_phase,
    seal_activity,
    sha256_file,
    verify_current_freeze,
)
from evidence_route.evaluation.production import GraphCampaignExecutor, build_campaign_plan
from evidence_route.evaluation.runner import (
    CampaignRunner,
    compute_gate_a_call_profile,
    estimate_call_bounds,
)
from evidence_route.evaluation.runtime_manifest import load_runtime_manifest
from evidence_route.llm import OpenAITransport, ensure_v1, make_call_id

TransportFactory = Callable[[Any], Any]
ExecutorFactory = Callable[..., Any]


class _DeferredExecutor:
    """Construct the paid transport only after runner preflight has completed."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._executor: Any | None = None

    async def __call__(self, work: CampaignWorkItem) -> Any:
        if self._executor is None:
            self._executor = self._factory()
        result = self._executor(work)
        return await result if inspect.isawaitable(result) else result


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"{label} input is missing: {path}")
    return path


def _claim_map(manifest: Any) -> dict[str, str]:
    return {item.claim_id: item.claim for item in manifest.items}


def _phase_status(items: Sequence[tuple[CampaignWorkItem, Any]]) -> CampaignStatus:
    if not items:
        return CampaignStatus.PLANNED
    reasons = [item.stop_reason for _, item in items if item.stop_reason is not None]
    if reasons:
        return derive_campaign_status([item for _, item in items], stop_reasons=reasons)
    statuses = [item.status for _, item in items]
    terminal = {WorkStatus.COMPLETED, WorkStatus.PARTIAL, WorkStatus.FAILED}
    if all(status in terminal for status in statuses):
        return CampaignStatus.COMPLETE
    if WorkStatus.INTERRUPTED in statuses:
        return CampaignStatus.INTERRUPTED
    if WorkStatus.RUNNING in statuses:
        return CampaignStatus.RUNNING
    if WorkStatus.NOT_RUN_BUDGET in statuses:
        return CampaignStatus.INCOMPLETE_BUDGET
    if WorkStatus.CANCELLED in statuses:
        return CampaignStatus.CANCELLED
    return CampaignStatus.PLANNED


class ProductionCampaignService:
    """Create, resume, and account for the production campaign."""

    def __init__(
        self,
        *,
        transport_factory: TransportFactory | None = None,
        executor_factory: ExecutorFactory | None = None,
        repository_root: Path | None = None,
    ) -> None:
        self.transport_factory = transport_factory
        self.executor_factory = executor_factory or GraphCampaignExecutor
        self.repository_root = (
            Path(repository_root) if repository_root else Path(__file__).resolve().parents[3]
        )

    def _transport(self, app_config: AppConfig) -> Any:
        if self.transport_factory is not None:
            return self.transport_factory(app_config.llm)
        return OpenAITransport(app_config.llm)

    @staticmethod
    def _load_pricing(path: Path) -> PriceConfig:
        from evidence_route.execution import load_price_config

        return load_price_config(_require_file(path, "pricing"))

    def _build_freeze(
        self,
        *,
        calibration_plan: Any,
        dev_manifest: Any,
        stability_manifest: Any,
        config_path: Path,
        pricing_path: Path,
        corpus_dir: Path,
        app_config: AppConfig,
    ) -> FreezeIdentity:
        receipt = _require_file(
            corpus_dir.parent / "preparation_receipt.json", "corpus preparation receipt"
        )
        prompts = _require_file(
            self.repository_root / "src" / "evidence_route" / "prompts.py", "prompt bundle"
        )
        requirements = _require_file(
            self.repository_root / "requirements.lock", "requirements lock"
        )
        return build_freeze_identity(
            calibration_manifest=calibration_plan.runtime_manifest_sha256,
            dev_manifest=dev_manifest,
            stability_manifest=stability_manifest,
            corpus_receipt=receipt,
            prompt_bundle=prompts,
            config_file=config_path,
            pricing_file=pricing_path,
            requirements_lock=requirements,
            endpoint_config={"base_url": ensure_v1(app_config.llm.base_url)},
            requested_alias=app_config.llm.requested_alias,
            seed=calibration_plan.seed,
            manifest_freeze_git_sha=calibration_plan.manifest_freeze_git_sha,
            dev_protocol_git_sha=git_head(self.repository_root),
        )

    @staticmethod
    def _validate_replay(
        activity_dir: Path,
        plan: Any,
        config_path: Path,
        report_path: Path,
        baseline_config_path: Path | None = None,
    ) -> None:
        metadata_path = activity_dir / "calibration-replay.json"
        if not metadata_path.is_file():
            raise ValueError("calibration replay metadata is missing")
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("plan_fingerprint") != plan.plan_fingerprint:
            raise ValueError("calibration replay belongs to a different plan")
        calibrated_config = baseline_config_path or config_path
        if metadata.get("calibrated_config_sha256") != sha256_file(calibrated_config):
            raise ValueError("calibrated config differs from replay metadata")
        if metadata.get("calibration_report_sha256") != sha256_file(report_path):
            raise ValueError("calibration report differs from replay metadata")

    @staticmethod
    def _validate_production_plan(plan: CampaignPlan) -> None:
        dev = [item for item in plan.schedule if item.phase == "dev"]
        stability = [item for item in plan.schedule if item.phase == "stability"]
        if len(dev) != 240 or len(stability) != 40 or len(plan.schedule) != 280:
            raise ValueError("Gate A campaign requires exactly 240 dev and 40 stability items")

    @staticmethod
    def _verify_calibration_freeze(
        *,
        calibration_plan: Any,
        current_freeze: FreezeIdentity,
        app_config: AppConfig,
        config_path: Path,
        pricing_path: Path,
        calibration_report: Path,
        replay_metadata: Path,
        allow_prompt_drift: bool = False,
    ) -> None:
        frozen = {
            "manifest_freeze_git_sha": calibration_plan.manifest_freeze_git_sha,
            "calibration_runtime_manifest_sha256": calibration_plan.runtime_manifest_sha256,
            "corpus_preparation_receipt_sha256": calibration_plan.corpus_preparation_receipt_sha256,
            "prompt_bundle_sha256": calibration_plan.prompt_bundle_sha256,
            "pricing_sha256": calibration_plan.pricing_sha256,
            "endpoint_config_sha256": calibration_plan.endpoint_config_sha256,
            "requirements_lock_sha256": calibration_plan.requirements_lock_sha256,
            "requested_alias": calibration_plan.requested_alias,
            "seed": calibration_plan.seed,
        }
        actual = {
            "manifest_freeze_git_sha": current_freeze.manifest_freeze_git_sha,
            "calibration_runtime_manifest_sha256": (
                current_freeze.calibration_runtime_manifest_sha256
            ),
            "corpus_preparation_receipt_sha256": current_freeze.corpus_preparation_receipt_sha256,
            "prompt_bundle_sha256": current_freeze.prompt_bundle_sha256,
            "pricing_sha256": current_freeze.pricing_sha256,
            "endpoint_config_sha256": current_freeze.endpoint_config_sha256,
            "requirements_lock_sha256": current_freeze.requirements_lock_sha256,
            "requested_alias": current_freeze.requested_alias,
            "seed": current_freeze.seed,
        }
        for field, expected in frozen.items():
            if allow_prompt_drift and field == "prompt_bundle_sha256":
                continue
            if actual[field] != expected:
                raise ValueError(f"calibration freeze differs in {field}")
        cap_micro_cny = int(app_config.budget.estimated_cost_cap_cny * 1_000_000)
        if cap_micro_cny != calibration_plan.cap_micro_cny:
            raise ValueError("calibration budget cap differs from frozen plan")
        if sha256_file(pricing_path) != calibration_plan.pricing_sha256:
            raise ValueError("pricing bytes differ from frozen calibration plan")
        payload = json.loads(calibration_report.read_text(encoding="utf-8"))
        if (
            payload.get("runtime_manifest_sha256") != calibration_plan.runtime_manifest_sha256
            or payload.get("prompt_bundle_sha256") != calibration_plan.prompt_bundle_sha256
            or len(payload.get("candidates", [])) != 54
            or not isinstance(payload.get("selected"), dict)
        ):
            raise ValueError("calibration report is not a complete frozen replay")
        selected_hash = payload["selected"].get("config_hash")
        if selected_hash != stable_hash(app_config.routing.model_dump(mode="json")):
            raise ValueError("current routing config differs from selected calibration policy")
        metadata = json.loads(replay_metadata.read_text(encoding="utf-8"))
        if metadata.get("runtime_manifest_sha256") != calibration_plan.runtime_manifest_sha256:
            raise ValueError("replay metadata runtime manifest differs from calibration plan")

    @staticmethod
    def _verify_calibration_ledger(
        *,
        activity_dir: Path,
        calibration_plan: Any,
        calibration_state: Any,
        activity: ActivityRecord,
        run_store: SQLiteRunStore,
        pricing: PriceConfig,
    ) -> tuple[list[CalibrationRuntimeCase], list[str]]:
        if activity.calibration_plan_sha256 != sha256_file(activity_dir / "calibration-plan.json"):
            raise ValueError("activity calibration plan hash differs from persisted plan")
        if activity.calibration_state_sha256 != sha256_file(
            activity_dir / "calibration-state.json"
        ):
            raise ValueError("activity calibration state hash differs from persisted state")
        if run_store.cap_micro_cny != calibration_plan.cap_micro_cny:
            raise ValueError("shared run store cap differs from calibration plan")
        if activity.calibration_status is not CampaignStatus.COMPLETE:
            raise ValueError("calibration activity is not complete")
        cases = load_calibration_cases(
            activity_dir, calibration_plan, calibration_state, require_complete=True
        )
        expected_links = [(case.case_id, case.artifact_sha256) for case in cases]
        actual_links = [
            (link.case_id, link.artifact_sha256) for link in activity.calibration_artifacts
        ]
        if actual_links != expected_links:
            raise ValueError("activity calibration artifacts differ from verified cases")
        calibration_call_ids: list[str] = []
        calibration_model_ids: set[str] = set()
        for case in cases:
            summaries = [
                run_store.summarize_run(run_id)
                for run_id in (case.router_run_id, case.single_run_id, case.multi_run_id)
            ]
            ledger_ids = [call_id for summary in summaries for call_id in summary.call_ids]
            if ledger_ids != case.call_ids:
                raise ValueError("calibration run store call IDs differ from saved case")
            calibration_call_ids.extend(case.call_ids)
            calibration_model_ids.update(case.response_model_ids_raw)
            model_ids = set(case.response_model_ids_raw)
            if len(model_ids) != 1:
                raise ValueError("calibration case requires exactly one raw model ID")
            for call_id in case.call_ids:
                metadata = run_store.get_call_metadata(call_id)
                if metadata is None or metadata.get("state") != "completed":
                    raise ValueError("calibration run store call is not completed")
                run_id = metadata.get("run_id")
                node = metadata.get("node")
                task_id = metadata.get("task_id")
                logical_attempt = metadata.get("logical_attempt")
                if (
                    metadata.get("activity_id") != calibration_plan.activity_id
                    or run_id not in {case.router_run_id, case.single_run_id, case.multi_run_id}
                    or not isinstance(run_id, str)
                    or not isinstance(node, str)
                    or not isinstance(task_id, str)
                    or not isinstance(logical_attempt, int)
                    or make_call_id(run_id, node, task_id, logical_attempt) != call_id
                ):
                    raise ValueError("calibration call slot identity differs from saved case")
                if metadata.get("request_sha256") != case.request_sha256_by_call_id[call_id]:
                    raise ValueError(
                        "calibration run store request fingerprint differs from saved case"
                    )
                if metadata.get("requested_alias") != case.requested_alias:
                    raise ValueError("calibration call alias differs from saved case")
                usage = metadata.get("usage")
                if (
                    metadata.get("usage_source") != "provider"
                    or usage is None
                    or not usage.complete
                ):
                    raise ValueError("calibration call usage is incomplete")
                if metadata.get("actual_cost_micro_cny") != pricing.estimate_micro_cny(usage):
                    raise ValueError("calibration call cost differs from saved usage")
                if (
                    not isinstance(metadata.get("response_model_id_raw"), str)
                    or metadata.get("response_model_id_raw") not in model_ids
                    or metadata.get("identity_verified") is not False
                ):
                    raise ValueError("calibration call model identity differs from saved case")
            saved_usages = [
                case.router_usage,
                case.single_result.usage,
                case.multi_result.usage,
            ]
            saved_costs = [
                case.router_actual_cost_micro_cny,
                case.single_result.estimated_cost_micro_cny,
                case.multi_result.estimated_cost_micro_cny,
            ]
            if any(
                summary.usage != usage
                for summary, usage in zip(summaries, saved_usages, strict=True)
            ):
                raise ValueError("calibration run store usage differs from saved case")
            if any(
                summary.actual_cost_micro_cny != cost
                for summary, cost in zip(summaries, saved_costs, strict=True)
            ):
                raise ValueError("calibration run store cost differs from saved case")
            if any(
                not summary.usage.complete or summary.billing_uncertain for summary in summaries
            ):
                raise ValueError("calibration run store accounting differs from saved case")
        ledger = run_store.summarize_activity()
        if not calibration_call_ids or not set(calibration_call_ids).issubset(ledger.call_ids):
            raise ValueError("shared run store does not contain the calibration ledger")
        if len(calibration_model_ids) != 1:
            raise ValueError("calibration requires exactly one raw model ID")
        if set(activity.observed_response_model_ids_raw) != calibration_model_ids:
            raise ValueError("activity model identity differs from calibration cases")
        expected_runs = [
            run_id
            for case in cases
            for run_id in (case.router_run_id, case.single_run_id, case.multi_run_id)
        ]
        collection = run_store.summarize_runs(expected_runs)
        if (
            collection.actual_cost_micro_cny is None
            or collection.cost_is_lower_bound
            or collection.billing_uncertain
            or not collection.usage.complete
            or any(source != "provider" for source in collection.usage_sources)
        ):
            raise ValueError("calibration collection accounting is incomplete")
        if collection.requested_aliases != [calibration_plan.requested_alias]:
            raise ValueError("calibration collection alias differs from the frozen plan")
        if collection.response_model_ids_raw != sorted(calibration_model_ids):
            raise ValueError("calibration collection model identity differs from cases")
        return cases, sorted(calibration_model_ids)

    @staticmethod
    def _activity_summary(
        run_store: SQLiteRunStore,
        *,
        calibration_cases: Sequence[CalibrationRuntimeCase],
        state: CampaignState,
        calibration_run_store: SQLiteRunStore | None = None,
    ) -> RunSummary:
        calibration_run_ids = [
            run_id
            for case in calibration_cases
            for run_id in (case.router_run_id, case.single_run_id, case.multi_run_id)
        ]
        campaign_run_ids = list(dict.fromkeys(item.run_id for item in state.items))
        calibration_ledger = (calibration_run_store or run_store).summarize_runs(
            list(dict.fromkeys(calibration_run_ids))
        )
        campaign_ledger = run_store.summarize_runs(campaign_run_ids)
        ledgers = [calibration_ledger, campaign_ledger]
        run_ids = list(
            dict.fromkeys(
                [run_id for case in calibration_cases for run_id in (
                    case.router_run_id,
                    case.single_run_id,
                    case.multi_run_id,
                )]
                + campaign_run_ids
            )
        )
        call_ids = list(
            dict.fromkeys(call_id for ledger in ledgers for call_id in ledger.call_ids)
        )
        usage_complete = all(ledger.usage.complete for ledger in ledgers)
        actual_exact = all(ledger.actual_cost_micro_cny is not None for ledger in ledgers)
        aliases = list(
            dict.fromkeys(alias for ledger in ledgers for alias in ledger.requested_aliases)
        )
        model_ids = list(
            dict.fromkeys(model for ledger in ledgers for model in ledger.response_model_ids_raw)
        )
        return RunSummary(
            run_ids=run_ids,
            call_ids=call_ids,
            usage=Usage(
                input_tokens=sum(ledger.usage.input_tokens for ledger in ledgers),
                output_tokens=sum(ledger.usage.output_tokens for ledger in ledgers),
                total_tokens=sum(ledger.usage.total_tokens for ledger in ledgers),
                complete=usage_complete,
            ),
            usage_sources=[
                source for ledger in ledgers for source in ledger.usage_sources
            ],
            actual_cost_micro_cny=(
                sum(ledger.actual_cost_micro_cny or 0 for ledger in ledgers)
                if actual_exact
                else None
            ),
            known_actual_cost_micro_cny=sum(
                ledger.known_actual_cost_micro_cny for ledger in ledgers
            ),
            committed_cost_micro_cny=sum(ledger.committed_cost_micro_cny for ledger in ledgers),
            cost_is_lower_bound=not actual_exact,
            fresh_call_count=sum(ledger.fresh_call_count for ledger in ledgers),
            cache_hit_count=sum(ledger.cache_hit_count for ledger in ledgers),
            transport_attempts=sum(ledger.transport_attempts for ledger in ledgers),
            requested_aliases=aliases,
            response_model_ids_raw=model_ids,
            identity_verified=False,
            billing_uncertain=any(ledger.billing_uncertain for ledger in ledgers),
        )

    @staticmethod
    def _update_activity(
        activity_path: Path,
        activity: Any,
        plan_path: Path,
        state_path: Path,
        plan: CampaignPlan,
        state: CampaignState,
        summary: RunSummary,
    ) -> Any:
        pairs = list(zip(plan.schedule, state.items, strict=True))
        dev_status = _phase_status([(work, item) for work, item in pairs if work.phase == "dev"])
        stability_status = _phase_status(
            [(work, item) for work, item in pairs if work.phase == "stability"]
        )
        observed = list(
            dict.fromkeys(
                [
                    *activity.observed_response_model_ids_raw,
                    *state.observed_response_model_ids_raw,
                    *summary.response_model_ids_raw,
                ]
            )
        )
        billing_uncertain = state.billing_uncertain or summary.billing_uncertain
        updated = activity.model_copy(
            update={
                "dev_status": dev_status,
                "stability_status": stability_status,
                "campaign_plan_sha256": sha256_file(plan_path),
                "campaign_state_sha256": sha256_file(state_path),
                "observed_response_model_ids_raw": observed,
                "billing_uncertain": billing_uncertain,
                "summary": summary,
                "stop_reason": (
                    CampaignStopReason.BILLING_UNCERTAIN if billing_uncertain else state.stop_reason
                ),
            },
            deep=True,
        )
        if state.stop_reason is CampaignStopReason.USER_PAUSED:
            pending = next(
                (
                    work
                    for work, item in zip(plan.schedule, state.items, strict=True)
                    if item.status is WorkStatus.PENDING
                ),
                None,
            )
            if pending is not None:
                updated = updated.model_copy(
                    update={f"{pending.phase}_status": CampaignStatus.PAUSED},
                    deep=True,
                )
        updated.status = derive_activity_status(
            updated.calibration_status,
            updated.dev_status,
            updated.stability_status,
            stop_reason=updated.stop_reason,
            billing_uncertain=updated.billing_uncertain,
        )
        return ActivityRecord.model_validate(updated.model_dump(mode="python"))

    async def evaluate(self, **kwargs: object) -> dict[str, object]:
        mode = str(kwargs["mode"])
        if mode not in {"start_after_calibration", "resume"}:
            raise ValueError("evaluate mode must be start_after_calibration or resume")
        manifest_path = Path(kwargs["manifest"])
        stability_path = Path(kwargs["stability_manifest"])
        config_path = Path(kwargs["config_path"])
        pricing_path = Path(kwargs["pricing_path"])
        corpus_dir = Path(kwargs["corpus_dir"])
        activity_dir = Path(kwargs["activity_dir"])
        activity_id = str(kwargs["activity_id"])
        campaign_id = str(kwargs["campaign_id"])
        parent_activity_value = kwargs.get("parent_activity")
        parent_report_value = kwargs.get("parent_report")
        parent_config_value = kwargs.get("parent_config")
        experiment_dir_value = kwargs.get("experiment_dir")
        experiment_mode = any(
            value is not None
            for value in (
                parent_activity_value,
                parent_report_value,
                parent_config_value,
                experiment_dir_value,
            )
        )
        if experiment_mode and not all(
            value is not None
            for value in (
                parent_activity_value,
                parent_report_value,
                parent_config_value,
                experiment_dir_value,
            )
        ):
            raise ValueError(
                "experiment mode requires --parent-activity, --parent-report, "
                "--parent-config, and --experiment-dir"
            )
        parent_activity = str(parent_activity_value) if parent_activity_value is not None else None
        parent_report = Path(parent_report_value) if parent_report_value is not None else None
        parent_config = Path(parent_config_value) if parent_config_value is not None else None
        experiment_dir = Path(experiment_dir_value) if experiment_dir_value is not None else None
        if experiment_mode:
            assert (
                parent_activity is not None
                and parent_report is not None
                and parent_config is not None
                and experiment_dir is not None
            )
            if parent_activity == activity_id:
                raise ValueError("experiment activity must be distinct from parent activity")
            validate_experiment_target(parent_report, experiment_dir)
        parent_activity_dir: Path | None = None
        parent_run_store_path: Path | None = None
        if experiment_mode:
            assert parent_activity is not None
            parent_activity_dir = (
                self.repository_root / "artifacts" / "evaluation" / parent_activity
            )
            parent_run_store_path = (
                self.repository_root / "artifacts" / parent_activity / "run-store.sqlite3"
            )
            if not parent_activity_dir.is_dir():
                raise ValueError(f"parent activity directory is missing: {parent_activity_dir}")
            if not parent_run_store_path.is_file():
                raise ValueError(f"parent run store is missing: {parent_run_store_path}")
            if mode == "start_after_calibration":
                materialize_calibration_assets(
                    parent_activity_dir=parent_activity_dir,
                    experiment_activity_dir=activity_dir,
                    experiment_activity_id=activity_id,
                    parent_activity_id=parent_activity,
                )
        max_items = kwargs.get("max_items")
        if max_items is not None and (
            not isinstance(max_items, int) or isinstance(max_items, bool) or max_items <= 0
        ):
            raise ValueError("max_items must be a positive integer")
        checkpoint_db = Path(kwargs["checkpoint_db"])
        run_store_path = Path(kwargs["run_store"])
        report_value = kwargs.get("calibration_report")
        if report_value is None:
            raise ValueError("calibration_report is required for evaluate")
        calibration_report = Path(report_value)
        activity_dir.mkdir(parents=True, exist_ok=True)
        campaign_plan_path = activity_dir / "plan.json"
        campaign_state_path = activity_dir / "campaign.json"
        activity_path = activity_dir / "activity.json"

        app_config = load_app_config(_require_file(config_path, "configuration"))
        pricing = self._load_pricing(pricing_path)
        dev_manifest = load_runtime_manifest(
            _require_file(manifest_path, "dev runtime manifest"),
            allowed_root=manifest_path.parent,
            corpus_root=corpus_dir,
        )
        stability_manifest = load_runtime_manifest(
            _require_file(stability_path, "stability runtime manifest"),
            allowed_root=stability_path.parent,
            corpus_root=corpus_dir,
        )
        if len(dev_manifest.items) != 80 or len(stability_manifest.items) != 20:
            raise ValueError("Gate A requires 80 dev and 20 stability runtime claims")
        calibration_plan = load_calibration_plan(activity_dir / "calibration-plan.json")
        calibration_state = load_calibration_state(
            activity_dir / "calibration-state.json", expected_plan=calibration_plan
        )
        if any(
            item.status is not CalibrationItemStatus.COMPLETE for item in calibration_state.items
        ):
            raise ValueError("calibration must be complete before evaluate")
        self._validate_replay(
            activity_dir,
            calibration_plan,
            config_path,
            _require_file(calibration_report, "calibration report"),
            baseline_config_path=parent_config if experiment_mode else None,
        )
        replay_metadata_path = activity_dir / "calibration-replay.json"
        _require_file(replay_metadata_path, "calibration replay metadata")
        activity = load_activity(activity_path)
        expected_calibration_activity = parent_activity if experiment_mode else activity_id
        if (
            activity.activity_id != activity_id
            or calibration_plan.activity_id != expected_calibration_activity
        ):
            raise ValueError("activity ID differs from calibration activity")
        resumable_stop_reasons = {
            None,
            CampaignStopReason.PROCESS_INTERRUPTION,
            CampaignStopReason.USER_PAUSED,
        }
        if (
            activity.calibration_status is not CampaignStatus.COMPLETE
            or activity.billing_uncertain
            or (mode == "start_after_calibration" and activity.stop_reason is not None)
            or (mode == "resume" and activity.stop_reason not in resumable_stop_reasons)
        ):
            raise ValueError("calibration activity is not closed")

        experiment_identity = None
        if experiment_mode:
            assert (
                parent_activity is not None
                and parent_report is not None
                and parent_config is not None
                and experiment_dir is not None
            )
            if mode == "resume":
                experiment_identity = load_experiment_identity(experiment_dir)
                if experiment_identity.activity_id != activity_id:
                    raise ValueError(
                        "experiment identity activity ID differs from requested activity"
                    )
                if experiment_identity.campaign_id != campaign_id:
                    raise ValueError(
                        "experiment identity campaign ID differs from requested campaign"
                    )
                validate_parent_baseline(
                    experiment_identity,
                    parent_report=parent_report,
                    parent_manifest=manifest_path,
                    pricing=pricing_path,
                    parent_config=parent_config,
                    config=config_path,
                    prompt=self.repository_root / "src" / "evidence_route" / "prompts.py",
                    stability_manifest=stability_path,
                )

        current_freeze = self._build_freeze(
            calibration_plan=calibration_plan,
            dev_manifest=dev_manifest,
            stability_manifest=stability_manifest,
            config_path=config_path,
            pricing_path=pricing_path,
            corpus_dir=corpus_dir,
            app_config=app_config,
        )
        self._verify_calibration_freeze(
            calibration_plan=calibration_plan,
            current_freeze=current_freeze,
            app_config=app_config,
            config_path=config_path,
            pricing_path=pricing_path,
            calibration_report=calibration_report,
            replay_metadata=replay_metadata_path,
            allow_prompt_drift=experiment_mode,
        )
        bounds = estimate_call_bounds(
            compute_gate_a_call_profile(),
            app_config.generation,
            pricing,
            reserve_ratio=app_config.budget.reserve_ratio,
        )
        cap_micro_cny = int(app_config.budget.estimated_cost_cap_cny * 1_000_000)
        calibration_run_store: SQLiteRunStore | None = None
        if experiment_mode:
            assert parent_activity is not None and parent_run_store_path is not None
            SQLiteRunStore.open_existing(
                parent_run_store_path,
                activity_id=parent_activity,
                cap_cny=app_config.budget.estimated_cost_cap_cny,
                pricing=pricing,
            )
            calibration_run_store = SQLiteRunStore.open_existing(
                parent_run_store_path,
                activity_id=parent_activity,
                cap_cny=app_config.budget.estimated_cost_cap_cny,
                pricing=pricing,
            )
        elif mode == "resume":
            _require_file(run_store_path, "shared run store")
            SQLiteRunStore.open_existing(
                run_store_path,
                activity_id=activity_id,
                cap_cny=app_config.budget.estimated_cost_cap_cny,
                pricing=pricing,
            )
        if experiment_mode and parent_run_store_path is not None:
            if run_store_path.resolve() == parent_run_store_path.resolve():
                raise ValueError("experiment run store must be distinct from parent run store")
        if not experiment_mode:
            _require_file(run_store_path, "shared run store")
        run_store = SQLiteRunStore(
            run_store_path,
            activity_id=activity_id,
            cap_cny=app_config.budget.estimated_cost_cap_cny,
            pricing=pricing,
        )
        calibration_cases, calibration_model_ids = self._verify_calibration_ledger(
            activity_dir=parent_activity_dir or activity_dir,
            calibration_plan=calibration_plan,
            calibration_state=calibration_state,
            activity=(
                load_activity((parent_activity_dir or activity_dir) / "activity.json")
                if experiment_mode
                else activity
            ),
            run_store=calibration_run_store or run_store,
            pricing=pricing,
        )
        baseline_artifact_paths: dict[str, Path] = {}
        parent_state: CampaignState | None = None
        if experiment_mode:
            assert parent_activity_dir is not None
            parent_plan = CampaignPlan.model_validate_json(
                (parent_activity_dir / "plan.json").read_text(encoding="utf-8")
            )
            parent_state = CampaignState.model_validate_json(
                (parent_activity_dir / "campaign.json").read_text(encoding="utf-8")
            )
            for link in parent_plan.stability_repeat_zero_links:
                artifact_path = (
                    parent_activity_dir / "artifacts" / f"{link.dev_adaptive_run_id}.json"
                )
                if link.claim_id not in parent_state.stability_repeat_zero_artifact_sha256s:
                    raise ValueError("parent activity is missing a stability repeat-zero link")
                baseline_artifact_paths[link.claim_id] = artifact_path
            if experiment_identity is not None:
                verify_repeat_zero_reuse(experiment_identity, baseline_artifact_paths)

        freeze_kwargs = {
            "calibration_manifest": calibration_plan.runtime_manifest_sha256,
            "dev_manifest": dev_manifest,
            "stability_manifest": stability_manifest,
            "corpus_receipt": corpus_dir.parent / "preparation_receipt.json",
            "prompt_bundle": self.repository_root / "src" / "evidence_route" / "prompts.py",
            "config_file": config_path,
            "pricing_file": pricing_path,
            "requirements_lock": self.repository_root / "requirements.lock",
            "endpoint_config": {"base_url": ensure_v1(app_config.llm.base_url)},
            "requested_alias": app_config.llm.requested_alias,
            "seed": calibration_plan.seed,
            "manifest_freeze_git_sha": calibration_plan.manifest_freeze_git_sha,
            "dev_protocol_git_sha": current_freeze.dev_protocol_git_sha,
            "repository_root": self.repository_root,
        }

        if mode == "start_after_calibration":
            plan = build_campaign_plan(
                list(dev_manifest.items),
                list(stability_manifest.items),
                activity_id=activity_id,
                campaign_id=campaign_id,
                freeze=current_freeze,
                cap_micro_cny=cap_micro_cny,
                call_bounds=bounds,
            )
            self._validate_production_plan(plan)
            verify_current_freeze(current_freeze, **freeze_kwargs)
            if experiment_mode:
                assert (
                    parent_activity is not None
                    and parent_report is not None
                    and parent_config is not None
                    and experiment_dir is not None
                )
                experiment_identity = create_experiment_identity(
                    experiment_id=activity_id,
                    activity_id=activity_id,
                    campaign_id=campaign_id,
                    parent_activity_id=parent_activity,
                    parent_report=parent_report,
                    parent_manifest=manifest_path,
                    pricing=pricing_path,
                    parent_config=parent_config,
                    config=config_path,
                    prompt=self.repository_root / "src" / "evidence_route" / "prompts.py",
                    stability_manifest=stability_path,
                    repeat_schedule={
                        link.claim_id: [0, 1, 2]
                        for link in plan.stability_repeat_zero_links
                    },
                    requested_alias=app_config.llm.requested_alias,
                    response_model_id=calibration_model_ids[0],
                    identity_verified=False,
                    output_dir=experiment_dir,
                    repeat_zero_artifact_sha256s=(
                        parent_state.stability_repeat_zero_artifact_sha256s
                        if parent_state is not None
                        else None
                    ),
                )
                verify_repeat_zero_reuse(experiment_identity, baseline_artifact_paths)
        else:
            if not campaign_plan_path.is_file() or not campaign_state_path.is_file():
                raise ValueError("--resume requires an existing campaign")
            plan = CampaignPlan.model_validate_json(campaign_plan_path.read_text(encoding="utf-8"))
            state = CampaignState.model_validate_json(
                campaign_state_path.read_text(encoding="utf-8")
            )
            if plan.activity_id != activity_id or plan.campaign_id != campaign_id:
                raise ValueError("persisted campaign identity differs from requested campaign")
            verify_campaign_fingerprint(plan)
            self._validate_production_plan(plan)
            if activity.freeze is None:
                raise ValueError("activity is missing the persisted campaign freeze")
            compare_freeze_identity(plan.freeze, activity.freeze)
            if activity.campaign_plan_sha256 != sha256_file(campaign_plan_path):
                raise ValueError("activity campaign plan hash differs from persisted plan")
            if activity.campaign_state_sha256 != sha256_file(campaign_state_path):
                raise ValueError("activity campaign state hash differs from persisted state")
            current_freeze = verify_current_freeze(plan.freeze, **freeze_kwargs)

        def build_executor() -> Any:
            nonlocal activity
            persisted_state = CampaignState.model_validate_json(
                campaign_state_path.read_text(encoding="utf-8")
            )
            current = load_activity(activity_path)
            if current.freeze is None:
                current = seal_activity(
                    current,
                    freeze=plan.freeze,
                    campaign_plan_sha256=sha256_file(campaign_plan_path),
                    campaign_state_sha256=sha256_file(campaign_state_path),
                )
            else:
                compare_freeze_identity(current.freeze, plan.freeze)
            pairs = list(zip(plan.schedule, persisted_state.items, strict=True))
            dev_status = _phase_status(
                [(work, item) for work, item in pairs if work.phase == "dev"]
            )
            stability_status = _phase_status(
                [(work, item) for work, item in pairs if work.phase == "stability"]
            )
            if any(item.status is WorkStatus.RUNNING for _, item in pairs):
                if any(
                    work.phase == "stability" and item.status is WorkStatus.RUNNING
                    for work, item in pairs
                ):
                    stability_status = CampaignStatus.RUNNING
                else:
                    dev_status = CampaignStatus.RUNNING
            if current.stop_reason in {
                CampaignStopReason.PROCESS_INTERRUPTION,
                CampaignStopReason.USER_PAUSED,
            }:
                interrupted_phases = [
                    phase
                    for phase in ("dev", "stability")
                    if getattr(current, f"{phase}_status")
                    in {CampaignStatus.INTERRUPTED, CampaignStatus.PAUSED}
                ]
                if len(interrupted_phases) != 1:
                    raise ValueError("resumable stop requires exactly one interrupted phase")
                current = resume_activity_phase(current, phase=interrupted_phases[0])
                if interrupted_phases[0] == "dev":
                    dev_status = CampaignStatus.RUNNING
                else:
                    stability_status = CampaignStatus.RUNNING
            current = current.model_copy(
                update={
                    "dev_status": dev_status,
                    "stability_status": stability_status,
                    "campaign_plan_sha256": sha256_file(campaign_plan_path),
                    "campaign_state_sha256": sha256_file(campaign_state_path),
                },
                deep=True,
            )
            current.status = derive_activity_status(
                current.calibration_status, current.dev_status, current.stability_status
            )
            activity = ActivityRecord.model_validate(current.model_dump(mode="python"))
            persist_activity(activity_path, activity)
            transport = self._transport(app_config)
            return self.executor_factory(
                activity_id=activity_id,
                campaign_id=campaign_id,
                claims=_claim_map(dev_manifest),
                app_config=app_config,
                pricing=pricing,
                run_store=run_store,
                corpus_dir=corpus_dir,
                checkpoint_db=checkpoint_db,
                trace_dir=activity_dir / "traces",
                transport=transport,
            )

        runner = CampaignRunner(
            activity_dir, _DeferredExecutor(build_executor), run_store=run_store
        )
        state: CampaignState | None = None
        try:
            if mode == "start_after_calibration":
                state = await runner.run(
                    plan,
                    initial_model_ids=calibration_model_ids,
                    max_items=max_items,
                )
            else:
                state = await runner.resume(
                    expected_identity=current_freeze,
                    max_items=max_items,
                )
            return {
                "status": state.status.value,
                "activity_id": activity_id,
                "campaign_id": campaign_id,
                "completed_items": sum(
                    item.status in {WorkStatus.COMPLETED, WorkStatus.PARTIAL, WorkStatus.FAILED}
                    for item in state.items
                ),
                "campaign_plan": str(campaign_plan_path),
                "campaign_state": str(campaign_state_path),
                "activity": str(activity_path),
            }
        finally:
            if campaign_plan_path.is_file() and campaign_state_path.is_file():
                persisted_plan = CampaignPlan.model_validate_json(
                    campaign_plan_path.read_text(encoding="utf-8")
                )
                persisted_state = CampaignState.model_validate_json(
                    campaign_state_path.read_text(encoding="utf-8")
                )
                current = load_activity(activity_path)
                if current.freeze is None:
                    current = seal_activity(
                        current,
                        freeze=persisted_plan.freeze,
                        campaign_plan_sha256=sha256_file(campaign_plan_path),
                        campaign_state_sha256=sha256_file(campaign_state_path),
                    )
                summary = self._activity_summary(
                    run_store,
                    calibration_cases=calibration_cases,
                    state=persisted_state,
                    calibration_run_store=calibration_run_store,
                )
                activity = self._update_activity(
                    activity_path,
                    current,
                    campaign_plan_path,
                    campaign_state_path,
                    persisted_plan,
                    persisted_state,
                    summary,
                )
                persist_activity(activity_path, activity)


__all__ = ["ProductionCampaignService"]
