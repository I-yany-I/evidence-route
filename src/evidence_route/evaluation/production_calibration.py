"""Paid, resumable calibration collection for the production CLI.

The collector deliberately keeps the scorer boundary out of this module.  It loads one
claim-only runtime manifest, freezes all inputs before constructing a transport, and binds
the resulting runtime cases to the shared SQLite call ledger.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evidence_route.analyzer import analyze_claim
from evidence_route.artifacts import BillingStateError, CallState, SQLiteRunStore
from evidence_route.budget import BudgetExceeded, PriceConfig, UsageUnavailable
from evidence_route.config import AppConfig, load_app_config
from evidence_route.contracts import Strategy
from evidence_route.evaluation.activity import CampaignStatus, CampaignStopReason
from evidence_route.evaluation.calibration import (
    CalibrationItemStatus,
    CalibrationPlan,
    CalibrationState,
    attach_calibration_gold,
    begin_calibration_case,
    build_calibration_plan,
    build_calibration_replay,
    build_calibration_state,
    load_calibration_cases,
    load_calibration_plan,
    load_calibration_state,
    persist_calibration_case,
    persist_calibration_plan,
    persist_calibration_state,
    reconcile_calibration_state,
    write_calibration_outputs,
    write_calibration_replay_view,
    write_canonical_json,
)
from evidence_route.evaluation.lifecycle import (
    endpoint_config_hash,
    link_calibration_artifact,
    load_activity,
    mark_calibration_complete,
    persist_activity,
    resume_activity_phase,
    sha256_file,
    transition_activity_phase,
)
from evidence_route.evaluation.production import (
    GraphCampaignExecutor,
    build_calibration_runtime_case,
)
from evidence_route.evaluation.runner import (
    CampaignProcessInterruption,
    compute_gate_a_call_profile,
    estimate_call_bounds,
)
from evidence_route.evaluation.runtime_manifest import load_runtime_manifest
from evidence_route.llm import (
    BillingUncertain,
    OpenAITransport,
    StructuredLLM,
    ensure_v1,
    make_call_id,
)
from evidence_route.providers.averitec import AveritecFrozenProvider
from evidence_route.routing import HybridRouter

TransportFactory = Callable[[Any], Any]
ExecutorFactory = Callable[..., Any]


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"{label} input is missing: {path}")
    return path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_head(root: Path) -> str:
    """Return the local commit that freezes the manifest/config inputs."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("manifest freeze git commit is unavailable") from exc
    value = completed.stdout.strip().lower()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("manifest freeze git commit is invalid")
    return value


def _digest(path: Path, label: str) -> str:
    _require_file(path, label)
    return sha256_file(path)


def _is_resume_file(path: Path) -> bool:
    return path.exists() or path.with_suffix(path.suffix + ".sha256").exists()


def _status_for_error(exc: Exception) -> tuple[CampaignStatus, bool, str]:
    name = exc.__class__.__name__
    message = str(exc).lower()
    if isinstance(exc, CampaignProcessInterruption) or name in {
        "CampaignProcessInterruption",
        "ProcessInterruption",
    }:
        return CampaignStatus.INTERRUPTED, False, "PROCESS_INTERRUPTION"
    if isinstance(exc, BudgetExceeded) or name == "BudgetExceeded":
        return CampaignStatus.INCOMPLETE_BUDGET, False, "BUDGET"
    if isinstance(exc, UsageUnavailable) or name == "UsageUnavailable":
        return CampaignStatus.INCOMPLETE_USAGE, False, "USAGE_MISSING"
    if isinstance(exc, (BillingUncertain, BillingStateError)) or name in {
        "BillingUncertain",
        "BillingStateError",
    }:
        return CampaignStatus.INCOMPLETE_COST_UNCERTAIN, True, "BILLING_UNCERTAIN"
    if "model drift" in message or "raw model" in message or "model" in name.lower():
        return CampaignStatus.INCOMPLETE_MODEL_DRIFT, False, "MODEL_DRIFT"
    if "accounting" in message or "usage" in message:
        return CampaignStatus.INCOMPLETE_USAGE, False, "USAGE_MISSING"
    return CampaignStatus.FAILED, False, "INTERNAL_ERROR"


def _sync_activity_journal_hashes(activity: Any, *, plan_path: Path, state_path: Path) -> Any:
    """Bind activity journal identity to the bytes persisted on disk."""

    return activity.model_copy(
        update={
            "calibration_plan_sha256": sha256_file(plan_path),
            "calibration_state_sha256": sha256_file(state_path),
        },
        deep=True,
    )


def _request_fingerprints(run_store: SQLiteRunStore, call_ids: list[str]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for call_id in call_ids:
        metadata = run_store.get_call_metadata(call_id)
        if metadata is None or not isinstance(metadata.get("request_sha256"), str):
            raise ValueError("completed calibration call is missing its request fingerprint")
        fingerprints[call_id] = metadata["request_sha256"]
    return fingerprints


def _verify_resume_ledger(
    activity_dir: Path,
    plan: CalibrationPlan,
    state: CalibrationState,
    run_store: SQLiteRunStore,
) -> None:
    """Audit persisted calibration runs before a resumed transport can be built."""

    cases = load_calibration_cases(activity_dir, plan, state, require_complete=False)
    complete_case_ids = {case.case_id for case in cases}
    for case in cases:
        summaries = [
            run_store.summarize_run(run_id)
            for run_id in (case.router_run_id, case.single_run_id, case.multi_run_id)
        ]
        ledger_call_ids = [call_id for summary in summaries for call_id in summary.call_ids]
        if ledger_call_ids != case.call_ids:
            raise ValueError("calibration resume ledger call IDs differ from saved case")
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
        for summary, usage, cost in zip(summaries, saved_usages, saved_costs, strict=True):
            if (
                summary.usage != usage
                or summary.actual_cost_micro_cny != cost
                or summary.billing_uncertain
                or not summary.usage.complete
            ):
                raise ValueError("calibration resume ledger accounting differs from saved case")
        for call_id in ledger_call_ids:
            metadata = run_store.get_call_metadata(call_id)
            if metadata is None or metadata.get("state") != CallState.COMPLETED.value:
                raise ValueError("calibration resume ledger contains a non-completed call")
            if metadata.get("request_sha256") != case.request_sha256_by_call_id[call_id]:
                raise ValueError("calibration resume ledger request fingerprint differs")
            if metadata.get("requested_alias") != case.requested_alias:
                raise ValueError("calibration resume ledger alias differs")
            if metadata.get("usage_source") != "provider" or not getattr(
                metadata.get("usage"), "complete", False
            ):
                raise ValueError("calibration resume ledger usage is incomplete")

    # An unfinished case may legitimately have reserved/completed calls, but a transmitted or
    # billing-uncertain row cannot be retried safely and must stop before transport construction.
    for work, item in zip(plan.items, state.items, strict=True):
        if item.case_id in complete_case_ids:
            continue
        for run_id in (work.router_run_id, work.single_run_id, work.multi_run_id):
            unresolved = run_store.unresolved_call_states(run_id)
            if any(
                value in {CallState.SENT, CallState.BILLING_UNCERTAIN}
                for value in unresolved.values()
            ):
                raise BillingStateError(
                    f"calibration resume run {run_id} has unresolved transmitted calls"
                )


class ProductionCalibrationCollector:
    """Orchestrate the frozen 32-case production collection.

    ``transport_factory`` and ``executor_factory`` are intentionally narrow seams for tests;
    the default factories construct the real OpenAI transport and LangGraph executor.
    """

    def __init__(
        self,
        *,
        transport_factory: TransportFactory | None = None,
        executor_factory: ExecutorFactory | None = None,
        repository_root: Path | None = None,
    ) -> None:
        self.transport_factory = transport_factory
        self.executor_factory = executor_factory or GraphCampaignExecutor
        self.repository_root = Path(repository_root) if repository_root else _repository_root()

    async def collect(self, **kwargs: object) -> dict[str, object]:
        runtime_manifest_path = Path(kwargs["runtime_manifest"])
        config_path = Path(kwargs["config_path"])
        pricing_value = kwargs.get("pricing_path")
        if pricing_value is None:
            pricing_value = os.environ.get("EVIDENCE_ROUTE_PRICE_FILE")
        if pricing_value is None:
            raise ValueError("pricing_path or EVIDENCE_ROUTE_PRICE_FILE is required")
        pricing_path = Path(pricing_value)
        corpus_dir = Path(kwargs["corpus_dir"])
        activity_dir = Path(kwargs["activity_dir"])
        checkpoint_db = Path(kwargs["checkpoint_db"])
        run_store_path = Path(kwargs["run_store"])
        activity_id = str(kwargs["activity_id"])
        resume = bool(kwargs.get("resume", False))
        max_cases = kwargs.get("max_cases")
        if max_cases is not None and (
            not isinstance(max_cases, int)
            or isinstance(max_cases, bool)
            or max_cases <= 0
        ):
            raise ValueError("max_cases must be a positive integer")
        activity_dir.mkdir(parents=True, exist_ok=True)

        plan_path = activity_dir / "calibration-plan.json"
        state_path = activity_dir / "calibration-state.json"
        activity_path = activity_dir / "activity.json"
        if resume:
            self._require_resume_files(plan_path, state_path, activity_path)
        elif any(_is_resume_file(path) for path in (plan_path, state_path, activity_path)):
            raise FileExistsError("calibration activity already exists; use --resume")

        # All freeze inputs are read before any transport object is used.  In particular, this
        # loader cannot import or inspect scorer/gold manifests.
        manifest = load_runtime_manifest(
            _require_file(runtime_manifest_path, "runtime manifest"),
            allowed_root=runtime_manifest_path.parent,
            corpus_root=corpus_dir,
        )
        claims = list(manifest.items)
        if len(claims) != 32:
            raise ValueError("calibration runtime manifest must contain exactly 32 claims")
        if any(item.split != "train" for item in claims):
            raise ValueError("calibration runtime manifest must contain only train claims")
        claim_ids = [item.claim_id for item in claims]
        if len(set(claim_ids)) != 32:
            raise ValueError("calibration runtime manifest must contain 32 unique train claims")

        app_config = load_app_config(_require_file(config_path, "configuration"))
        pricing = self._load_pricing(pricing_path)
        receipt_path = _require_file(
            corpus_dir.parent / "preparation_receipt.json", "corpus preparation receipt"
        )
        prompt_path = _require_file(
            self.repository_root / "src" / "evidence_route" / "prompts.py", "prompt bundle"
        )
        requirements_path = _require_file(
            self.repository_root / "requirements.lock", "requirements lock"
        )
        manifest_sha = getattr(manifest, "_manifest_sha256", None)
        if not isinstance(manifest_sha, str):
            manifest_sha = hashlib.sha256(runtime_manifest_path.read_bytes()).hexdigest()
        freeze_git_sha = _git_head(self.repository_root)
        endpoint_sha = endpoint_config_hash(base_url=ensure_v1(app_config.llm.base_url))
        cap_micro_cny = int(app_config.budget.estimated_cost_cap_cny * 1_000_000)
        if cap_micro_cny <= 0:
            raise ValueError("configured calibration cost cap is below one micro-CNY")
        gate_a_bounds = estimate_call_bounds(
            compute_gate_a_call_profile(),
            app_config.generation,
            pricing,
            reserve_ratio=app_config.budget.reserve_ratio,
        )
        if gate_a_bounds.startup_required_micro_cny > cap_micro_cny:
            raise BudgetExceeded("startup worst-case Gate A reservation exceeds configured cap")
        expected_plan = build_calibration_plan(
            claims,
            activity_id=activity_id,
            manifest_freeze_git_sha=freeze_git_sha,
            runtime_manifest_sha256=manifest_sha,
            corpus_preparation_receipt_sha256=sha256_file(receipt_path),
            prompt_bundle_sha256=sha256_file(prompt_path),
            config_sha256=sha256_file(config_path),
            pricing_sha256=sha256_file(pricing_path),
            endpoint_config_sha256=endpoint_sha,
            requirements_lock_sha256=sha256_file(requirements_path),
            requested_alias=app_config.llm.requested_alias,
            seed=manifest.seed,
            cap_micro_cny=cap_micro_cny,
        )

        if resume:
            plan = load_calibration_plan(plan_path)
            if plan != expected_plan:
                raise ValueError("persisted calibration plan differs from current freeze inputs")
            state = load_calibration_state(state_path, expected_plan=plan)
            activity = load_activity(activity_path)
            self._validate_activity_identity(activity, plan, state, plan_path)
        else:
            plan = expected_plan
            state = build_calibration_state(plan)
            # Persist the complete freeze journal before creating the transport or making a
            # possible paid call.  Each writer uses fsync + replace and activity has a sidecar.
            persist_calibration_plan(plan_path, plan, fresh=True)
            persist_calibration_state(state_path, state)
            from evidence_route.evaluation.lifecycle import build_initial_activity

            activity = build_initial_activity(
                activity_id=activity_id,
                calibration_plan_sha256=sha256_file(plan_path),
                calibration_state_sha256=sha256_file(state_path),
                case_ids=[item.case_id for item in plan.items],
            )
            persist_activity(activity_path, activity, fresh=True)

        if resume:
            if not run_store_path.is_file():
                raise ValueError(f"resume requires an existing run store: {run_store_path}")
            # Validate the immutable ledger without creating tables or metadata.  The writable
            # handle below is opened only after this preflight succeeds.
            SQLiteRunStore.open_existing(
                run_store_path,
                activity_id=activity_id,
                cap_cny=app_config.budget.estimated_cost_cap_cny,
                pricing=pricing,
            )
        run_store = SQLiteRunStore(
            run_store_path,
            activity_id=activity_id,
            cap_cny=app_config.budget.estimated_cost_cap_cny,
            pricing=pricing,
        )

        # Resume reconciliation happens before the first possible LLM call.  Closed artifacts
        # win over stale journal entries, while an orphaned RUNNING entry is made resumable.
        if resume:
            state = reconcile_calibration_state(activity_dir, plan, state, state_path=state_path)
            _verify_resume_ledger(activity_dir, plan, state, run_store)
            activity = self._link_reconciled_cases(activity, plan, state)
            if activity.stop_reason in {
                CampaignStopReason.PROCESS_INTERRUPTION,
                CampaignStopReason.USER_PAUSED,
            }:
                activity = resume_activity_phase(activity, phase="calibration")
            activity = _sync_activity_journal_hashes(
                activity,
                plan_path=plan_path,
                state_path=state_path,
            )
            persist_activity(activity_path, activity)

        existing_model_ids = set(activity.observed_response_model_ids_raw)
        existing_model_ids.update(run_store.summarize_activity().response_model_ids_raw)
        if len(existing_model_ids) > 1:
            raise ValueError("calibration model drift detected in persisted call ledger")
        if existing_model_ids and sorted(existing_model_ids) != sorted(
            activity.observed_response_model_ids_raw
        ):
            activity = activity.model_copy(
                update={"observed_response_model_ids_raw": sorted(existing_model_ids)},
                deep=True,
            )
            persist_activity(activity_path, activity)

        if all(item.status is CalibrationItemStatus.COMPLETE for item in state.items):
            return self._finalize(
                activity_dir=activity_dir,
                activity_path=activity_path,
                state_path=state_path,
                plan=plan,
                state=state,
                activity=activity,
                run_store=run_store,
            )

        transport = self._make_transport(app_config)
        provider = AveritecFrozenProvider(corpus_dir)
        router_llm = StructuredLLM(
            settings=app_config.llm, transport=transport, run_store=run_store
        )
        forced_router = HybridRouter(app_config.routing, app_config.generation, llm=router_llm)
        claims_by_id = {item.claim_id: item.claim for item in claims}
        executor_kwargs = {
            "activity_id": activity_id,
            "campaign_id": "calibration",
            "claims": claims_by_id,
            "app_config": app_config,
            "pricing": pricing,
            "run_store": run_store,
            "corpus_dir": corpus_dir,
            "checkpoint_db": checkpoint_db,
            "trace_dir": activity_dir / "traces",
            "transport": transport,
        }
        single_executor = self.executor_factory(**executor_kwargs)
        multi_executor = self.executor_factory(**executor_kwargs)

        completed_this_invocation = 0
        for work, state_item in zip(plan.items, state.items, strict=True):
            if state_item.status is CalibrationItemStatus.COMPLETE:
                continue
            if state_item.status is CalibrationItemStatus.STOPPED:
                raise ValueError(
                    f"calibration case {work.case_id} is stopped; explicit recovery is required"
                )
            try:
                begin_calibration_case(plan, state, work.case_id, state_path=state_path)
                activity = transition_activity_phase(
                    activity, phase="calibration", status=CampaignStatus.RUNNING
                )
                activity = activity.model_copy(
                    update={"calibration_state_sha256": sha256_file(state_path)}, deep=True
                )
                persist_activity(activity_path, activity)
                probe = await provider.search(
                    work.claim_id,
                    claims_by_id[work.claim_id],
                    top_k=app_config.evidence.probe_top_k,
                    max_chars=app_config.evidence.probe_chars,
                )
                features = analyze_claim(claims_by_id[work.claim_id], probe)
                decision = await forced_router.route(
                    work.router_run_id,
                    Strategy.ADAPTIVE,
                    features,
                    allow_repair=False,
                    force_llm=True,
                )
                if decision.source != "llm":
                    raise ValueError("forced calibration router did not produce source=llm")

                single = await self._execute_graph(
                    single_executor,
                    run_id=work.single_run_id,
                    claim_id=work.claim_id,
                    strategy=Strategy.ALWAYS_SINGLE,
                )
                multi = await self._execute_graph(
                    multi_executor,
                    run_id=work.multi_run_id,
                    claim_id=work.claim_id,
                    strategy=Strategy.ALWAYS_MULTI,
                )
                case = build_calibration_runtime_case(
                    work=work,
                    runtime_manifest_sha256=plan.runtime_manifest_sha256,
                    features=features,
                    saved_llm_route=decision.route,
                    router_summary=run_store.summarize_run(work.router_run_id),
                    single_result=single.result,
                    single_summary=run_store.summarize_run(work.single_run_id),
                    multi_result=multi.result,
                    multi_summary=run_store.summarize_run(work.multi_run_id),
                    requested_alias=plan.requested_alias,
                    price_config_id=pricing.config_id,
                    request_sha256_by_call_id=_request_fingerprints(
                        run_store,
                        [
                            *run_store.summarize_run(work.router_run_id).call_ids,
                            *run_store.summarize_run(work.single_run_id).call_ids,
                            *run_store.summarize_run(work.multi_run_id).call_ids,
                        ],
                    ),
                )
                case_model_ids = set(case.response_model_ids_raw)
                if existing_model_ids and case_model_ids != existing_model_ids:
                    raise ValueError("calibration model drift detected across cases")
                if len(case_model_ids) != 1:
                    raise ValueError("calibration case requires exactly one raw model ID")
                existing_model_ids = case_model_ids
                persist_calibration_case(activity_dir, plan, state, case, state_path=state_path)
                activity = link_calibration_artifact(
                    activity, case_id=case.case_id, artifact_sha256=case.artifact_sha256
                )
                model_ids = list(activity.observed_response_model_ids_raw)
                model_ids.extend(case.response_model_ids_raw)
                activity = activity.model_copy(
                    update={
                        "calibration_state_sha256": sha256_file(state_path),
                        "observed_response_model_ids_raw": list(dict.fromkeys(model_ids)),
                    },
                    deep=True,
                )
                persist_activity(activity_path, activity)
                completed_this_invocation += 1
                remaining = any(
                    item.status is not CalibrationItemStatus.COMPLETE for item in state.items
                )
                if max_cases is not None and completed_this_invocation >= max_cases and remaining:
                    activity = transition_activity_phase(
                        activity,
                        phase="calibration",
                        status=CampaignStatus.PAUSED,
                        stop_reason=CampaignStopReason.USER_PAUSED,
                    )
                    persist_activity(activity_path, activity)
                    return {
                        "status": CampaignStatus.PAUSED.value,
                        "activity_id": plan.activity_id,
                        "completed_cases": sum(
                            item.status is CalibrationItemStatus.COMPLETE for item in state.items
                        ),
                        "paused": True,
                    }
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                self._persist_failure(
                    activity_path,
                    state_path,
                    activity,
                    plan,
                    state,
                    work.case_id,
                    exc,
                    run_store=run_store,
                )
                raise

        state = load_calibration_state(state_path, expected_plan=plan)
        activity = load_activity(activity_path)
        return self._finalize(
            activity_dir=activity_dir,
            activity_path=activity_path,
            state_path=state_path,
            plan=plan,
            state=state,
            activity=activity,
            run_store=run_store,
        )

    def replay(self, **kwargs: object) -> dict[str, object]:
        """Replay the immutable train cases without constructing a transport."""

        runtime_manifest_path = Path(kwargs["runtime_manifest"])
        gold_manifest_path = Path(kwargs["gold_manifest"])
        config_path = Path(kwargs["config_path"])
        pricing_path = Path(kwargs["pricing_path"])
        corpus_dir = Path(kwargs["corpus_dir"])
        activity_dir = Path(kwargs["activity_dir"])
        checkpoint_db = Path(kwargs["checkpoint_db"])
        run_store_path = Path(kwargs["run_store"])
        requested_activity_id_value = kwargs.get("activity_id")
        requested_activity_id = (
            str(requested_activity_id_value) if requested_activity_id_value is not None else None
        )
        output_config = Path(kwargs["output_config"])
        output_report = Path(kwargs["output_report"])
        plan_path = activity_dir / "calibration-plan.json"
        state_path = activity_dir / "calibration-state.json"
        activity_path = activity_dir / "activity.json"
        replay_meta_path = activity_dir / "calibration-replay.json"
        receipt_path = corpus_dir.parent / "preparation_receipt.json"
        prompt_path = self.repository_root / "src" / "evidence_route" / "prompts.py"
        requirements_path = self.repository_root / "requirements.lock"
        output_paths = [output_config, output_report, replay_meta_path]
        resolved_outputs = {path.resolve() for path in output_paths}
        if len(resolved_outputs) != len(output_paths):
            raise ValueError("replay output paths must be distinct")
        input_paths = {
            runtime_manifest_path,
            runtime_manifest_path.with_suffix(runtime_manifest_path.suffix + ".sha256"),
            gold_manifest_path,
            gold_manifest_path.with_suffix(gold_manifest_path.suffix + ".sha256"),
            config_path,
            pricing_path,
            checkpoint_db,
            run_store_path,
            plan_path,
            state_path,
            activity_path,
            activity_path.with_suffix(activity_path.suffix + ".sha256"),
            receipt_path,
            prompt_path,
            requirements_path,
        }
        if resolved_outputs & {path.resolve() for path in input_paths}:
            raise ValueError("replay output path collides with input artifact")
        _require_file(run_store_path, "run store")
        self._require_resume_files(plan_path, state_path, activity_path)

        runtime = load_runtime_manifest(
            _require_file(runtime_manifest_path, "runtime manifest"),
            allowed_root=runtime_manifest_path.parent,
            corpus_root=corpus_dir,
        )
        resolved_corpus = corpus_dir.resolve()
        if any(
            output.resolve() == resolved_corpus or output.resolve().is_relative_to(resolved_corpus)
            for output in output_paths
        ):
            raise ValueError("replay output path collides with corpus input")
        for item in runtime.items:
            corpus_path = (corpus_dir / item.corpus_relpath).resolve()
            if any(output.resolve() == corpus_path for output in output_paths):
                raise ValueError("replay output path collides with corpus input")
        plan = load_calibration_plan(plan_path)
        if sha256_file(runtime_manifest_path) != plan.runtime_manifest_sha256:
            raise ValueError("runtime manifest digest differs from frozen calibration plan")
        if _digest(config_path, "configuration") != plan.config_sha256:
            raise ValueError("source config digest differs from frozen calibration plan")
        if _digest(receipt_path, "corpus preparation receipt") != (
            plan.corpus_preparation_receipt_sha256
        ):
            raise ValueError("corpus preparation receipt differs from frozen calibration plan")
        if _digest(prompt_path, "prompt bundle") != plan.prompt_bundle_sha256:
            raise ValueError("prompt bundle differs from frozen calibration plan")
        if _digest(requirements_path, "requirements lock") != plan.requirements_lock_sha256:
            raise ValueError("requirements lock differs from frozen calibration plan")
        current_base_url = os.environ.get("EVIDENCE_ROUTE_BASE_URL")
        if current_base_url:
            current_endpoint = endpoint_config_hash(base_url=ensure_v1(current_base_url))
            if current_endpoint != plan.endpoint_config_sha256:
                raise ValueError("endpoint configuration differs from frozen calibration plan")
        current_alias = os.environ.get("EVIDENCE_ROUTE_MODEL")
        if current_alias and current_alias != plan.requested_alias:
            raise ValueError("requested model alias differs from frozen calibration plan")
        current_head = _git_head(self.repository_root)
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", plan.manifest_freeze_git_sha, current_head],
                cwd=self.repository_root,
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                "manifest freeze git commit is not an ancestor of replay code"
            ) from exc
        state = load_calibration_state(state_path, expected_plan=plan)
        activity = load_activity(activity_path)
        if requested_activity_id is not None and (
            plan.activity_id != requested_activity_id
            or activity.activity_id != requested_activity_id
        ):
            raise ValueError("activity ID differs from requested replay activity")
        self._validate_activity_identity(activity, plan, state, plan_path)
        if (
            activity.calibration_status is not CampaignStatus.COMPLETE
            or activity.stop_reason is not None
            or activity.billing_uncertain
        ):
            raise ValueError("calibration activity is not closed")
        if activity.calibration_state_sha256 != sha256_file(state_path):
            raise ValueError("activity calibration state hash differs from persisted state")
        case_paths = {(activity_dir / item.artifact_relpath).resolve() for item in state.items}
        if resolved_outputs & case_paths:
            raise ValueError("replay output path collides with input calibration case")
        cases = load_calibration_cases(activity_dir, plan, state, require_complete=True)
        activity_artifacts = [
            (item.case_id, item.artifact_sha256) for item in activity.calibration_artifacts
        ]
        if activity_artifacts != [
            (item.case_id, item.artifact_sha256) for item in state.items
        ] or activity_artifacts != [(case.case_id, case.artifact_sha256) for case in cases]:
            raise ValueError("activity calibration artifact hashes differ from state or cases")
        observed_models = {model_id for case in cases for model_id in case.response_model_ids_raw}
        if len(observed_models) != 1 or activity.observed_response_model_ids_raw != sorted(
            observed_models
        ):
            raise ValueError("activity model identity differs from saved cases")

        from evidence_route.evaluation.scorer_manifest import (
            align_runtime_and_gold,
            load_gold_manifest,
        )

        gold = load_gold_manifest(
            _require_file(gold_manifest_path, "gold manifest"),
            allowed_root=gold_manifest_path.parent,
        )
        aligned = align_runtime_and_gold(runtime, gold)
        scored = attach_calibration_gold(cases, aligned)
        pricing = self._load_pricing(pricing_path)
        if sha256_file(pricing_path) != plan.pricing_sha256:
            raise ValueError("pricing digest differs from frozen calibration plan")
        run_store = SQLiteRunStore.open_existing(
            run_store_path,
            activity_id=plan.activity_id,
            cap_cny=plan.cap_micro_cny / 1_000_000,
            pricing=pricing,
        )
        expected_run_ids = [
            run_id
            for item in plan.items
            for run_id in (item.router_run_id, item.single_run_id, item.multi_run_id)
        ]
        for case in cases:
            summaries = [
                run_store.summarize_run(run_id)
                for run_id in (case.router_run_id, case.single_run_id, case.multi_run_id)
            ]
            ledger_call_ids = [call_id for summary in summaries for call_id in summary.call_ids]
            if ledger_call_ids != case.call_ids:
                raise ValueError("calibration run store call IDs differ from saved case")
            # Check terminal billing state before comparing derived usage/cost fields.  A
            # tampered in-flight row intentionally reports no exact cost, and the actionable
            # failure is that the calibration call is not completed.
            for call_id in ledger_call_ids:
                metadata = run_store.get_call_metadata(call_id)
                if metadata is None or metadata.get("state") != "completed":
                    raise ValueError("calibration run store call is not completed")
            saved_usages = [
                case.router_usage,
                case.single_result.usage,
                case.multi_result.usage,
            ]
            if any(
                summary.usage != saved_usage
                for summary, saved_usage in zip(summaries, saved_usages, strict=True)
            ):
                raise ValueError("calibration run store usage differs from saved case")
            saved_costs = [
                case.router_actual_cost_micro_cny,
                case.single_result.estimated_cost_micro_cny,
                case.multi_result.estimated_cost_micro_cny,
            ]
            if any(
                summary.actual_cost_micro_cny != saved_cost
                for summary, saved_cost in zip(summaries, saved_costs, strict=True)
            ):
                raise ValueError("calibration run store cost differs from saved case")
            for call_id in ledger_call_ids:
                metadata = run_store.get_call_metadata(call_id)
                run_id = metadata.get("run_id")
                node = metadata.get("node")
                task_id = metadata.get("task_id")
                logical_attempt = metadata.get("logical_attempt")
                if (
                    metadata.get("activity_id") != plan.activity_id
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
                usage = metadata.get("usage")
                if (
                    usage is None
                    or not usage.complete
                    or metadata.get("usage_source") != "provider"
                ):
                    raise ValueError("calibration call usage is incomplete")
                if metadata.get("actual_cost_micro_cny") != pricing.estimate_micro_cny(usage):
                    raise ValueError("calibration call cost differs from saved usage")
                if metadata.get("requested_alias") != case.requested_alias:
                    raise ValueError("calibration call alias differs from saved case")
                model_ids = set(case.response_model_ids_raw)
                if (
                    len(model_ids) != 1
                    or not isinstance(metadata.get("response_model_id_raw"), str)
                    or metadata.get("response_model_id_raw") not in model_ids
                    or metadata.get("identity_verified") is not False
                ):
                    raise ValueError("calibration call model identity differs from saved case")
        collection_accounting = run_store.summarize_runs(expected_run_ids)
        if (
            collection_accounting.actual_cost_micro_cny is None
            or collection_accounting.cost_is_lower_bound
            or collection_accounting.billing_uncertain
            or not collection_accounting.usage.complete
            or any(source != "provider" for source in collection_accounting.usage_sources)
        ):
            raise ValueError("calibration collection accounting is incomplete")
        if collection_accounting.requested_aliases != [plan.requested_alias]:
            raise ValueError("calibration collection accounting alias differs from plan")
        if collection_accounting.response_model_ids_raw != sorted(observed_models):
            raise ValueError("calibration collection accounting model identity differs from cases")
        replay = build_calibration_replay(
            scored,
            plan=plan,
            code_git_sha=_git_head(self.repository_root),
            collection_accounting=collection_accounting.model_dump(mode="json"),
        )
        replay = write_calibration_outputs(
            source_config=config_path,
            output_config=output_config,
            output_report=output_report,
            replay=replay,
        )
        metadata = {
            "schema_version": "1",
            "activity_id": plan.activity_id,
            "plan_fingerprint": plan.plan_fingerprint,
            "runtime_manifest_sha256": plan.runtime_manifest_sha256,
            "calibration_report_sha256": sha256_file(output_report),
            "calibrated_config_sha256": sha256_file(output_config),
            "selected_config_hash": replay.selected.config_hash,
        }
        write_canonical_json(replay_meta_path, metadata)
        return {
            "status": "complete",
            "activity_id": plan.activity_id,
            "selected_config_hash": replay.selected.config_hash,
            "calibrated_config": str(output_config),
            "calibration_report": str(output_report),
            "replay_metadata": str(replay_meta_path),
            "candidate_count": len(replay.candidates),
        }

    @staticmethod
    def _load_pricing(path: Path) -> PriceConfig:
        from evidence_route.execution import load_price_config

        return load_price_config(_require_file(path, "pricing"))

    def _make_transport(self, app_config: AppConfig) -> Any:
        if self.transport_factory is not None:
            return self.transport_factory(app_config.llm)
        return OpenAITransport(app_config.llm)

    @staticmethod
    def _require_resume_files(plan_path: Path, state_path: Path, activity_path: Path) -> None:
        missing = [
            str(path) for path in (plan_path, state_path, activity_path) if not path.is_file()
        ]
        if missing:
            raise ValueError(
                "resume requires existing calibration plan/state/activity: " + ", ".join(missing)
            )

    @staticmethod
    def _validate_activity_identity(
        activity: Any, plan: CalibrationPlan, state: CalibrationState, plan_path: Path
    ) -> None:
        if activity.activity_id != plan.activity_id:
            raise ValueError("activity ID differs from calibration plan")
        if activity.calibration_plan_sha256 != sha256_file(plan_path):
            raise ValueError("activity calibration plan hash differs from persisted plan")
        # The activity journal may legitimately lag the state journal if a process died after
        # writing a case/state pair but before linking that artifact.  Reconciliation below
        # repairs that ordering, so do not reject a resumable stale state hash here.
        if [item.case_id for item in activity.calibration_artifacts] != [
            item.case_id for item in plan.items
        ]:
            raise ValueError("activity calibration artifact order differs from persisted plan")

    @staticmethod
    def _link_reconciled_cases(
        activity: Any, plan: CalibrationPlan, state: CalibrationState
    ) -> Any:
        updated = activity
        for item in state.items:
            if item.status is CalibrationItemStatus.COMPLETE and item.artifact_sha256 is not None:
                updated = link_calibration_artifact(
                    updated, case_id=item.case_id, artifact_sha256=item.artifact_sha256
                )
        return updated

    @staticmethod
    async def _execute_graph(
        executor: Any, *, run_id: str, claim_id: str, strategy: Strategy
    ) -> Any:
        if hasattr(executor, "execute"):
            return await executor.execute(
                run_id=run_id,
                claim_id=claim_id,
                strategy=strategy,
                phase="calibration",
                repeat=0,
            )
        result = executor(run_id=run_id, claim_id=claim_id, strategy=strategy)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    @staticmethod
    def _persist_failure(
        activity_path: Path,
        state_path: Path,
        activity: Any,
        plan: CalibrationPlan,
        state: CalibrationState,
        case_id: str,
        exc: Exception,
        run_store: SQLiteRunStore,
    ) -> None:
        status, billing_uncertain, code = _status_for_error(exc)
        item = next(item for item in state.items if item.case_id == case_id)
        item.status = (
            CalibrationItemStatus.STOPPED
            if status is not CampaignStatus.INTERRUPTED
            else CalibrationItemStatus.INTERRUPTED
        )
        item.artifact_sha256 = None
        item.errors = [*item.errors, code]
        persist_calibration_state(state_path, state)
        reason = {
            "BUDGET": CampaignStopReason.BUDGET,
            "USAGE_MISSING": CampaignStopReason.USAGE_MISSING,
            "BILLING_UNCERTAIN": CampaignStopReason.BILLING_UNCERTAIN,
            "MODEL_DRIFT": CampaignStopReason.MODEL_DRIFT,
            "INTERNAL_ERROR": CampaignStopReason.INTERNAL_ERROR,
            "PROCESS_INTERRUPTION": CampaignStopReason.PROCESS_INTERRUPTION,
        }.get(code, CampaignStopReason.INTERNAL_ERROR)
        activity = transition_activity_phase(
            activity,
            phase="calibration",
            status=status,
            stop_reason=reason,
            billing_uncertain=billing_uncertain,
        )
        observed = list(activity.observed_response_model_ids_raw)
        try:
            observed.extend(run_store.summarize_activity().response_model_ids_raw)
        except Exception:
            # Preserve the primary safety journal even if the diagnostic aggregation itself
            # cannot be read after a provider/accounting failure.
            pass
        activity = activity.model_copy(
            update={
                "calibration_state_sha256": sha256_file(state_path),
                "observed_response_model_ids_raw": list(dict.fromkeys(observed)),
            },
            deep=True,
        )
        persist_activity(activity_path, activity)

    @staticmethod
    def _finalize(
        *,
        activity_dir: Path,
        activity_path: Path,
        state_path: Path,
        plan: CalibrationPlan,
        state: CalibrationState,
        activity: Any,
        run_store: SQLiteRunStore,
    ) -> dict[str, object]:
        cases = load_calibration_cases(activity_dir, plan, state, require_complete=True)
        replay_path = activity_dir / "calibration" / "runtime-cases.jsonl"
        write_calibration_replay_view(replay_path, cases, plan=plan)
        activity = mark_calibration_complete(activity)
        activity = activity.model_copy(
            update={"calibration_state_sha256": sha256_file(state_path)}, deep=True
        )
        persist_activity(activity_path, activity)
        summary = run_store.summarize_activity().model_dump(mode="json")
        return {
            "status": "complete",
            "activity_id": plan.activity_id,
            "completed_cases": len(cases),
            "replay_view": str(replay_path),
            "replay_view_path": str(replay_path),
            "activity_path": str(activity_path),
            "activity": str(activity_path),
            "summary": summary,
        }


__all__ = ["ProductionCalibrationCollector"]
