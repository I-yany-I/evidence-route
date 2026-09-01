"""Production orchestration helpers for the frozen Gate A evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from evidence_route.artifacts import RunCallSummary, SQLiteRunStore, TraceWriter
from evidence_route.budget import PriceConfig
from evidence_route.config import AppConfig
from evidence_route.contracts import ClaimFeatures, Strategy, VerificationResult
from evidence_route.evaluation.activity import (
    CallBounds,
    CampaignPlan,
    CampaignWorkItem,
    FreezeIdentity,
    LatencyBreakdown,
    RunArtifact,
    StabilityBaselineLink,
    campaign_fingerprint,
    derive_run_id,
)
from evidence_route.evaluation.calibration import (
    CalibrationRuntimeCase,
    CalibrationWorkItem,
    runtime_case_fingerprint,
)
from evidence_route.evaluation.runner import build_dev_schedule
from evidence_route.execution import build_run_artifact
from evidence_route.graph import GraphComponents, build_graph, initial_state
from evidence_route.llm import OpenAITransport, StructuredLLM
from evidence_route.providers.averitec import AveritecFrozenProvider
from evidence_route.routing import HybridRouter
from evidence_route.validation import ResultValidator, adjudicate_verification_results
from evidence_route.verification import (
    ClaimDecomposer,
    EvidenceWorker,
    SingleVerifier,
    VerdictJudge,
)


def _claim_id(value: object) -> str:
    claim_id = getattr(value, "claim_id", None)
    if claim_id is None and isinstance(value, dict):
        claim_id = value.get("claim_id")
    if not isinstance(claim_id, str) or not claim_id:
        raise ValueError("runtime claim must contain a non-empty claim_id")
    return claim_id


def validate_gate_a_cohorts(
    dev_claims: Sequence[object], stability_claims: Sequence[object]
) -> None:
    """Validate the pre-registered 80-row dev and 20-row stability cohorts."""

    dev_ids = [_claim_id(item) for item in dev_claims]
    stability_ids = [_claim_id(item) for item in stability_claims]
    if len(dev_ids) != 80 or len(set(dev_ids)) != 80:
        raise ValueError("dev runtime manifest must contain exactly 80 unique claims")
    if len(stability_ids) != 20 or len(set(stability_ids)) != 20:
        raise ValueError("stability runtime manifest must contain exactly 20 unique claims")
    if not set(stability_ids).issubset(dev_ids):
        raise ValueError("stability runtime claims must be a subset of the dev manifest")


def build_campaign_plan(
    dev_claims: Sequence[object],
    stability_claims: Sequence[object],
    *,
    activity_id: str,
    campaign_id: str,
    freeze: FreezeIdentity,
    cap_micro_cny: int,
    call_bounds: CallBounds,
) -> CampaignPlan:
    """Build the immutable dev/stability plan from the exact frozen cohorts."""

    validate_gate_a_cohorts(dev_claims, stability_claims)
    schedule = build_dev_schedule(dev_claims, seed=freeze.seed, campaign_id=campaign_id)
    adaptive_by_claim = {
        item.claim_id: item.run_id for item in schedule if item.strategy is Strategy.ADAPTIVE
    }
    links: list[StabilityBaselineLink] = []
    for claim in stability_claims:
        claim_id = _claim_id(claim)
        links.append(
            StabilityBaselineLink(
                claim_id=claim_id,
                dev_adaptive_run_id=adaptive_by_claim[claim_id],
            )
        )
        for repeat in (1, 2):
            schedule.append(
                CampaignWorkItem(
                    order=len(schedule),
                    phase="stability",
                    claim_id=claim_id,
                    strategy=Strategy.ADAPTIVE,
                    repeat=repeat,
                    run_id=derive_run_id(
                        campaign_id,
                        "stability",
                        claim_id,
                        Strategy.ADAPTIVE,
                        repeat,
                    ),
                )
            )

    payload = {
        "schema_version": "1",
        "activity_id": activity_id,
        "campaign_id": campaign_id,
        "freeze": freeze.model_dump(mode="json"),
        "cap_micro_cny": cap_micro_cny,
        "call_bounds": call_bounds.model_dump(mode="json"),
        "schedule": [item.model_dump(mode="json") for item in schedule],
        "stability_repeat_zero_links": [item.model_dump(mode="json") for item in links],
    }
    payload["campaign_fingerprint"] = campaign_fingerprint(payload)
    return CampaignPlan.model_validate(payload)


def _require_complete_summary(
    summary: RunCallSummary,
    *,
    requested_alias: str,
    label: str,
) -> None:
    if (
        not summary.call_ids
        or not summary.usage.complete
        or summary.actual_cost_micro_cny is None
        or summary.cost_is_lower_bound
        or summary.billing_uncertain
    ):
        raise ValueError(f"{label} calibration accounting is incomplete")
    if summary.requested_aliases != [requested_alias]:
        raise ValueError(f"{label} calibration alias differs from the frozen alias")
    if len(summary.response_model_ids_raw) != 1:
        raise ValueError(f"{label} calibration requires exactly one raw model ID")


def _bind_result_summary(
    result: VerificationResult,
    summary: RunCallSummary,
    *,
    price_config_id: str,
) -> VerificationResult:
    updated = result.model_copy(
        update={
            "usage": summary.usage,
            "estimated_cost_micro_cny": summary.actual_cost_micro_cny,
            "cost_currency": "CNY",
            "price_config_id": price_config_id,
        }
    )
    return VerificationResult.model_validate(updated.model_dump(mode="python"))


def build_calibration_runtime_case(
    *,
    work: CalibrationWorkItem,
    runtime_manifest_sha256: str,
    features: ClaimFeatures,
    saved_llm_route: Literal["single", "multi"],
    router_summary: RunCallSummary,
    single_result: VerificationResult,
    single_summary: RunCallSummary,
    multi_result: VerificationResult,
    multi_summary: RunCallSummary,
    requested_alias: str,
    price_config_id: str,
    request_sha256_by_call_id: Mapping[str, str],
) -> CalibrationRuntimeCase:
    """Bind one calibration case exclusively to persisted call-store accounting."""

    summaries = {
        "router": router_summary,
        "single": single_summary,
        "multi": multi_summary,
    }
    for label, summary in summaries.items():
        _require_complete_summary(
            summary,
            requested_alias=requested_alias,
            label=label,
        )
    model_ids = {
        model_id for summary in summaries.values() for model_id in summary.response_model_ids_raw
    }
    if len(model_ids) != 1:
        raise ValueError("calibration model drift detected across saved paths")
    call_ids = [call_id for summary in summaries.values() for call_id in summary.call_ids]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("calibration call IDs must be unique across saved paths")

    payload = {
        "claim_id": work.claim_id,
        "case_id": work.case_id,
        "router_run_id": work.router_run_id,
        "single_run_id": work.single_run_id,
        "multi_run_id": work.multi_run_id,
        "call_ids": call_ids,
        "request_sha256_by_call_id": dict(request_sha256_by_call_id),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "features": features.model_dump(mode="json"),
        "saved_llm_route": saved_llm_route,
        "router_usage": router_summary.usage.model_dump(mode="json"),
        "router_actual_cost_micro_cny": router_summary.actual_cost_micro_cny,
        "single_result": _bind_result_summary(
            single_result,
            single_summary,
            price_config_id=price_config_id,
        ).model_dump(mode="json"),
        "multi_result": _bind_result_summary(
            multi_result,
            multi_summary,
            price_config_id=price_config_id,
        ).model_dump(mode="json"),
        "requested_alias": requested_alias,
        "response_model_ids_raw": sorted(model_ids),
        "identity_verified": False,
        "artifact_sha256": "0" * 64,
    }
    payload["artifact_sha256"] = runtime_case_fingerprint(payload)
    return CalibrationRuntimeCase.model_validate(payload)


class GraphCampaignExecutor:
    """Execute one immutable campaign item through the production LangGraph."""

    def __init__(
        self,
        *,
        activity_id: str,
        campaign_id: str,
        claims: dict[str, str],
        app_config: AppConfig,
        pricing: PriceConfig,
        run_store: SQLiteRunStore,
        corpus_dir: Path,
        checkpoint_db: Path,
        trace_dir: Path,
        transport: Any | None = None,
    ) -> None:
        self.activity_id = activity_id
        self.campaign_id = campaign_id
        self.claims = dict(claims)
        self.app_config = app_config
        self.pricing = pricing
        self.run_store = run_store
        self.checkpoint_db = Path(checkpoint_db)
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        provider = AveritecFrozenProvider(corpus_dir)
        llm = StructuredLLM(
            settings=app_config.llm,
            transport=transport or OpenAITransport(app_config.llm),
            run_store=run_store,
        )
        self.provider = provider
        self.components = GraphComponents(
            provider=provider,
            router=HybridRouter(
                app_config.routing,
                app_config.generation,
                llm=llm,
                deterministic_ambiguous=app_config.hardening.deterministic_ambiguous,
            ),
            single=SingleVerifier(
                provider,
                llm,
                app_config.evidence,
                app_config.generation,
                hardened=app_config.hardening.hardened_judge,
            ),
            decomposer=ClaimDecomposer(
                llm,
                app_config.generation,
                deterministic=app_config.hardening.deterministic_decomposition,
            ),
            worker=EvidenceWorker(
                provider,
                llm,
                app_config.evidence,
                app_config.generation,
                hardened=app_config.hardening.hardened_worker,
            ),
            judge=VerdictJudge(
                llm,
                app_config.evidence,
                app_config.generation,
                hardened=app_config.hardening.hardened_judge,
            ),
            validator=ResultValidator(
                low_confidence=app_config.routing.low_confidence,
                minimum_coverage=app_config.routing.minimum_coverage,
                normalize_output=app_config.hardening.normalize_output,
            ),
            evidence_settings=app_config.evidence,
            run_store=run_store,
            trace=TraceWriter(self.trace_dir / "graph.jsonl"),
            adjudicator=(
                adjudicate_verification_results
                if app_config.hardening.adjudication
                else None
            ),
        )

    async def execute(
        self,
        *,
        run_id: str,
        claim_id: str,
        strategy: Strategy,
        phase: Literal["calibration", "dev", "stability"],
        repeat: int,
    ) -> RunArtifact:
        try:
            claim = self.claims[claim_id]
        except KeyError as exc:
            raise ValueError(
                f"scheduled claim is absent from runtime manifests: {claim_id}"
            ) from exc

        graph_config = {"configurable": {"thread_id": run_id}}
        async with AsyncSqliteSaver.from_conn_string(str(self.checkpoint_db)) as saver:
            saver.serde = JsonPlusSerializer(pickle_fallback=True)
            graph = build_graph(self.components, checkpointer=saver)
            snapshot = await graph.aget_state(graph_config)
            values = getattr(snapshot, "values", None) or {}
            if values:
                if values.get("claim_id") != claim_id:
                    raise ValueError("checkpoint claim_id differs from scheduled work")
                if values.get("claim_text") != claim:
                    raise ValueError("checkpoint claim text differs from runtime manifest")
                try:
                    checkpoint_strategy = Strategy(str(values.get("strategy")))
                except ValueError as exc:
                    raise ValueError("checkpoint strategy is invalid") from exc
                if checkpoint_strategy is not strategy:
                    raise ValueError("checkpoint strategy differs from scheduled work")
                input_state = None
            else:
                input_state = initial_state(
                    run_id,
                    claim_id,
                    claim,
                    strategy,
                )
            if values and not getattr(snapshot, "next", ()):
                state = values
            else:
                state = await graph.ainvoke(input_state, config=graph_config)

        result = state.get("final_result")
        if result is None:
            raise ValueError("campaign graph completed without final_result")
        summary = self.run_store.summarize_run(run_id)
        node_ms = sum(item.latency_ms for item in state.get("node_timings", []))
        latency = LatencyBreakdown(
            fresh_end_to_end_ms=node_ms,
            model_active_ms=max(result.latency_ms, 0),
            retry_ms=0,
            queue_ms=0,
            checkpoint_downtime_ms=0,
            total_elapsed_ms=node_ms,
            interruption_count=0,
        )
        return build_run_artifact(
            activity_id=self.activity_id,
            campaign_id=self.campaign_id,
            phase=phase,
            run_id=run_id,
            claim_id=claim_id,
            strategy=strategy,
            repeat=repeat,
            result=result,
            summary=summary,
            requested_alias=self.app_config.llm.requested_alias,
            price_config_id=self.pricing.config_id,
            latency=latency,
        )

    async def __call__(self, work: CampaignWorkItem) -> RunArtifact:
        return await self.execute(
            run_id=work.run_id,
            claim_id=work.claim_id,
            strategy=work.strategy,
            phase=work.phase,
            repeat=work.repeat,
        )


__all__ = [
    "GraphCampaignExecutor",
    "build_calibration_runtime_case",
    "build_campaign_plan",
    "validate_gate_a_cohorts",
]
