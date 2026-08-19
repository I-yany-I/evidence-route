"""Production orchestration helpers for the frozen Gate A evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from evidence_route.artifacts import SQLiteRunStore, TraceWriter
from evidence_route.budget import PriceConfig
from evidence_route.config import AppConfig
from evidence_route.contracts import Strategy
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
from evidence_route.evaluation.runner import build_dev_schedule
from evidence_route.execution import build_run_artifact
from evidence_route.graph import GraphComponents, build_graph, initial_state
from evidence_route.llm import OpenAITransport, StructuredLLM
from evidence_route.providers.averitec import AveritecFrozenProvider
from evidence_route.routing import HybridRouter
from evidence_route.validation import ResultValidator
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
        item.claim_id: item.run_id
        for item in schedule
        if item.strategy is Strategy.ADAPTIVE
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
            router=HybridRouter(app_config.routing, app_config.generation, llm=llm),
            single=SingleVerifier(
                provider, llm, app_config.evidence, app_config.generation
            ),
            decomposer=ClaimDecomposer(llm, app_config.generation),
            worker=EvidenceWorker(
                provider, llm, app_config.evidence, app_config.generation
            ),
            judge=VerdictJudge(llm, app_config.evidence, app_config.generation),
            validator=ResultValidator(
                low_confidence=app_config.routing.low_confidence,
                minimum_coverage=app_config.routing.minimum_coverage,
            ),
            evidence_settings=app_config.evidence,
            run_store=run_store,
            trace=TraceWriter(self.trace_dir / "graph.jsonl"),
        )

    async def __call__(self, work: CampaignWorkItem) -> RunArtifact:
        try:
            claim = self.claims[work.claim_id]
        except KeyError as exc:
            raise ValueError(
                f"scheduled claim is absent from runtime manifests: {work.claim_id}"
            ) from exc

        graph_config = {"configurable": {"thread_id": work.run_id}}
        async with AsyncSqliteSaver.from_conn_string(str(self.checkpoint_db)) as saver:
            saver.serde = JsonPlusSerializer(pickle_fallback=True)
            graph = build_graph(self.components, checkpointer=saver)
            snapshot = await graph.aget_state(graph_config)
            values = getattr(snapshot, "values", None) or {}
            if values:
                if values.get("claim_id") != work.claim_id:
                    raise ValueError("checkpoint claim_id differs from scheduled work")
                if values.get("claim_text") != claim:
                    raise ValueError("checkpoint claim text differs from runtime manifest")
                try:
                    checkpoint_strategy = Strategy(str(values.get("strategy")))
                except ValueError as exc:
                    raise ValueError("checkpoint strategy is invalid") from exc
                if checkpoint_strategy is not work.strategy:
                    raise ValueError("checkpoint strategy differs from scheduled work")
                input_state = None
            else:
                input_state = initial_state(
                    work.run_id,
                    work.claim_id,
                    claim,
                    work.strategy,
                )
            if values and not getattr(snapshot, "next", ()):
                state = values
            else:
                state = await graph.ainvoke(input_state, config=graph_config)

        result = state.get("final_result")
        if result is None:
            raise ValueError("campaign graph completed without final_result")
        summary = self.run_store.summarize_run(work.run_id)
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
            phase=work.phase,
            run_id=work.run_id,
            claim_id=work.claim_id,
            strategy=work.strategy,
            repeat=work.repeat,
            result=result,
            summary=summary,
            requested_alias=self.app_config.llm.requested_alias,
            price_config_id=self.pricing.config_id,
            latency=latency,
        )


__all__ = [
    "GraphCampaignExecutor",
    "build_campaign_plan",
    "validate_gate_a_cohorts",
]
