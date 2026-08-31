from __future__ import annotations

import operator
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from evidence_route.analyzer import analyze_claim
from evidence_route.contracts import (
    NodeTiming,
    ResultStatus,
    RunStatus,
    Strategy,
    Usage,
    VerificationState,
    VerificationTask,
)
from evidence_route.validation import ValidationAction
from evidence_route.verification import (
    EvidenceWorker,
    SingleVerifier,
    VerificationEnvelope,
    failed_result,
)


@dataclass(frozen=True)
class GraphComponents:
    provider: Any
    router: Any
    single: Any
    decomposer: Any
    worker: Any
    judge: Any
    validator: Any
    evidence_settings: Any
    run_store: Any
    trace: Any


class WorkerInput(TypedDict):
    run_id: str
    claim_id: str
    task: VerificationTask


class ExecutionVerificationState(VerificationState, total=False):
    execution_evidence_ids: Annotated[set[str], operator.or_]


def initial_state(
    run_id: str,
    claim_id: str,
    claim_text: str,
    strategy: Strategy,
    language: str = "auto",
) -> VerificationState:
    return {
        "run_id": run_id,
        "claim_id": claim_id,
        "claim_text": claim_text,
        "language": language,
        "strategy": strategy,
        "status": RunStatus.RUNNING,
        "escalation_count": 0,
        "escalated": False,
        "execution_evidence_ids": set(),
        "worker_results": [],
        "usage": Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=True),
        "node_timings": [],
        "errors": [],
    }


def route_after_router(state: VerificationState) -> Literal["single", "decompose", "end"]:
    if "final_result" in state:
        return "end"
    return "single" if state["route_decision"].route == "single" else "decompose"


def fan_out_workers(state: VerificationState) -> list[Send]:
    return [
        Send(
            "worker",
            {"run_id": state["run_id"], "claim_id": state["claim_id"], "task": task},
        )
        for task in state.get("tasks", [])[:3]
    ]


def route_after_validation(state: VerificationState) -> Literal["decompose", "end"]:
    return "decompose" if state.get("validation_action") == "escalate" else "end"


def _timing(node: str, started: float) -> NodeTiming:
    finished = time.perf_counter()
    now = datetime.now(UTC).isoformat()
    return NodeTiming(
        node=node,
        started_at=now,
        finished_at=now,
        latency_ms=max(0, int((finished - started) * 1000)),
    )


async def _single_envelope(verifier: Any, state: VerificationState) -> VerificationEnvelope:
    args = (
        state["run_id"],
        state["claim_id"],
        state["claim_text"],
        state["claim_features"],
    )
    if isinstance(verifier, SingleVerifier) and type(verifier).verify is SingleVerifier.verify:
        return await verifier.verify_with_evidence(*args)
    return VerificationEnvelope(result=await verifier.verify(*args), evidence_ids=frozenset())


async def _worker_envelope(verifier: Any, payload: WorkerInput) -> VerificationEnvelope:
    args = (payload["run_id"], payload["claim_id"], payload["task"])
    if (
        isinstance(verifier, EvidenceWorker)
        and type(verifier).verify_task is EvidenceWorker.verify_task
    ):
        return await verifier.verify_task_with_evidence(*args)
    return VerificationEnvelope(
        result=await verifier.verify_task(*args), evidence_ids=frozenset()
    )


def make_analyze_node(components: GraphComponents):
    async def analyze(state: VerificationState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            evidence = await components.provider.search(
                state["claim_id"],
                state["claim_text"],
                top_k=components.evidence_settings.probe_top_k,
                max_chars=components.evidence_settings.probe_chars,
            )
            features = analyze_claim(state["claim_text"], evidence)
            return {
                "probe_evidence": evidence,
                "claim_features": features,
                "execution_evidence_ids": {item.evidence_id for item in evidence},
                "node_timings": [_timing("analyze", started)],
            }
        except Exception as exc:
            # Safety-stop exceptions must reach the campaign runner unchanged.
            if exc.__class__.__name__ in {"BudgetExceeded", "UsageUnavailable", "BillingUncertain"}:
                raise
            result = failed_result(
                state["claim_id"],
                initial_route=None,
                failure_stage="pre_route",
                error_code="PROBE_RETRIEVAL_FAILED",
            )
            return {
                "final_result": result,
                "status": RunStatus.FAILED,
                "errors": ["PROBE_RETRIEVAL_FAILED"],
                "node_timings": [_timing("analyze", started)],
            }

    return analyze


def make_route_node(components: GraphComponents):
    async def route(state: VerificationState) -> dict[str, Any]:
        if "final_result" in state:
            return {"route_decision": None}
        decision = await components.router.route(
            state["run_id"], state["strategy"], state["claim_features"]
        )
        return {"route_decision": decision}

    return route


def make_single_node(components: GraphComponents):
    async def single(state: VerificationState) -> dict[str, Any]:
        envelope = await _single_envelope(components.single, state)
        result = envelope.result
        return {
            "draft_result": result,
            "draft_origin": "single",
            "usage": result.usage,
            "execution_evidence_ids": set(envelope.evidence_ids),
        }

    return single


def make_decompose_node(components: GraphComponents):
    async def decompose(state: VerificationState) -> dict[str, Any]:
        tasks = await components.decomposer.decompose(
            state["run_id"], state["claim_text"], state["claim_features"]
        )
        return {"tasks": tasks}

    return decompose


def make_worker_node(components: GraphComponents):
    async def worker(payload: WorkerInput) -> dict[str, Any]:
        envelope = await _worker_envelope(components.worker, payload)
        return {
            "worker_results": [envelope.result],
            "execution_evidence_ids": set(envelope.evidence_ids),
        }

    return worker


def make_judge_node(components: GraphComponents):
    async def judge(state: ExecutionVerificationState) -> dict[str, Any]:
        initial_route = state["route_decision"].route
        result = await components.judge.judge(
            state["run_id"],
            state["claim_id"],
            state["claim_text"],
            state.get("worker_results", []),
            initial_route=initial_route,
            escalated=state.get("escalated", False),
        )
        return {
            "draft_result": result,
            "draft_origin": "multi",
            "usage": result.usage,
            "execution_evidence_ids": set(state.get("execution_evidence_ids", set())),
        }

    return judge


def make_validate_node(components: GraphComponents):
    async def validate(state: ExecutionVerificationState) -> dict[str, Any]:
        result = state["draft_result"]
        evidence_ids = set(state.get("execution_evidence_ids", set()))
        decision = components.validator.validate(
            result=result,
            claim_unit_ids=[item.unit_id for item in state["claim_features"].claim_units],
            evidence_ids=evidence_ids,
            escalation_count=state.get("escalation_count", 0),
            strategy=state["strategy"],
            draft_origin=state.get("draft_origin"),
        )
        action = decision.action
        if isinstance(action, str):
            action = ValidationAction(action)
        if action is ValidationAction.ESCALATE:
            return {
                "validation_action": "escalate",
                "escalation_count": state.get("escalation_count", 0) + 1,
                "escalated": True,
                "errors": list(decision.errors),
            }
        status = (
            RunStatus.COMPLETED
            if decision.result.status is ResultStatus.COMPLETED
            else RunStatus.PARTIAL
            if decision.result.status is ResultStatus.PARTIAL
            else RunStatus.FAILED
        )
        return {
            "validation_action": "accept" if action is ValidationAction.ACCEPT else "fail",
            "final_result": decision.result,
            "status": status,
        }

    return validate


def build_graph(components: GraphComponents, *, checkpointer: Any):
    builder = StateGraph(ExecutionVerificationState)
    builder.add_node("analyze", make_analyze_node(components))
    builder.add_node("route", make_route_node(components))
    builder.add_node("single", make_single_node(components))
    builder.add_node("decompose", make_decompose_node(components))
    builder.add_node("worker", make_worker_node(components))
    builder.add_node("judge", make_judge_node(components))
    builder.add_node("validate", make_validate_node(components))
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "route")
    builder.add_conditional_edges(
        "route",
        route_after_router,
        {"single": "single", "decompose": "decompose", "end": END},
    )
    builder.add_edge("single", "validate")
    builder.add_conditional_edges("decompose", fan_out_workers, ["worker"])
    builder.add_edge("worker", "judge")
    builder.add_edge("judge", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {"decompose": "decompose", "end": END},
    )
    return builder.compile(checkpointer=checkpointer)
