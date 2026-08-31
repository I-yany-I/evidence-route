import pytest
from langgraph.checkpoint.memory import InMemorySaver

from evidence_route.config import EvidenceSettings
from evidence_route.contracts import (
    Citation,
    ResultStatus,
    RouteDecision,
    Strategy,
    Usage,
    Verdict,
    VerificationResult,
    VerificationTask,
)
from evidence_route.graph import GraphComponents, build_graph, initial_state
from evidence_route.validation import ResultValidator
from evidence_route.verification import SingleVerifier


class Provider:
    async def search(self, *args, **kwargs):
        return []


class Router:
    def __init__(self, route="single"):
        self.route_name = route

    async def route(self, *args):
        return RouteDecision(
            route=self.route_name,
            source="strategy",
            reason_codes=["test"],
            explanation="test",
            config_hash="a" * 64,
        )


def result(claim_id="dev-0", route="single", escalated=False, citations=None, available=None):
    return VerificationResult(
        claim_id=claim_id,
        status=ResultStatus.COMPLETED,
        verdict=Verdict.SUPPORTED,
        confidence=0.9,
        rationale="supported",
        citations=citations or [],
        available_evidence_ids=available or [],
        initial_route=route,
        escalated=escalated,
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
    )


class Single:
    async def verify(self, run_id, claim_id, claim, features):
        return result(claim_id)


def forged_result(claim_id):
    citation = Citation(
        evidence_id="forged-evidence",
        claim_unit_ids=["u0"],
        question="q",
        answer="a",
        quote="q",
        stance="supports",
        source_url="https://example.org/source",
    )
    return result(
        claim_id,
        citations=[citation],
        available=["forged-evidence"],
    )


class LegacySingle(SingleVerifier):
    def __init__(self):
        self.called = False
        self._execution_evidence_ids = frozenset()

    async def verify(self, run_id, claim_id, claim, features):
        self.called = True
        return forged_result(claim_id)


class Decomposer:
    async def decompose(self, run_id, claim, features):
        return [
            VerificationTask(task_id="t0", claim_unit_ids=["u0"], query="one"),
            VerificationTask(task_id="t1", claim_unit_ids=["u0"], query="two"),
            VerificationTask(task_id="t2", claim_unit_ids=["u0"], query="three"),
        ]


class Worker:
    async def verify_task(self, run_id, claim_id, task):
        from evidence_route.contracts import WorkerResult

        return WorkerResult(
            task_id=task.task_id,
            claim_unit_ids=task.claim_unit_ids,
            status=ResultStatus.COMPLETED,
            verdict=Verdict.SUPPORTED,
            confidence=0.9,
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
        )


class Judge:
    async def judge(self, run_id, claim_id, claim, workers, *, initial_route, escalated):
        return result(claim_id, initial_route, escalated)


class Validator:
    def __init__(self, actions=None):
        self.actions = list(actions or ["accept"])

    def validate(self, **kwargs):
        from evidence_route.validation import ValidationAction, ValidationDecision

        action = self.actions.pop(0)
        return ValidationDecision(ValidationAction(action), kwargs["result"])


def components(route="single", validator=None):
    return GraphComponents(
        provider=Provider(),
        router=Router(route),
        single=Single(),
        decomposer=Decomposer(),
        worker=Worker(),
        judge=Judge(),
        validator=validator or Validator(),
        evidence_settings=EvidenceSettings(),
        run_store=None,
        trace=None,
    )


def forged_components(single=None):
    values = components(validator=None)
    return GraphComponents(
        provider=values.provider,
        router=values.router,
        single=single or LegacySingle(),
        decomposer=values.decomposer,
        worker=values.worker,
        judge=values.judge,
        validator=ResultValidator(low_confidence=0.0, minimum_coverage=0.0),
        evidence_settings=values.evidence_settings,
        run_store=values.run_store,
        trace=values.trace,
    )


@pytest.mark.asyncio
async def test_single_result_finishes_without_multi() -> None:
    graph = build_graph(components(), checkpointer=InMemorySaver())
    state = await graph.ainvoke(
        initial_state("run-single", "dev-0", "Atomic claim", Strategy.ALWAYS_SINGLE),
        config={"configurable": {"thread_id": "run-single"}},
    )
    assert state["final_result"].status is ResultStatus.COMPLETED


@pytest.mark.asyncio
async def test_three_workers_feed_judge() -> None:
    graph = build_graph(components("multi"), checkpointer=InMemorySaver())
    state = await graph.ainvoke(
        initial_state("run-multi", "dev-1", "Compound claim", Strategy.ALWAYS_MULTI),
        config={"configurable": {"thread_id": "run-multi"}},
    )
    assert state["final_result"].status is ResultStatus.COMPLETED
    assert len(state["worker_results"]) == 3


@pytest.mark.asyncio
async def test_single_escalation_preserves_initial_route() -> None:
    validator = Validator(["escalate", "accept"])
    graph = build_graph(components("single", validator), checkpointer=InMemorySaver())
    state = await graph.ainvoke(
        initial_state("run-escalate", "dev-2", "Claim", Strategy.ADAPTIVE),
        config={"configurable": {"thread_id": "run-escalate"}},
    )
    assert state["escalation_count"] == 1
    assert state["final_result"].initial_route == "single"
    assert state["final_result"].escalated is True


@pytest.mark.asyncio
async def test_validation_uses_execution_evidence_ids_not_forged_result_available_ids() -> None:
    single = LegacySingle()
    graph = build_graph(forged_components(single), checkpointer=None)

    state = await graph.ainvoke(
        initial_state("run-forged", "dev-3", "Claim", Strategy.ALWAYS_SINGLE),
        config={"configurable": {"thread_id": "run-forged"}},
    )

    assert single.called is True
    assert state["final_result"].status is ResultStatus.FAILED
    assert "UNKNOWN_EVIDENCE:forged-evidence" in state["final_result"].errors
