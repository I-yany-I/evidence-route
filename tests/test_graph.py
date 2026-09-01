import asyncio
from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from evidence_route.config import EvidenceSettings, GenerationSettings
from evidence_route.contracts import (
    Citation,
    Evidence,
    ResultStatus,
    RouteDecision,
    Strategy,
    Usage,
    Verdict,
    VerificationResult,
    VerificationTask,
)
from evidence_route.graph import GraphComponents, build_graph, initial_state, make_worker_node
from evidence_route.validation import ResultValidator
from evidence_route.verification import EvidenceWorker, SingleVerifier


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


class ConcurrentProvider:
    async def search(self, claim_id, query, *, top_k, max_chars):
        if query not in {"one", "two", "three"}:
            return []
        return [
            Evidence(
                evidence_id=f"worker-{query}",
                title=query,
                source_url=f"https://example.org/{query}",
                text=f"Evidence for {query}.",
                provider="averitec_frozen",
                snapshot_sha256="a" * 64,
                ranking_score=1,
            )
        ]


class ConcurrentWorkerLLM:
    async def invoke(self, **kwargs):
        if kwargs["task_id"] == "t0":
            await asyncio.sleep(0.01)
        draft = type(
            "Draft",
            (),
            {
                "verdict": Verdict.SUPPORTED,
                "confidence": 0.9,
                "citations": [],
                "errors": [],
            },
        )()
        return type(
            "Response",
            (),
            {
                "value": draft,
                "usage": Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
            },
        )()


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


class RecordingValidator(Validator):
    def __init__(self):
        super().__init__()
        self.evidence_ids = []

    def validate(self, **kwargs):
        self.evidence_ids.append(set(kwargs["evidence_ids"]))
        return super().validate(**kwargs)


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


def test_graph_components_accept_an_optional_adjudicator() -> None:
    values = components()

    def adjudicator(candidates):
        return candidates[-1]

    configured = replace(values, adjudicator=adjudicator)

    assert configured.adjudicator is adjudicator


@pytest.mark.asyncio
async def test_escalation_passes_single_and_multi_candidates_to_adjudicator() -> None:
    candidates_seen = []

    def adjudicator(candidates):
        candidates_seen.append(candidates)
        return candidates[-1]

    validator = Validator(["escalate", "accept"])
    graph = build_graph(
        replace(components("single", validator), adjudicator=adjudicator),
        checkpointer=InMemorySaver(),
    )

    state = await graph.ainvoke(
        initial_state("run-adjudication", "dev-2", "Claim", Strategy.ADAPTIVE),
        config={"configurable": {"thread_id": "run-adjudication"}},
    )

    assert len(candidates_seen) == 1
    assert len(candidates_seen[0]) == 2
    assert [candidate.initial_route for candidate in candidates_seen[0]] == ["single", "single"]
    assert state["final_result"].status is ResultStatus.COMPLETED


@pytest.mark.asyncio
async def test_adjudicator_does_not_promote_incomplete_candidates() -> None:
    called = False

    def adjudicator(candidates):
        nonlocal called
        called = True
        raise AssertionError("incomplete candidates must not reach adjudicator")

    validator = Validator(["escalate", "accept"])
    values = components("single", validator)

    class PartialJudge(Judge):
        async def judge(self, run_id, claim_id, claim, workers, *, initial_route, escalated):
            return result(claim_id, initial_route, escalated).model_copy(
                update={"status": ResultStatus.PARTIAL}
            )

    state = await build_graph(
        replace(values, judge=PartialJudge(), adjudicator=adjudicator),
        checkpointer=InMemorySaver(),
    ).ainvoke(
        initial_state("run-adjudication-partial", "dev-2", "Claim", Strategy.ADAPTIVE),
        config={"configurable": {"thread_id": "run-adjudication-partial"}},
    )

    assert called is False
    assert state["final_result"].status is ResultStatus.PARTIAL


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


@pytest.mark.asyncio
async def test_concurrent_worker_nodes_return_their_own_execution_evidence_ids() -> None:
    values = components("multi")
    worker = EvidenceWorker(
        ConcurrentProvider(), ConcurrentWorkerLLM(), EvidenceSettings(), GenerationSettings()
    )
    node = make_worker_node(replace(values, worker=worker))
    tasks = [
        VerificationTask(task_id="t0", claim_unit_ids=["u0"], query="one"),
        VerificationTask(task_id="t1", claim_unit_ids=["u0"], query="two"),
        VerificationTask(task_id="t2", claim_unit_ids=["u0"], query="three"),
    ]
    payloads = [
        {"run_id": "run-workers", "claim_id": "dev-4", "task": task} for task in tasks
    ]

    outputs = await asyncio.gather(*(node(payload) for payload in payloads))

    assert [output["execution_evidence_ids"] for output in outputs] == [
        {"worker-one"},
        {"worker-two"},
        {"worker-three"},
    ]


@pytest.mark.asyncio
async def test_multi_worker_graph_validates_against_aggregated_execution_evidence() -> None:
    values = components("multi")
    validator = RecordingValidator()
    worker = EvidenceWorker(
        ConcurrentProvider(), ConcurrentWorkerLLM(), EvidenceSettings(), GenerationSettings()
    )
    graph = build_graph(
        replace(values, worker=worker, validator=validator), checkpointer=None
    )

    state = await graph.ainvoke(
        initial_state("run-workers", "dev-4", "Compound claim", Strategy.ALWAYS_MULTI),
        config={"configurable": {"thread_id": "run-workers"}},
    )

    expected = {"worker-one", "worker-two", "worker-three"}
    assert state["execution_evidence_ids"] == expected
    assert validator.evidence_ids == [expected]
