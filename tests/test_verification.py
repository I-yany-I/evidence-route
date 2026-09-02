import pytest

from evidence_route.config import EvidenceSettings, GenerationSettings
from evidence_route.contracts import (
    Citation,
    ClaimFeatures,
    ClaimUnit,
    DecompositionDraft,
    Evidence,
    ResultStatus,
    Usage,
    Verdict,
    VerificationTask,
)
from evidence_route.verification import (
    ClaimDecomposer,
    EvidenceWorker,
    SingleVerifier,
    VerdictJudge,
    project_citations,
    result_from_draft,
)


def features() -> ClaimFeatures:
    return ClaimFeatures(
        claim_units=[ClaimUnit(unit_id="u0", text="claim")],
        atomic_clause_count=1,
        entity_count=1,
        numeric_count=0,
        time_scope_count=0,
        has_comparison=False,
        has_causal=False,
        has_contrast=False,
        probe_source_count=1,
        probe_score_spread=0,
        probe_conflict_hint=False,
    )


def evidence() -> list[Evidence]:
    return [
        Evidence(
            evidence_id="av:dev:0:0:0",
            title="Source",
            source_url="https://example.org/source",
            text="The claim is supported.",
            provider="averitec_frozen",
            snapshot_sha256="a" * 64,
            ranking_score=1,
        )
    ]


class Provider:
    async def search(self, claim_id: str, query: str, *, top_k: int, max_chars: int):
        return evidence()


class FailingProvider:
    async def search(self, claim_id: str, query: str, *, top_k: int, max_chars: int):
        raise ValueError("fixture corpus failure")


class LLM:
    def __init__(self, value):
        self.value = value
        self.messages = []

    async def invoke(self, **kwargs):
        self.messages.append(kwargs["messages"])
        return type(
            "Response",
            (),
            {
                "value": self.value,
                "usage": Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
                "actual_cost_micro_cny": 2,
                "response_model_id_raw": "relay",
                "requested_alias": "alias",
                "identity_verified": False,
                "call_ids": ("c",),
                "transport_attempts": 1,
                "usage_source": "provider",
            },
        )()


@pytest.mark.asyncio
async def test_single_verifier_returns_structured_result() -> None:
    draft = type(
        "Draft",
        (),
        {
            "verdict": Verdict.SUPPORTED,
            "confidence": 0.9,
            "rationale": "Source supports it.",
            "citations": [],
        },
    )()
    verifier = SingleVerifier(
        Provider(), LLM(draft), EvidenceSettings(), GenerationSettings()
    )
    envelope = await verifier.verify_with_evidence("run", "dev-0", "claim", features())
    result = envelope.result
    assert result.status is ResultStatus.COMPLETED
    assert result.verdict is Verdict.SUPPORTED
    assert result.available_evidence_ids == ["av:dev:0:0:0"]
    assert envelope.evidence_ids == {"av:dev:0:0:0"}


@pytest.mark.asyncio
async def test_hardened_single_verifier_uses_hardened_prompt() -> None:
    draft = type(
        "Draft",
        (),
        {
            "verdict": Verdict.NOT_ENOUGH_EVIDENCE,
            "confidence": 0.9,
            "rationale": "insufficient",
            "citations": [],
        },
    )()
    llm = LLM(draft)
    verifier = SingleVerifier(
        Provider(), llm, EvidenceSettings(), GenerationSettings(), hardened=True
    )

    await verifier.verify_with_evidence("run", "dev-0", "claim", features())

    assert "full literal claim" in llm.messages[0][0]["content"]


@pytest.mark.asyncio
async def test_hardened_worker_uses_hardened_prompt() -> None:
    task = VerificationTask(task_id="t0", claim_unit_ids=["u0"], query="claim")
    llm = LLM(
        type(
            "Draft",
            (),
            {
                "verdict": Verdict.NOT_ENOUGH_EVIDENCE,
                "confidence": 0.9,
                "citations": [],
                "errors": [],
            },
        )()
    )
    worker = EvidenceWorker(
        Provider(), llm, EvidenceSettings(), GenerationSettings(), hardened=True
    )

    await worker.verify_task_with_evidence("run", "dev-0", task)

    assert "full assigned verification task" in llm.messages[0][0]["content"]


@pytest.mark.asyncio
async def test_worker_failure_records_exception_type_for_diagnostics() -> None:
    task = VerificationTask(task_id="t0", claim_unit_ids=["u0"], query="claim")
    worker = EvidenceWorker(
        FailingProvider(), LLM(object()), EvidenceSettings(), GenerationSettings()
    )

    envelope = await worker.verify_task_with_evidence("run", "dev-0", task)

    assert envelope.result.status is ResultStatus.FAILED
    assert envelope.result.errors == ["WORKER_VERIFICATION_FAILED:ValueError"]


def multi_features() -> ClaimFeatures:
    return features().model_copy(
        update={
            "claim_units": [
                ClaimUnit(unit_id="u0", text="first atomic claim"),
                ClaimUnit(unit_id="u1", text="second atomic claim"),
                ClaimUnit(unit_id="u2", text="third atomic claim"),
            ],
            "atomic_clause_count": 3,
        }
    )


def citation(evidence_id: str, source_url: str, unit_ids: list[str] | None = None) -> Citation:
    return Citation(
        evidence_id=evidence_id,
        claim_unit_ids=unit_ids or ["u0"],
        question="What does the source say?",
        answer=f"The source answers the claim using {evidence_id}.",
        quote=f"Quoted text from {evidence_id}.",
        stance="supports",
        source_url=source_url,
    )


def citation_evidence(evidence_id: str, source_url: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        title=evidence_id,
        source_url=source_url,
        text=f"Evidence text for {evidence_id}.",
        provider="averitec_frozen",
        snapshot_sha256="a" * 64,
        ranking_score=1,
    )


def test_project_citations_is_order_independent_and_deduplicates_source_urls() -> None:
    evidence = [
        citation_evidence("e1", "https://example.org/one"),
        citation_evidence("e2", "https://example.org/two"),
    ]
    first = project_citations(
        [
            citation("e2", "https://example.org/two", ["u1"]),
            citation("e1", "https://example.org/one"),
            citation("unknown", "https://example.org/unknown"),
            citation("e2", "https://EXAMPLE.org:443/two/#quote", ["u1"]),
        ],
        evidence,
        max_citations=4,
    )
    second = project_citations(
        [
            citation("e1", "https://example.org/one"),
            citation("e2", "https://example.org/two", ["u1"]),
        ],
        list(reversed(evidence)),
        max_citations=4,
    )

    assert [item.evidence_id for item in first] == ["e1", "e2"]
    assert [item.evidence_id for item in second] == ["e1", "e2"]


def test_project_citations_does_not_invent_missing_claim_unit_evidence() -> None:
    evidence = [citation_evidence("e1", "https://example.org/one")]

    projected = project_citations(
        [citation("e1", "https://example.org/one", ["u0"])], evidence, max_citations=4
    )

    assert [item.evidence_id for item in projected] == ["e1"]
    assert {unit for item in projected for unit in item.claim_unit_ids} == {"u0"}


def test_project_citations_keeps_same_source_when_it_covers_distinct_units() -> None:
    evidence = [citation_evidence("e1", "https://example.org/one")]
    projected = project_citations(
        [
            citation("e1", "https://example.org/one", ["u0"]),
            citation("e1", "https://example.org/one", ["u1"]),
        ],
        evidence,
        max_citations=4,
    )

    assert len(projected) == 2
    assert [item.claim_unit_ids for item in projected] == [["u0"], ["u1"]]


def test_result_from_draft_discards_unknown_claim_units_without_fabricating_coverage() -> None:
    draft = type(
        "Draft",
        (),
        {
            "verdict": Verdict.SUPPORTED,
            "confidence": 0.9,
            "rationale": "supported",
            "citations": [citation("e1", "https://example.org/one", ["u0", "u9"])],
        },
    )()
    response = type(
        "Response",
        (),
        {
            "value": draft,
            "usage": Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
        },
    )()

    result = result_from_draft(
        response,
        "dev-0",
        "single",
        [citation_evidence("e1", "https://example.org/one")],
        known_claim_unit_ids=["u0"],
    )

    assert [item.claim_unit_ids for item in result.citations] == [["u0"]]


@pytest.mark.asyncio
async def test_decomposer_rejects_unknown_unit_reference() -> None:
    draft = DecompositionDraft(
        tasks=[VerificationTask(task_id="t0", claim_unit_ids=["u9"], query="bad")]
    )
    with pytest.raises(ValueError, match="claim unit"):
        await ClaimDecomposer(LLM(draft), GenerationSettings()).decompose(
            "run", "claim", features()
        )


@pytest.mark.asyncio
async def test_deterministic_decomposer_uses_claim_units_in_order() -> None:
    decomposer = ClaimDecomposer(
        LLM(
            DecompositionDraft(
                tasks=[VerificationTask(task_id="t0", claim_unit_ids=["u0"], query="unused")]
            )
        ),
        GenerationSettings(),
        deterministic=True,
    )

    tasks = await decomposer.decompose("run", "claim", multi_features())

    assert tasks == [
        VerificationTask(task_id="t0", claim_unit_ids=["u0"], query="first atomic claim"),
        VerificationTask(task_id="t1", claim_unit_ids=["u1"], query="second atomic claim"),
        VerificationTask(task_id="t2", claim_unit_ids=["u2"], query="third atomic claim"),
    ]


@pytest.mark.asyncio
async def test_judge_deduplicates_citations() -> None:
    citation = Citation(
        evidence_id="av:dev:0:0:0",
        claim_unit_ids=["u0"],
        question="q",
        answer="a",
        quote="The claim is supported.",
        stance="supports",
        source_url="https://example.org/source",
    )
    worker = type(
        "Worker",
        (),
        {
            "status": ResultStatus.COMPLETED,
            "verdict": Verdict.SUPPORTED,
            "confidence": 0.9,
            "citations": [citation],
            "available_evidence_ids": ["av:dev:0:0:0"],
            "errors": [],
            "usage": Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
        },
    )()
    draft = type(
        "Draft",
        (),
        {
            "verdict": Verdict.SUPPORTED,
            "confidence": 0.9,
            "rationale": "ok",
            "citations": [citation, citation],
        },
    )()
    result = await VerdictJudge(LLM(draft), EvidenceSettings(), GenerationSettings()).judge(
        "run", "dev-0", "claim", [worker], initial_route="multi", escalated=False
    )
    assert len(result.citations) == 1


@pytest.mark.asyncio
async def test_judge_emits_citations_in_canonical_evidence_id_order() -> None:
    citations = [
        Citation(
            evidence_id="e2",
            claim_unit_ids=["u1"],
            question="q2",
            answer="a2",
            quote="quote2",
            stance="supports",
            source_url="https://example.org/2",
        ),
        Citation(
            evidence_id="e1",
            claim_unit_ids=["u0"],
            question="q1",
            answer="a1",
            quote="quote1",
            stance="supports",
            source_url="https://example.org/1",
        ),
    ]
    worker = type(
        "Worker",
        (),
        {
            "status": ResultStatus.COMPLETED,
            "verdict": Verdict.SUPPORTED,
            "confidence": 0.9,
            "citations": [],
            "available_evidence_ids": ["e1", "e2"],
            "errors": [],
            "usage": Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
        },
    )()
    draft = type(
        "Draft",
        (),
        {
            "verdict": Verdict.SUPPORTED,
            "confidence": 0.9,
            "rationale": "ok",
            "citations": citations,
        },
    )()

    result = await VerdictJudge(
        LLM(draft), EvidenceSettings(), GenerationSettings()
    ).judge("run", "dev-0", "claim", [worker], initial_route="multi", escalated=False)

    assert [citation.evidence_id for citation in result.citations] == ["e1", "e2"]


@pytest.mark.asyncio
async def test_judge_completes_insufficient_evidence_with_recoverable_partial_worker() -> None:
    worker = type(
        "Worker",
        (),
        {
            "task_id": "t0",
            "claim_unit_ids": ["u0"],
            "status": ResultStatus.PARTIAL,
            "verdict": Verdict.NOT_ENOUGH_EVIDENCE,
            "confidence": 0.9,
            "citations": [],
            "available_evidence_ids": [],
            "errors": ["INCOMPLETE_COVERAGE"],
            "usage": Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
        },
    )()
    draft = type(
        "Draft",
        (),
        {
            "verdict": Verdict.NOT_ENOUGH_EVIDENCE,
            "confidence": 0.9,
            "rationale": "The available evidence is insufficient.",
            "citations": [],
        },
    )()

    result = await VerdictJudge(
        LLM(draft), EvidenceSettings(), GenerationSettings()
    ).judge("run", "dev-0", "claim", [worker], initial_route="multi", escalated=False)

    assert result.status is ResultStatus.COMPLETED
