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
from evidence_route.verification import ClaimDecomposer, SingleVerifier, VerdictJudge


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


class LLM:
    def __init__(self, value):
        self.value = value

    async def invoke(self, **kwargs):
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
async def test_decomposer_rejects_unknown_unit_reference() -> None:
    draft = DecompositionDraft(
        tasks=[VerificationTask(task_id="t0", claim_unit_ids=["u9"], query="bad")]
    )
    with pytest.raises(ValueError, match="claim unit"):
        await ClaimDecomposer(LLM(draft), GenerationSettings()).decompose(
            "run", "claim", features()
        )


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
