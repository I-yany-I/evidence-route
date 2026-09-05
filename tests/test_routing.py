import pytest

from evidence_route.budget import BudgetExceeded, UsageUnavailable
from evidence_route.config import GenerationSettings, RoutingSettings, stable_hash
from evidence_route.contracts import ClaimFeatures, ClaimUnit, Strategy
from evidence_route.routing import HybridRouter, RouterPayload, StructuredCallError


def features(**overrides) -> ClaimFeatures:
    values = dict(
        claim_units=[ClaimUnit(unit_id="u0", text="Atomic claim")],
        atomic_clause_count=1,
        entity_count=1,
        numeric_count=0,
        time_scope_count=0,
        has_comparison=False,
        has_causal=False,
        has_contrast=False,
        probe_source_count=2,
        probe_score_spread=1.0,
        probe_conflict_hint=False,
    )
    values.update(overrides)
    return ClaimFeatures.model_validate(values)


@pytest.mark.asyncio
async def test_clear_multi_precedes_clear_single() -> None:
    router = HybridRouter(RoutingSettings(), GenerationSettings(), llm=None)
    decision = await router.route(
        "run",
        Strategy.ADAPTIVE,
        features(
            atomic_clause_count=3,
            claim_units=[ClaimUnit(unit_id=f"u{i}", text=f"claim {i}") for i in range(3)],
        ),
    )
    assert decision.route == "multi"
    assert decision.source == "rule"


@pytest.mark.asyncio
async def test_atomic_claim_uses_single_rule() -> None:
    decision = await HybridRouter(RoutingSettings(), GenerationSettings(), llm=None).route(
        "run", Strategy.ADAPTIVE, features()
    )
    assert decision.route == "single"


@pytest.mark.asyncio
async def test_fixed_strategy_never_calls_llm() -> None:
    decision = await HybridRouter(RoutingSettings(), GenerationSettings(), llm=None).route(
        "run", Strategy.ALWAYS_MULTI, features()
    )
    assert decision.route == "multi"
    assert decision.source == "strategy"


class FakeLLM:
    async def invoke(self, **kwargs: object):
        return type(
            "Result",
            (),
            {
                "value": RouterPayload(
                    route="single",
                    reason_codes=["bounded_claim"],
                    explanation="One source question",
                )
            },
        )()


@pytest.mark.asyncio
async def test_router_normalizes_reason_code_order_and_duplicates() -> None:
    class UnstableReasonLLM:
        async def invoke(self, **kwargs: object):
            return type(
                "Result",
                (),
                {
                    "value": RouterPayload(
                        route="single",
                        reason_codes=["z_reason", "a_reason", "z_reason"],
                        explanation="One source question",
                    )
                },
            )()

    decision = await HybridRouter(
        RoutingSettings(clear_multi_clauses=999, clear_single_min_sources=999),
        GenerationSettings(),
        llm=UnstableReasonLLM(),
    ).route("run", Strategy.ADAPTIVE, features(atomic_clause_count=2, probe_source_count=1))

    assert decision.reason_codes == ["a_reason", "z_reason"]


class CountingLLM(FakeLLM):
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, **kwargs: object):
        self.calls += 1
        return await super().invoke(**kwargs)


class FailingLLM:
    async def invoke(self, **kwargs: object):
        raise StructuredCallError("router unavailable")


class SafetyStopLLM:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def invoke(self, **kwargs: object):
        raise self.error


@pytest.mark.asyncio
async def test_uncertain_route_uses_structured_llm() -> None:
    claim_features = features(
        atomic_clause_count=2,
        claim_units=[ClaimUnit(unit_id="u0", text="one"), ClaimUnit(unit_id="u1", text="two")],
        probe_source_count=1,
    )
    decision = await HybridRouter(RoutingSettings(), GenerationSettings(), llm=FakeLLM()).route(
        "run", Strategy.ADAPTIVE, claim_features
    )
    assert decision.source == "llm"
    assert decision.route == "single"


@pytest.mark.asyncio
async def test_hardened_ambiguous_route_skips_llm() -> None:
    llm = CountingLLM()
    claim_features = features(
        atomic_clause_count=2,
        claim_units=[ClaimUnit(unit_id="u0", text="one"), ClaimUnit(unit_id="u1", text="two")],
        probe_source_count=1,
    )

    decision = await HybridRouter(
        RoutingSettings(), GenerationSettings(), llm=llm, deterministic_ambiguous=True
    ).route("run", Strategy.ADAPTIVE, claim_features)

    assert decision.route == "multi"
    assert decision.source == "rule"
    assert decision.reason_codes == ["deterministic_ambiguous_multi"]
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_force_llm_bypasses_compound_rule() -> None:
    llm = CountingLLM()
    claim_features = features(
        atomic_clause_count=2,
        claim_units=[ClaimUnit(unit_id="u0", text="one"), ClaimUnit(unit_id="u1", text="two")],
        has_comparison=True,
    )
    decision = await HybridRouter(
        RoutingSettings(clear_multi_clauses=999, clear_single_min_sources=999),
        GenerationSettings(),
        llm=llm,
    ).route("run", Strategy.ADAPTIVE, claim_features, force_llm=True)
    assert decision.source == "llm"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_router_failure_falls_back_to_multi() -> None:
    claim_features = features(
        atomic_clause_count=2,
        claim_units=[ClaimUnit(unit_id="u0", text="one"), ClaimUnit(unit_id="u1", text="two")],
        probe_source_count=1,
    )
    decision = await HybridRouter(RoutingSettings(), GenerationSettings(), llm=FailingLLM()).route(
        "run", Strategy.ADAPTIVE, claim_features
    )
    assert decision.source == "fallback"
    assert decision.route == "multi"
    assert decision.config_hash == stable_hash(RoutingSettings().model_dump(mode="json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [BudgetExceeded("cap"), UsageUnavailable("usage")])
async def test_router_propagates_safety_stops(error: Exception) -> None:
    claim_features = features(
        atomic_clause_count=2,
        claim_units=[ClaimUnit(unit_id="u0", text="one"), ClaimUnit(unit_id="u1", text="two")],
        probe_source_count=1,
    )
    router = HybridRouter(RoutingSettings(), GenerationSettings(), llm=SafetyStopLLM(error))
    with pytest.raises(type(error), match=str(error)):
        await router.route("run", Strategy.ADAPTIVE, claim_features)
