from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence_route.artifacts import RunCallSummary, SQLiteRunStore
from evidence_route.budget import PriceConfig
from evidence_route.config import GenerationSettings, load_app_config, stable_hash
from evidence_route.evaluation.activity import (
    CallBounds,
    CallProfile,
    CampaignWorkItem,
    FreezeIdentity,
    Strategy,
    derive_run_id,
)
from evidence_route.evaluation.production import (
    GraphCampaignExecutor,
    build_calibration_runtime_case,
    build_campaign_plan,
    validate_gate_a_cohorts,
)
from evidence_route.evaluation.production_evaluation import estimate_campaign_batch_startup
from evidence_route.evaluation.runner import estimate_call_bounds
from evidence_route.execution import load_price_config
from evidence_route.llm import RawCompletion

pytest_plugins = ["tests.fixtures.evaluation.factories"]


def _freeze() -> FreezeIdentity:
    return FreezeIdentity(
        manifest_freeze_git_sha="1" * 40,
        dev_protocol_git_sha="2" * 40,
        calibration_runtime_manifest_sha256="a" * 64,
        dev_runtime_manifest_sha256="b" * 64,
        stability_runtime_manifest_sha256="c" * 64,
        corpus_preparation_receipt_sha256="d" * 64,
        prompt_bundle_sha256="e" * 64,
        config_sha256="f" * 64,
        pricing_sha256="0" * 64,
        endpoint_config_sha256="3" * 64,
        requirements_lock_sha256="4" * 64,
        requested_alias="relay-model",
        seed=20260817,
    )


def _bounds() -> CallBounds:
    return CallBounds(
        base_call_upper_bound=1544,
        repair_upper_bound=3088,
        fault_upper_bound=9264,
        base_cost_micro_cny=1,
        repair_cost_micro_cny=2,
        fault_cost_micro_cny=6,
        reserve_basis_points=2000,
        startup_required_micro_cny=2,
    )


def test_campaign_plan_uses_exact_stability_manifest_order() -> None:
    dev = [SimpleNamespace(claim_id=f"dev-{index}") for index in range(80)]
    stability = [
        dev[index]
        for index in (79, 3, 41, 7, 19, 22, 28, 31, 35, 39, 44, 48, 52, 56, 60, 64, 68, 72, 76, 0)
    ]
    validate_gate_a_cohorts(dev, stability)
    plan = build_campaign_plan(
        dev,
        stability,
        activity_id="activity",
        campaign_id="campaign",
        freeze=_freeze(),
        cap_micro_cny=350_000_000,
        call_bounds=_bounds(),
    )

    assert len(plan.schedule) == 280
    assert len([item for item in plan.schedule if item.phase == "dev"]) == 240
    stability_items = [item for item in plan.schedule if item.phase == "stability"]
    assert [item.claim_id for item in stability_items[::2]] == [item.claim_id for item in stability]
    assert all(item.strategy is Strategy.ADAPTIVE for item in stability_items)
    assert [link.claim_id for link in plan.stability_repeat_zero_links] == [
        item.claim_id for item in stability
    ]


def test_campaign_batch_startup_cost_uses_strategy_specific_node_bounds() -> None:
    dev = [SimpleNamespace(claim_id=f"dev-{index}") for index in range(80)]
    stability = dev[:20]
    plan = build_campaign_plan(
        dev,
        stability,
        activity_id="activity",
        campaign_id="campaign",
        freeze=_freeze(),
        cap_micro_cny=350_000_000,
        call_bounds=_bounds(),
    )
    pricing = PriceConfig(
        provider="fixture",
        currency="CNY",
        input_per_million=1,
        output_per_million=2,
        price_source="fixture",
        strict_evaluation=True,
    )
    expected = estimate_call_bounds(
        CallProfile(router=1, single=3, decomposer=2, worker=6, judge=2),
        GenerationSettings(),
        pricing,
        reserve_ratio=0.2,
    ).startup_required_micro_cny

    assert estimate_campaign_batch_startup(
        plan.schedule[:3],
        generation=GenerationSettings(),
        pricing=pricing,
        reserve_ratio=0.2,
        include_multi_recovery=True,
    ) == expected


@pytest.mark.parametrize(
    ("stability_count", "bad_id"),
    [(19, None), (21, None), (20, "dev-999")],
)
def test_validate_gate_a_cohorts_rejects_wrong_or_foreign_stability(
    stability_count: int, bad_id: str | None
) -> None:
    dev = [SimpleNamespace(claim_id=f"dev-{index}") for index in range(80)]
    stability = [dev[index] for index in range(min(stability_count, 20))]
    if stability_count == 21:
        stability.append(dev[20])
    if bad_id is not None:
        stability[-1] = SimpleNamespace(claim_id=bad_id)
    with pytest.raises(ValueError, match="stability"):
        validate_gate_a_cohorts(dev, stability)


def test_campaign_plan_fingerprint_changes_with_frozen_routing_config() -> None:
    dev = [SimpleNamespace(claim_id=f"dev-{index}") for index in range(80)]
    stability = dev[:20]
    first = build_campaign_plan(
        dev,
        stability,
        activity_id="activity",
        campaign_id="campaign",
        freeze=_freeze(),
        cap_micro_cny=350_000_000,
        call_bounds=_bounds(),
    )
    changed = _freeze().model_copy(update={"config_sha256": stable_hash({"routing": 1})})
    second = build_campaign_plan(
        dev,
        stability,
        activity_id="activity",
        campaign_id="campaign",
        freeze=changed,
        cap_micro_cny=350_000_000,
        call_bounds=_bounds(),
    )
    assert first.campaign_fingerprint != second.campaign_fingerprint


class _Transport:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **request: object) -> RawCompletion:
        self.calls += 1
        assert request["schema"].__name__ == "VerdictDraft"
        return RawCompletion(
            content=json.dumps(
                {
                    "verdict": "Supported",
                    "confidence": 0.95,
                    "rationale": "The frozen source supports the claim.",
                    "citations": [
                        {
                            "evidence_id": "av:dev:0:0:0",
                            "claim_unit_ids": ["u0"],
                            "question": "What does the source say?",
                            "answer": "It supports the claim.",
                            "quote": "Sean Connery never sent the letter to Steve Jobs.",
                            "stance": "supports",
                            "source_url": "https://example.org/cnet",
                        }
                    ],
                }
            ),
            response_model_id_raw="relay-model-a",
            input_tokens=12,
            output_tokens=8,
        )


@pytest.mark.asyncio
async def test_graph_campaign_executor_returns_accounted_artifact_and_reuses_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "test-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "relay-model")
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        "provider: fake\ncurrency: CNY\ninput_per_million: 1\n"
        "output_per_million: 1\nprice_source: fixture\nstrict_evaluation: true\n",
        encoding="utf-8",
    )
    config = load_app_config(Path("configs/default.yaml"))
    pricing = load_price_config(pricing_path)
    run_store = SQLiteRunStore(
        tmp_path / "run-store.sqlite3",
        activity_id="activity",
        cap_cny=350,
        pricing=pricing,
    )
    transport = _Transport()
    run_id = derive_run_id("campaign", "dev", "dev-0", Strategy.ALWAYS_SINGLE, 0)
    work = CampaignWorkItem(
        order=0,
        phase="dev",
        claim_id="dev-0",
        strategy=Strategy.ALWAYS_SINGLE,
        repeat=0,
        run_id=run_id,
    )
    executor = GraphCampaignExecutor(
        activity_id="activity",
        campaign_id="campaign",
        claims={"dev-0": "Sean Connery sent the letter to Steve Jobs."},
        app_config=config,
        pricing=pricing,
        run_store=run_store,
        corpus_dir=Path("tests/fixtures/averitec/corpora"),
        checkpoint_db=tmp_path / "checkpoints.sqlite3",
        trace_dir=tmp_path / "traces",
        transport=transport,
    )

    first = await executor(work)
    second = await executor(work)

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.actual_cost_micro_cny is not None
    assert first.usage.total_tokens == 20
    assert first.response_model_ids_raw == ["relay-model-a"]
    assert first.call_ids == second.call_ids
    assert transport.calls == 1


def _summary(call_id: str, tokens: int, cost: int) -> RunCallSummary:
    return RunCallSummary(
        call_ids=[call_id],
        usage={
            "input_tokens": tokens - 1,
            "output_tokens": 1,
            "total_tokens": tokens,
            "complete": True,
        },
        actual_cost_micro_cny=cost,
        known_actual_cost_micro_cny=cost,
        committed_cost_micro_cny=cost,
        cost_is_lower_bound=False,
        fresh_call_count=1,
        cache_hit_count=0,
        transport_attempts=1,
        requested_aliases=["relay-model"],
        response_model_ids_raw=["relay-model-a"],
        usage_sources=["provider"],
        identity_verified=False,
        billing_uncertain=False,
    )


def test_calibration_runtime_case_uses_only_persisted_accounting(
    calibration_case,
) -> None:
    plan = __import__(
        "evidence_route.evaluation.calibration", fromlist=["build_calibration_plan"]
    ).build_calibration_plan(
        [SimpleNamespace(claim_id=f"train-{index}", split="train") for index in range(32)],
        activity_id="activity",
        manifest_freeze_git_sha="1" * 40,
        runtime_manifest_sha256="a" * 64,
        corpus_preparation_receipt_sha256="b" * 64,
        prompt_bundle_sha256="c" * 64,
        config_sha256="d" * 64,
        pricing_sha256="e" * 64,
        endpoint_config_sha256="f" * 64,
        requirements_lock_sha256="0" * 64,
        requested_alias="relay-model",
        seed=20260817,
        cap_micro_cny=350_000_000,
    )
    work = plan.items[0]
    case = build_calibration_runtime_case(
        work=work,
        runtime_manifest_sha256=plan.runtime_manifest_sha256,
        features=calibration_case.runtime.features,
        saved_llm_route="single",
        router_summary=_summary("router-call", 5, 7),
        single_result=calibration_case.runtime.single_result,
        single_summary=_summary("single-call", 10, 11),
        multi_result=calibration_case.runtime.multi_result,
        multi_summary=_summary("multi-call", 20, 23),
        requested_alias="relay-model",
        price_config_id="9" * 64,
        request_sha256_by_call_id={
            "router-call": "1" * 64,
            "single-call": "2" * 64,
            "multi-call": "3" * 64,
        },
    )

    assert case.call_ids == ["router-call", "single-call", "multi-call"]
    assert case.router_usage.total_tokens == 5
    assert case.router_actual_cost_micro_cny == 7
    assert case.single_result.estimated_cost_micro_cny == 11
    assert case.multi_result.estimated_cost_micro_cny == 23
    assert case.response_model_ids_raw == ["relay-model-a"]


def test_calibration_runtime_case_rejects_model_drift(calibration_case) -> None:
    from evidence_route.evaluation.calibration import CalibrationWorkItem

    work = CalibrationWorkItem(
        order=0,
        claim_id="train-0",
        case_id="a" * 64,
        router_run_id="b" * 64,
        single_run_id="c" * 64,
        multi_run_id="d" * 64,
    )
    drifted = _summary("multi-call", 20, 23).model_copy(
        update={"response_model_ids_raw": ["relay-model-b"]}
    )
    with pytest.raises(ValueError, match="model drift"):
        build_calibration_runtime_case(
            work=work,
            runtime_manifest_sha256="e" * 64,
            features=calibration_case.runtime.features,
            saved_llm_route="single",
            router_summary=_summary("router-call", 5, 7),
            single_result=calibration_case.runtime.single_result,
            single_summary=_summary("single-call", 10, 11),
            multi_result=calibration_case.runtime.multi_result,
            multi_summary=drifted,
            requested_alias="relay-model",
            price_config_id="9" * 64,
            request_sha256_by_call_id={
                "router-call": "1" * 64,
                "single-call": "2" * 64,
                "multi-call": "3" * 64,
            },
        )
