from pathlib import Path

import pytest

from evidence_route.artifacts import BillingStateError
from evidence_route.budget import PriceConfig
from evidence_route.config import GenerationSettings
from evidence_route.contracts import Strategy
from evidence_route.evaluation.activity import CampaignState, CampaignStatus, WorkStatus
from evidence_route.evaluation.runner import (
    CampaignProcessInterruption,
    CampaignRunner,
    build_dev_schedule,
    compute_gate_a_call_profile,
    estimate_call_bounds,
)

pytest_plugins = ["tests.fixtures.evaluation.campaign_factory"]


def test_gate_a_profile_is_1544_calls() -> None:
    profile = compute_gate_a_call_profile(32, 80, 20, 2)
    assert profile.model_dump() == {
        "router": 152,
        "single": 232,
        "decomposer": 232,
        "worker": 696,
        "judge": 232,
    }
    assert profile.total == 1544


def test_call_bounds_include_repair_fault_and_reserve() -> None:
    pricing = PriceConfig(
        provider="relay",
        currency="CNY",
        input_per_million=1,
        output_per_million=2,
        price_source="fixture",
        strict_evaluation=True,
    )
    bounds = estimate_call_bounds(
        compute_gate_a_call_profile(32, 80, 20, 2),
        GenerationSettings(),
        pricing,
        reserve_ratio=0.2,
    )
    assert bounds.base_call_upper_bound == 1544
    assert bounds.repair_upper_bound == 3088
    assert bounds.fault_upper_bound == 9264
    assert bounds.startup_required_micro_cny == (bounds.base_cost_micro_cny * 120 + 99) // 100


def test_schedule_is_deterministic_cyclic_and_interleaved() -> None:
    claims = [{"claim_id": f"dev-{i}"} for i in range(4)]
    schedule = build_dev_schedule(claims, seed=20260817, campaign_id="campaign")
    base = [item.strategy for item in schedule[:3]]
    assert set(base) == set(Strategy)
    for offset in range(0, len(schedule), 3):
        group = schedule[offset : offset + 3]
        assert len({item.claim_id for item in group}) == 1
        claim_index = offset // 3
        assert [item.strategy for item in group] == (
            base[claim_index % 3 :] + base[: claim_index % 3]
        )
    assert schedule == build_dev_schedule(claims, seed=20260817, campaign_id="campaign")


@pytest.mark.asyncio
async def test_runner_resumes_after_injected_interruption(tmp_path: Path, campaign_factory) -> None:
    first = campaign_factory.executor(fail_after=2)
    with pytest.raises(RuntimeError, match="injected interruption"):
        await CampaignRunner(tmp_path, first).run(campaign_factory.plan())
    second = campaign_factory.executor()
    state = await CampaignRunner(tmp_path, second).resume(
        expected_identity=campaign_factory.freeze_identity()
    )
    assert state.status is CampaignStatus.COMPLETE
    assert not {item.run_id for item in first.completed} & {
        item.run_id for item in second.completed
    }


@pytest.mark.asyncio
async def test_runner_marks_model_drift_and_leaves_pending_tail(
    tmp_path: Path, campaign_factory
) -> None:
    executor = campaign_factory.executor(model_ids=["relay-a", "relay-b"])
    state = await CampaignRunner(tmp_path, executor).run(campaign_factory.plan())
    assert state.status is CampaignStatus.INCOMPLETE_MODEL_DRIFT
    assert state.items[1].status is WorkStatus.STOPPED
    assert all(item.status is WorkStatus.PENDING for item in state.items[2:])


@pytest.mark.asyncio
async def test_runner_marks_budget_and_skips_tail(tmp_path: Path, campaign_factory) -> None:
    executor = campaign_factory.executor(exceed_budget_after=1)
    state = await CampaignRunner(tmp_path, executor).run(campaign_factory.plan())
    assert state.status is CampaignStatus.INCOMPLETE_BUDGET
    assert all(item.status is WorkStatus.NOT_RUN_BUDGET for item in state.items[1:])


@pytest.mark.asyncio
async def test_billing_uncertainty_persists_reloadable_diagnostic(
    tmp_path: Path, campaign_factory
) -> None:
    async def executor(_item):
        raise BillingStateError("sent call has unknown billing")

    state = await CampaignRunner(tmp_path, executor).run(campaign_factory.plan())
    reloaded = CampaignState.model_validate_json(
        (tmp_path / "campaign.json").read_text(encoding="utf-8")
    )
    assert state.status is CampaignStatus.INCOMPLETE_COST_UNCERTAIN
    assert reloaded.items[0].status is WorkStatus.STOPPED
    assert reloaded.items[0].artifact_sha256 is not None


@pytest.mark.asyncio
async def test_only_linked_dev_artifacts_become_stability_baselines(
    tmp_path: Path, campaign_factory
) -> None:
    state = await CampaignRunner(tmp_path, campaign_factory.executor()).run(
        campaign_factory.plan()
    )
    linked = {
        link.claim_id for link in campaign_factory.plan().stability_repeat_zero_links
    }
    assert set(state.stability_repeat_zero_artifact_sha256s) == linked


@pytest.mark.asyncio
async def test_untyped_runtime_error_is_persisted_as_internal_failure(
    tmp_path: Path, campaign_factory
) -> None:
    async def executor(_item):
        raise RuntimeError("bug in graph adapter")

    with pytest.raises(RuntimeError, match="bug in graph adapter"):
        await CampaignRunner(tmp_path, executor).run(campaign_factory.plan())
    state = CampaignState.model_validate_json(
        (tmp_path / "campaign.json").read_text(encoding="utf-8")
    )
    assert state.status is CampaignStatus.FAILED
    assert state.stop_reason is not None
    assert state.stop_reason.value == "internal_error"


@pytest.mark.asyncio
async def test_typed_process_interruption_remains_resumable(
    tmp_path: Path, campaign_factory
) -> None:
    async def executor(_item):
        raise CampaignProcessInterruption("worker stopped")

    with pytest.raises(CampaignProcessInterruption, match="worker stopped"):
        await CampaignRunner(tmp_path, executor).run(campaign_factory.plan())
    state = CampaignState.model_validate_json(
        (tmp_path / "campaign.json").read_text(encoding="utf-8")
    )
    assert state.status is CampaignStatus.INTERRUPTED
