import hashlib
import json
from pathlib import Path

import pytest

from evidence_route.artifacts import BillingStateError, SQLiteRunStore, atomic_write_json
from evidence_route.budget import PriceConfig, UsageUnavailable
from evidence_route.config import GenerationSettings
from evidence_route.contracts import Strategy
from evidence_route.evaluation.activity import (
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    FreezeMismatch,
    RunArtifact,
    WorkStatus,
    artifact_fingerprint,
)
from evidence_route.evaluation.runner import (
    CampaignProcessInterruption,
    CampaignRunner,
    build_campaign_schedule,
    build_dev_schedule,
    compute_gate_a_call_profile,
    estimate_call_bounds,
)
from evidence_route.execution import build_run_artifact
from evidence_route.llm import BillingUncertain

pytest_plugins = ["tests.fixtures.evaluation.campaign_factory"]


def _load_state(root: Path) -> CampaignState:
    return CampaignState.model_validate_json((root / "campaign.json").read_text(encoding="utf-8"))


def _load_state_payload(root: Path) -> dict[str, object]:
    return json.loads((root / "campaign.json").read_text(encoding="utf-8"))


def _write_state_payload(root: Path, payload: dict[str, object]) -> None:
    atomic_write_json(root / "campaign.json", payload)


async def _persist_interrupted_after_one(
    root: Path,
    campaign_factory,
    *,
    run_store: SQLiteRunStore | None = None,
    executor=None,
) -> None:
    first = executor or campaign_factory.executor(fail_after=1)
    with pytest.raises(CampaignProcessInterruption, match="injected interruption"):
        await CampaignRunner(root, first, run_store=run_store).run(campaign_factory.plan())


def _run_store(path: Path, activity_id: str) -> SQLiteRunStore:
    return SQLiteRunStore(
        path,
        activity_id=activity_id,
        cap_cny=1.0,
        pricing=PriceConfig(
            provider="fixture",
            currency="CNY",
            input_per_million=1,
            output_per_million=1,
            price_source="fixture",
            strict_evaluation=True,
        ),
    )


def _record_artifact_call(
    store: SQLiteRunStore, artifact: RunArtifact, *, suffix: str = ""
) -> None:
    call_id = artifact.call_ids[0] + suffix
    request_sha256 = hashlib.sha256(call_id.encode()).hexdigest()
    store.reserve_call(
        call_id,
        request_sha256=request_sha256,
        run_id=artifact.run_id,
        node="single",
        task_id=f"root{suffix}",
        logical_attempt=0,
        max_input_tokens=artifact.usage.input_tokens,
        max_output_tokens=artifact.usage.output_tokens,
    )
    store.mark_sent(call_id)
    store.complete_call(
        call_id,
        request_sha256=request_sha256,
        payload={"content": "fixture"},
        usage=artifact.usage,
        usage_source="provider",
        requested_alias=artifact.requested_alias,
        response_model_id_raw=artifact.response_model_ids_raw[0],
        identity_verified=False,
    )


def test_gate_a_profile_is_1544_calls_without_recovery() -> None:
    profile = compute_gate_a_call_profile(32, 80, 20, 2)
    assert profile.model_dump() == {
        "router": 152,
        "single": 232,
        "decomposer": 232,
        "worker": 696,
        "judge": 232,
    }
    assert profile.total == 1544


def test_v4_profile_includes_one_single_recovery_per_adaptive_multi_item() -> None:
    profile = compute_gate_a_call_profile(32, 80, 20, 2, include_multi_recovery=True)

    assert profile.single == 352
    assert profile.total == 1664


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
async def test_campaign_runner_pauses_after_item_limit_and_resumes_without_replay(
    tmp_path: Path, campaign_factory
) -> None:
    first_executor = campaign_factory.executor()
    first = CampaignRunner(tmp_path, first_executor)

    paused = await first.run(campaign_factory.plan(), max_items=2)

    assert paused.status is CampaignStatus.PAUSED
    assert paused.stop_reason is CampaignStopReason.USER_PAUSED
    assert first_executor.calls == 2
    assert [item.status for item in paused.items[:2]] == [
        WorkStatus.COMPLETED,
        WorkStatus.COMPLETED,
    ]
    assert all(item.status is WorkStatus.PENDING for item in paused.items[2:])

    second_executor = campaign_factory.executor()
    resumed = await CampaignRunner(tmp_path, second_executor).resume(max_items=2)

    assert resumed.status is CampaignStatus.PAUSED
    assert resumed.stop_reason is CampaignStopReason.USER_PAUSED
    assert second_executor.calls == 2
    assert all(
        item.status is WorkStatus.COMPLETED for item in resumed.items[:4]
    )


@pytest.mark.asyncio
async def test_repeated_resume_with_ledger_does_not_duplicate_completed_items(
    tmp_path: Path, campaign_factory
) -> None:
    store = _run_store(tmp_path / "run-store.sqlite3", campaign_factory.plan().activity_id)
    executors = [campaign_factory.executor() for _ in range(3)]

    async def record(executor, item):
        artifact = await executor(item)
        _record_artifact_call(store, artifact)
        return artifact

    first = await CampaignRunner(
        tmp_path,
        lambda item: record(executors[0], item),
        run_store=store,
    ).run(campaign_factory.plan(), max_items=2)
    second = await CampaignRunner(
        tmp_path,
        lambda item: record(executors[1], item),
        run_store=store,
    ).resume(max_items=2)
    third = await CampaignRunner(
        tmp_path,
        lambda item: record(executors[2], item),
        run_store=store,
    ).resume(max_items=2)

    assert first.status is CampaignStatus.PAUSED
    assert second.status is CampaignStatus.PAUSED
    assert third.status is CampaignStatus.PAUSED
    assert [executor.calls for executor in executors] == [2, 2, 2]
    assert sum(
        len(store.summarize_run(item.run_id).call_ids)
        for item in third.items[:6]
    ) == 6
    assert all(item.status is WorkStatus.COMPLETED for item in third.items[:6])


@pytest.mark.asyncio
async def test_campaign_runner_rejects_non_positive_item_limit(
    tmp_path: Path, campaign_factory
) -> None:
    with pytest.raises(ValueError, match="max_items"):
        await CampaignRunner(tmp_path, campaign_factory.executor()).run(
            campaign_factory.plan(), max_items=0
        )


def test_campaign_schedule_uses_frozen_stability_manifest_order() -> None:
    dev_claims = [{"claim_id": f"dev-{index}"} for index in range(21)]
    stability_claims = [
        {"claim_id": claim_id}
        for claim_id in ["dev-20", *[f"dev-{index}" for index in range(1, 20)]]
    ]

    schedule, links = build_campaign_schedule(
        dev_claims,
        stability_runtime_claims=stability_claims,
        seed=20260817,
        campaign_id="campaign",
    )

    expected = [claim["claim_id"] for claim in stability_claims]
    assert [link["claim_id"] for link in links] == expected
    assert [item.claim_id for item in schedule[len(dev_claims) * 3 :: 2]] == expected


@pytest.mark.parametrize("count", [19, 21])
def test_campaign_schedule_requires_exactly_twenty_stability_claims(count: int) -> None:
    dev_claims = [{"claim_id": f"dev-{index}"} for index in range(21)]

    with pytest.raises(ValueError, match="exactly 20"):
        build_campaign_schedule(
            dev_claims,
            stability_runtime_claims=dev_claims[:count],
            seed=20260817,
        )


def test_campaign_schedule_rejects_duplicate_stability_claim_ids() -> None:
    dev_claims = [{"claim_id": f"dev-{index}"} for index in range(21)]
    stability_claims = [*dev_claims[:19], dev_claims[0]]

    with pytest.raises(ValueError, match="unique"):
        build_campaign_schedule(
            dev_claims,
            stability_runtime_claims=stability_claims,
            seed=20260817,
        )


def test_campaign_schedule_rejects_stability_claim_missing_from_dev() -> None:
    dev_claims = [{"claim_id": f"dev-{index}"} for index in range(20)]
    stability_claims = [*dev_claims[:19], {"claim_id": "outside-dev"}]

    with pytest.raises(ValueError, match="subset"):
        build_campaign_schedule(
            dev_claims,
            stability_runtime_claims=stability_claims,
            seed=20260817,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["count", "order", "run_id", "relpath"])
async def test_resume_rejects_state_items_that_do_not_match_plan(
    tmp_path: Path, campaign_factory, mutation: str
) -> None:
    await _persist_interrupted_after_one(tmp_path, campaign_factory)
    payload = _load_state_payload(tmp_path)
    items = payload["items"]
    assert isinstance(items, list)
    if mutation == "count":
        items.pop()
    elif mutation == "order":
        items[1], items[2] = items[2], items[1]
    elif mutation == "run_id":
        items[1]["run_id"] = "f" * 64
    else:
        items[1]["artifact_relpath"] = "artifacts/unexpected.json"
    _write_state_payload(tmp_path, payload)
    executor = campaign_factory.executor()

    with pytest.raises(ValueError, match="campaign state items"):
        await CampaignRunner(tmp_path, executor).resume(
            expected_identity=campaign_factory.freeze_identity()
        )

    assert executor.calls == 0


@pytest.mark.asyncio
async def test_resume_rejects_missing_recorded_artifact_before_execution(
    tmp_path: Path, campaign_factory
) -> None:
    await _persist_interrupted_after_one(tmp_path, campaign_factory)
    state = _load_state(tmp_path)
    completed = next(item for item in state.items if item.artifact_sha256 is not None)
    (tmp_path / completed.artifact_relpath).unlink()
    executor = campaign_factory.executor()

    with pytest.raises(FileNotFoundError, match=completed.run_id):
        await CampaignRunner(tmp_path, executor).resume(
            expected_identity=campaign_factory.freeze_identity()
        )

    assert executor.calls == 0


@pytest.mark.asyncio
async def test_resume_rejects_valid_artifact_file_without_state_hash(
    tmp_path: Path, campaign_factory
) -> None:
    await _persist_interrupted_after_one(tmp_path, campaign_factory)
    state = _load_state(tmp_path)
    orphan_item = next(item for item in state.items if item.status is WorkStatus.INTERRUPTED)
    work = next(
        item for item in campaign_factory.plan().schedule if item.run_id == orphan_item.run_id
    )
    artifact = await campaign_factory.executor()(work)
    atomic_write_json(
        tmp_path / orphan_item.artifact_relpath,
        artifact.model_dump(mode="json"),
    )
    executor = campaign_factory.executor()

    with pytest.raises(FreezeMismatch, match="unlinked_artifact"):
        await CampaignRunner(tmp_path, executor).resume(
            expected_identity=campaign_factory.freeze_identity()
        )

    assert executor.calls == 0


@pytest.mark.asyncio
async def test_resume_rejects_state_artifact_sha_mismatch_before_execution(
    tmp_path: Path, campaign_factory
) -> None:
    await _persist_interrupted_after_one(tmp_path, campaign_factory)
    payload = _load_state_payload(tmp_path)
    items = payload["items"]
    assert isinstance(items, list)
    completed = next(item for item in items if item["artifact_sha256"] is not None)
    completed["artifact_sha256"] = "0" * 64
    _write_state_payload(tmp_path, payload)
    executor = campaign_factory.executor()

    with pytest.raises(FreezeMismatch, match="artifact_sha256"):
        await CampaignRunner(tmp_path, executor).resume(
            expected_identity=campaign_factory.freeze_identity()
        )

    assert executor.calls == 0


@pytest.mark.asyncio
async def test_resume_rejects_artifact_identity_mismatch_before_execution(
    tmp_path: Path, campaign_factory
) -> None:
    await _persist_interrupted_after_one(tmp_path, campaign_factory)
    state_payload = _load_state_payload(tmp_path)
    items = state_payload["items"]
    assert isinstance(items, list)
    completed = next(item for item in items if item["artifact_sha256"] is not None)
    artifact_path = tmp_path / completed["artifact_relpath"]
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_payload["campaign_id"] = "different-campaign"
    artifact_payload["artifact_sha256"] = artifact_fingerprint(artifact_payload)
    completed["artifact_sha256"] = artifact_payload["artifact_sha256"]
    baselines = state_payload["stability_repeat_zero_artifact_sha256s"]
    if artifact_payload["claim_id"] in baselines:
        baselines[artifact_payload["claim_id"]] = artifact_payload["artifact_sha256"]
    atomic_write_json(artifact_path, artifact_payload)
    _write_state_payload(tmp_path, state_payload)
    executor = campaign_factory.executor()

    with pytest.raises(ValueError, match="artifact identity"):
        await CampaignRunner(tmp_path, executor).resume(
            expected_identity=campaign_factory.freeze_identity()
        )

    assert executor.calls == 0


@pytest.mark.asyncio
async def test_resume_rejects_multiple_running_items_as_corruption(
    tmp_path: Path, campaign_factory
) -> None:
    await _persist_interrupted_after_one(tmp_path, campaign_factory)
    payload = _load_state_payload(tmp_path)
    items = payload["items"]
    assert isinstance(items, list)
    for item in items[1:3]:
        item.update(status="running", stop_reason=None, interrupted_at=None, resumed_at=None)
    payload.update(status="running", stop_reason=None)
    _write_state_payload(tmp_path, payload)
    executor = campaign_factory.executor()

    with pytest.raises(ValueError, match="multiple running"):
        await CampaignRunner(tmp_path, executor).resume(
            expected_identity=campaign_factory.freeze_identity()
        )

    assert executor.calls == 0


@pytest.mark.asyncio
async def test_resume_normalizes_one_running_item(tmp_path: Path, campaign_factory) -> None:
    await _persist_interrupted_after_one(tmp_path, campaign_factory)
    payload = _load_state_payload(tmp_path)
    items = payload["items"]
    assert isinstance(items, list)
    items[1].update(status="running", stop_reason=None, interrupted_at=None, resumed_at=None)
    payload.update(status="running", stop_reason=None)
    _write_state_payload(tmp_path, payload)
    executor = campaign_factory.executor()

    state = await CampaignRunner(tmp_path, executor).resume(
        expected_identity=campaign_factory.freeze_identity()
    )

    assert state.status is CampaignStatus.COMPLETE
    assert state.items[1].resumed_at is not None


@pytest.mark.asyncio
async def test_resume_rejects_ambiguous_current_interrupted_items(
    tmp_path: Path, campaign_factory
) -> None:
    await _persist_interrupted_after_one(tmp_path, campaign_factory)
    payload = _load_state_payload(tmp_path)
    items = payload["items"]
    assert isinstance(items, list)
    items[3].update(
        status="interrupted",
        stop_reason="process_interruption",
        interrupted_at=items[1]["interrupted_at"],
        resumed_at=None,
    )
    _write_state_payload(tmp_path, payload)
    executor = campaign_factory.executor()

    with pytest.raises(ValueError, match="multiple resumable interruptions"):
        await CampaignRunner(tmp_path, executor).resume(
            expected_identity=campaign_factory.freeze_identity()
        )

    assert executor.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "status", "billing_uncertain"),
    [
        (
            CampaignStopReason.BILLING_UNCERTAIN,
            CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
            True,
        ),
        (
            CampaignStopReason.USAGE_MISSING,
            CampaignStatus.INCOMPLETE_USAGE,
            False,
        ),
        (
            CampaignStopReason.MODEL_DRIFT,
            CampaignStatus.INCOMPLETE_MODEL_DRIFT,
            False,
        ),
        (
            CampaignStopReason.BUDGET,
            CampaignStatus.INCOMPLETE_BUDGET,
            False,
        ),
        (
            CampaignStopReason.INTERNAL_ERROR,
            CampaignStatus.FAILED,
            False,
        ),
        (
            CampaignStopReason.USER_CANCELLED,
            CampaignStatus.CANCELLED,
            False,
        ),
    ],
)
async def test_resume_keeps_terminal_safety_stop_without_executor_calls(
    tmp_path: Path,
    campaign_factory,
    reason: CampaignStopReason,
    status: CampaignStatus,
    billing_uncertain: bool,
) -> None:
    await _persist_interrupted_after_one(tmp_path, campaign_factory)
    payload = _load_state_payload(tmp_path)
    payload.update(
        status=status.value,
        stop_reason=reason.value,
        billing_uncertain=billing_uncertain,
    )
    _write_state_payload(tmp_path, payload)
    before = (tmp_path / "campaign.json").read_bytes()
    executor = campaign_factory.executor()

    resumed = await CampaignRunner(tmp_path, executor).resume(
        expected_identity=campaign_factory.freeze_identity()
    )

    assert resumed.status is status
    assert resumed.stop_reason is reason
    assert executor.calls == 0
    assert (tmp_path / "campaign.json").read_bytes() == before


@pytest.mark.asyncio
async def test_resume_normalizes_legacy_stop_reason_to_safety_precedence(
    tmp_path: Path, campaign_factory
) -> None:
    await _persist_interrupted_after_one(tmp_path, campaign_factory)
    payload = _load_state_payload(tmp_path)
    items = payload["items"]
    assert isinstance(items, list)
    items[0].update(
        status="stopped",
        stop_reason="usage_missing",
        artifact_sha256=items[0]["artifact_sha256"],
    )
    payload.update(
        status="incomplete_model_drift",
        stop_reason="model_drift",
        billing_uncertain=False,
    )
    _write_state_payload(tmp_path, payload)

    executor = campaign_factory.executor()
    resumed = await CampaignRunner(tmp_path, executor).resume(
        expected_identity=campaign_factory.freeze_identity()
    )

    assert resumed.status is CampaignStatus.INCOMPLETE_USAGE
    assert resumed.stop_reason is CampaignStopReason.USAGE_MISSING
    assert executor.calls == 0


@pytest.mark.asyncio
async def test_running_normalization_does_not_reset_historical_interrupted_item(
    tmp_path: Path, campaign_factory
) -> None:
    await _persist_interrupted_after_one(tmp_path, campaign_factory)
    payload = _load_state_payload(tmp_path)
    items = payload["items"]
    assert isinstance(items, list)
    items[1].update(status="running", stop_reason=None, interrupted_at=None, resumed_at=None)
    items[2].update(
        status="interrupted",
        stop_reason="process_interruption",
        interrupted_at="2026-08-19T00:00:00+00:00",
        resumed_at=None,
    )
    payload.update(status="running", stop_reason=None)
    _write_state_payload(tmp_path, payload)

    async def interrupt_normalized_item(_item):
        raise CampaignProcessInterruption("stop after normalization")

    with pytest.raises(CampaignProcessInterruption, match="stop after normalization"):
        await CampaignRunner(tmp_path, interrupt_normalized_item).resume(
            expected_identity=campaign_factory.freeze_identity()
        )

    reloaded = _load_state(tmp_path)
    historical = reloaded.items[2]
    assert historical.status is WorkStatus.INTERRUPTED
    assert historical.stop_reason is CampaignStopReason.PROCESS_INTERRUPTION
    assert historical.interrupted_at == "2026-08-19T00:00:00+00:00"
    assert historical.resumed_at is None


@pytest.mark.asyncio
async def test_new_artifact_alias_must_match_frozen_plan_before_persistence(
    tmp_path: Path, campaign_factory
) -> None:
    plan = campaign_factory.plan()
    underlying = campaign_factory.executor()
    calls = 0

    async def executor(item):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("identity check must stop before a second work item")
        artifact = await underlying(item)
        payload = artifact.model_dump(mode="json")
        payload["requested_alias"] = "different-alias"
        payload["artifact_sha256"] = artifact_fingerprint(payload)
        return RunArtifact.model_validate(payload)

    with pytest.raises(ValueError, match="artifact identity"):
        await CampaignRunner(tmp_path, executor).run(plan)

    first = plan.schedule[0]
    assert not (tmp_path / f"artifacts/{first.run_id}.json").exists()
    state = _load_state(tmp_path)
    assert state.items[0].status is WorkStatus.RUNNING
    assert state.items[0].artifact_sha256 is None
    assert calls == 1


@pytest.mark.asyncio
async def test_identity_failure_after_completed_call_keeps_ledger_accounting_auditable(
    tmp_path: Path, campaign_factory
) -> None:
    plan = campaign_factory.plan()
    store = _run_store(tmp_path / "run-store.sqlite3", plan.activity_id)
    underlying = campaign_factory.executor()

    async def executor(item):
        artifact = await underlying(item)
        _record_artifact_call(store, artifact)
        payload = artifact.model_dump(mode="json")
        payload["requested_alias"] = "wrong-alias"
        payload["artifact_sha256"] = artifact_fingerprint(payload)
        return RunArtifact.model_validate(payload)

    with pytest.raises(ValueError, match="artifact identity"):
        await CampaignRunner(tmp_path / "campaign", executor, run_store=store).run(plan)

    first = plan.schedule[0]
    assert not (tmp_path / "campaign" / f"artifacts/{first.run_id}.json").exists()
    state = _load_state(tmp_path / "campaign")
    assert state.items[0].status is WorkStatus.RUNNING
    assert state.items[0].artifact_sha256 is None
    summary = store.summarize_run(first.run_id)
    assert summary.call_ids == [f"call-{first.run_id}"]
    assert summary.usage.complete is True


@pytest.mark.asyncio
async def test_artifact_identity_mismatch_precedes_ledger_alias_drift(
    tmp_path: Path, campaign_factory
) -> None:
    plan = campaign_factory.plan()
    store = _run_store(tmp_path / "run-store.sqlite3", plan.activity_id)
    underlying = campaign_factory.executor()

    async def executor(item):
        artifact = await underlying(item)
        _record_artifact_call(store, artifact)
        connection = store._connect()
        try:
            connection.execute(
                "UPDATE calls SET requested_alias = 'ledger-drift' WHERE run_id = ?",
                (item.run_id,),
            )
        finally:
            connection.close()
        payload = artifact.model_dump(mode="json")
        payload["requested_alias"] = "wrong-alias"
        payload["artifact_sha256"] = artifact_fingerprint(payload)
        return RunArtifact.model_validate(payload)

    with pytest.raises(ValueError, match="artifact identity"):
        await CampaignRunner(tmp_path / "campaign", executor, run_store=store).run(plan)

    first = plan.schedule[0]
    assert not (tmp_path / "campaign" / f"artifacts/{first.run_id}.json").exists()
    state = _load_state(tmp_path / "campaign")
    assert state.items[0].status is WorkStatus.RUNNING


@pytest.mark.asyncio
async def test_runner_rejects_executor_artifact_that_disagrees_with_run_store(
    tmp_path: Path, campaign_factory
) -> None:
    plan = campaign_factory.plan()
    store = _run_store(tmp_path / "run-store.sqlite3", plan.activity_id)
    underlying = campaign_factory.executor()

    async def executor(item):
        artifact = await underlying(item)
        _record_artifact_call(store, artifact)
        payload = artifact.model_dump(mode="json")
        payload.update(
            actual_cost_micro_cny=2,
            known_actual_cost_micro_cny=2,
            committed_cost_micro_cny=2,
        )
        payload["result"]["estimated_cost_micro_cny"] = 2
        payload["artifact_sha256"] = artifact_fingerprint(payload)
        return RunArtifact.model_validate(payload)

    with pytest.raises(FreezeMismatch, match="actual_cost_micro_cny"):
        await CampaignRunner(tmp_path / "campaign", executor, run_store=store).run(plan)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unresolved_state", "expected_status", "expected_reason"),
    [
        ("reserved", CampaignStatus.INCOMPLETE_USAGE, CampaignStopReason.USAGE_MISSING),
        (
            "sent",
            CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
            CampaignStopReason.BILLING_UNCERTAIN,
        ),
        (
            "billing_uncertain",
            CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
            CampaignStopReason.BILLING_UNCERTAIN,
        ),
    ],
)
async def test_mixed_completed_and_unresolved_accounting_stops_campaign(
    tmp_path: Path,
    campaign_factory,
    unresolved_state: str,
    expected_status: CampaignStatus,
    expected_reason: CampaignStopReason,
) -> None:
    plan = campaign_factory.plan()
    store = _run_store(tmp_path / "run-store.sqlite3", plan.activity_id)
    underlying = campaign_factory.executor()
    calls = 0

    async def executor(item):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("incomplete accounting must stop after its work item")
        artifact = await underlying(item)
        _record_artifact_call(store, artifact)
        unresolved_id = f"unresolved-{item.run_id}"
        store.reserve_call(
            unresolved_id,
            request_sha256=hashlib.sha256(unresolved_id.encode()).hexdigest(),
            run_id=item.run_id,
            node="worker",
            task_id="unresolved",
            logical_attempt=0,
            max_input_tokens=1,
            max_output_tokens=0,
        )
        if unresolved_state in {"sent", "billing_uncertain"}:
            store.mark_sent(unresolved_id)
        if unresolved_state == "billing_uncertain":
            store.mark_billing_uncertain(unresolved_id)
        return build_run_artifact(
            activity_id=plan.activity_id,
            campaign_id=plan.campaign_id,
            phase=item.phase,
            run_id=item.run_id,
            claim_id=item.claim_id,
            strategy=item.strategy,
            repeat=item.repeat,
            result=artifact.result,
            summary=store.summarize_run(item.run_id),
            requested_alias=plan.freeze.requested_alias,
            price_config_id=store.pricing.config_id,
            latency=artifact.latency,
        )

    state = await CampaignRunner(tmp_path / "campaign", executor, run_store=store).run(plan)

    assert state.status is expected_status
    assert state.stop_reason is expected_reason
    assert state.items[0].status is WorkStatus.STOPPED
    assert calls == 1


@pytest.mark.asyncio
async def test_empty_ledger_cannot_hide_nonzero_artifact_accounting(
    tmp_path: Path, campaign_factory
) -> None:
    plan = campaign_factory.plan()
    store = _run_store(tmp_path / "run-store.sqlite3", plan.activity_id)
    underlying = campaign_factory.executor()

    async def executor(item):
        artifact = await underlying(item)
        payload = artifact.model_dump(mode="json")
        payload["call_ids"] = []
        payload["usage"] = {
            "input_tokens": 1,
            "output_tokens": 0,
            "total_tokens": 1,
            "complete": False,
        }
        payload["result"]["usage"] = payload["usage"]
        payload["usage_source"] = "missing"
        payload["actual_cost_micro_cny"] = None
        payload["known_actual_cost_micro_cny"] = 0
        payload["committed_cost_micro_cny"] = 0
        payload["cost_is_lower_bound"] = True
        payload["fresh_call_count"] = 0
        payload["response_model_ids_raw"] = []
        payload["diagnostic_only"] = True
        payload["result"]["estimated_cost_micro_cny"] = None
        payload["result"]["cost_currency"] = None
        payload["result"]["price_config_id"] = None
        payload["artifact_sha256"] = artifact_fingerprint(payload)
        return RunArtifact.model_validate(payload)

    with pytest.raises(FreezeMismatch, match="zero_call_accounting"):
        await CampaignRunner(tmp_path / "campaign", executor, run_store=store).run(plan)


@pytest.mark.asyncio
async def test_resume_rejects_run_store_rows_missing_from_artifact(
    tmp_path: Path, campaign_factory
) -> None:
    plan = campaign_factory.plan()
    campaign_root = tmp_path / "campaign"
    store = _run_store(tmp_path / "run-store.sqlite3", plan.activity_id)
    underlying = campaign_factory.executor(fail_after=1)

    async def executor(item):
        artifact = await underlying(item)
        _record_artifact_call(store, artifact)
        return artifact

    await _persist_interrupted_after_one(
        campaign_root,
        campaign_factory,
        run_store=store,
        executor=executor,
    )
    state = _load_state(campaign_root)
    completed = next(item for item in state.items if item.artifact_sha256 is not None)
    artifact = RunArtifact.model_validate_json(
        (campaign_root / completed.artifact_relpath).read_text(encoding="utf-8")
    )
    _record_artifact_call(store, artifact, suffix="-late")
    resume_executor = campaign_factory.executor()

    with pytest.raises(FreezeMismatch, match="call_ids"):
        await CampaignRunner(campaign_root, resume_executor, run_store=store).resume(
            expected_identity=campaign_factory.freeze_identity()
        )

    assert resume_executor.calls == 0


@pytest.mark.asyncio
async def test_resume_rejects_run_store_for_another_activity(
    tmp_path: Path, campaign_factory
) -> None:
    await _persist_interrupted_after_one(tmp_path, campaign_factory)
    wrong_store = _run_store(tmp_path / "wrong.sqlite3", "different-activity")
    executor = campaign_factory.executor()

    with pytest.raises(FreezeMismatch, match="run_store.activity_id"):
        await CampaignRunner(tmp_path, executor, run_store=wrong_store).resume(
            expected_identity=campaign_factory.freeze_identity()
        )

    assert executor.calls == 0


@pytest.mark.asyncio
async def test_runner_resumes_after_injected_interruption(tmp_path: Path, campaign_factory) -> None:
    first = campaign_factory.executor(fail_after=2)
    with pytest.raises(RuntimeError, match="injected interruption"):
        await CampaignRunner(tmp_path, first).run(campaign_factory.plan())
    second = campaign_factory.executor()
    state = await CampaignRunner(tmp_path, second).resume(
        expected_identity=campaign_factory.freeze_identity()
    )
    reloaded = CampaignState.model_validate_json(
        (tmp_path / "campaign.json").read_text(encoding="utf-8")
    )
    assert state.status is CampaignStatus.COMPLETE
    assert state.stop_reason is None
    assert reloaded.status is CampaignStatus.COMPLETE
    assert reloaded.stop_reason is None
    assert not {item.run_id for item in first.completed} & {
        item.run_id for item in second.completed
    }


@pytest.mark.asyncio
async def test_resume_reopens_item_that_interrupts_again_after_prior_resume(
    tmp_path: Path, campaign_factory
) -> None:
    async def interrupt(_item):
        raise CampaignProcessInterruption("repeat interruption")

    with pytest.raises(CampaignProcessInterruption, match="repeat interruption"):
        await CampaignRunner(tmp_path, interrupt).run(campaign_factory.plan())
    with pytest.raises(CampaignProcessInterruption, match="repeat interruption"):
        await CampaignRunner(tmp_path, interrupt).resume(
            expected_identity=campaign_factory.freeze_identity()
        )

    retried = _load_state(tmp_path).items[0]
    assert retried.status is WorkStatus.INTERRUPTED
    assert retried.resumed_at is not None

    state = await CampaignRunner(tmp_path, campaign_factory.executor()).resume(
        expected_identity=campaign_factory.freeze_identity()
    )
    assert state.status is CampaignStatus.COMPLETE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unresolved_state", "expected_reason", "expected_status"),
    [
        (
            "sent",
            CampaignStopReason.BILLING_UNCERTAIN,
            CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
        ),
        (
            "billing_uncertain",
            CampaignStopReason.BILLING_UNCERTAIN,
            CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
        ),
        (
            "usage_missing",
            CampaignStopReason.USAGE_MISSING,
            CampaignStatus.INCOMPLETE_USAGE,
        ),
    ],
)
async def test_resume_audits_interrupted_run_ledger_before_executor(
    tmp_path: Path,
    campaign_factory,
    unresolved_state: str,
    expected_reason: CampaignStopReason,
    expected_status: CampaignStatus,
) -> None:
    plan = campaign_factory.plan()
    store = _run_store(tmp_path / "run-store.sqlite3", plan.activity_id)

    async def interrupt_after_ledger_write(work):
        call_id = hashlib.sha256(f"{work.run_id}\0interrupted".encode()).hexdigest()
        request_sha256 = hashlib.sha256(call_id.encode()).hexdigest()
        store.reserve_call(
            call_id,
            request_sha256=request_sha256,
            run_id=work.run_id,
            node="single",
            task_id="root",
            logical_attempt=0,
            max_input_tokens=1,
            max_output_tokens=1,
        )
        store.mark_sent(call_id)
        if unresolved_state == "billing_uncertain":
            store.mark_billing_uncertain(call_id)
        elif unresolved_state == "usage_missing":
            from evidence_route.contracts import Usage

            store.complete_call(
                call_id,
                request_sha256=request_sha256,
                payload={"content": "saved response"},
                usage=Usage(
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    complete=False,
                ),
                usage_source="missing",
                requested_alias=plan.freeze.requested_alias,
                response_model_id_raw="relay-a",
                identity_verified=False,
            )
        raise CampaignProcessInterruption("fixture process kill")

    with pytest.raises(CampaignProcessInterruption, match="process kill"):
        await CampaignRunner(
            tmp_path,
            interrupt_after_ledger_write,
            run_store=store,
        ).run(plan)

    executor = campaign_factory.executor()
    resumed = await CampaignRunner(
        tmp_path,
        executor,
        run_store=store,
    ).resume(expected_identity=campaign_factory.freeze_identity())

    assert resumed.status is expected_status
    assert resumed.stop_reason is expected_reason
    assert resumed.items[0].status is WorkStatus.STOPPED
    assert resumed.items[0].stop_reason is expected_reason
    assert resumed.items[0].artifact_sha256 is not None
    assert executor.calls == 0


@pytest.mark.asyncio
async def test_authorized_billing_recovery_requeues_stopped_item(
    tmp_path: Path, campaign_factory
) -> None:
    plan = campaign_factory.plan()
    store = _run_store(tmp_path / "run-store.sqlite3", plan.activity_id)
    call_id: str | None = None
    request_sha256: str | None = None
    calls = 0

    async def executor(item):
        nonlocal calls, call_id, request_sha256
        calls += 1
        if calls == 1:
            call_id = f"call-{item.run_id}"
            request_sha256 = hashlib.sha256(call_id.encode()).hexdigest()
            store.reserve_call(
                call_id,
                request_sha256=request_sha256,
                run_id=item.run_id,
                node="single",
                task_id="root",
                logical_attempt=0,
                max_input_tokens=1,
                max_output_tokens=1,
            )
            store.mark_sent(call_id)
            store.mark_billing_uncertain(call_id)
            raise BillingUncertain("fixture handoff")
        artifact = await campaign_factory.executor()(item)
        usage = artifact.usage
        store.complete_call(
            call_id,
            request_sha256=request_sha256,
            payload={"content": "fixture"},
            usage=usage,
            usage_source="provider",
            requested_alias=plan.freeze.requested_alias,
            response_model_id_raw="relay-a",
            identity_verified=False,
        )
        return artifact

    runner = CampaignRunner(tmp_path, executor, run_store=store)
    first = await runner.run(plan, max_items=1)
    assert first.stop_reason is CampaignStopReason.BILLING_UNCERTAIN

    assert call_id is not None and request_sha256 is not None
    store.authorize_billing_uncertain_retry(
        call_id,
        reason="fixture recovery authorization",
        evidence="explicit test authorization",
    )
    resumed = await runner.resume_after_billing_recovery(
        expected_identity=campaign_factory.freeze_identity(),
        max_items=1,
        authorized_call_ids={call_id},
    )

    assert resumed.items[0].status is WorkStatus.COMPLETED
    assert resumed.stop_reason is CampaignStopReason.USER_PAUSED
    assert resumed.billing_uncertain is False
    assert calls == 2


@pytest.mark.asyncio
async def test_resume_audits_ledger_before_clearing_interruption(
    tmp_path: Path,
    campaign_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = campaign_factory.plan()
    store = _run_store(tmp_path / "run-store.sqlite3", plan.activity_id)

    async def interrupt_before_call(_work):
        raise CampaignProcessInterruption("injected interruption")

    await _persist_interrupted_after_one(
        tmp_path,
        campaign_factory,
        run_store=store,
        executor=interrupt_before_call,
    )
    original = store.unresolved_call_states
    audited = False

    def assert_persisted_interruption(run_id: str):
        nonlocal audited
        if audited:
            return original(run_id)
        persisted = _load_state(tmp_path)
        current = next(item for item in persisted.items if item.run_id == run_id)
        assert persisted.status is CampaignStatus.INTERRUPTED
        assert persisted.stop_reason is CampaignStopReason.PROCESS_INTERRUPTION
        assert current.status is WorkStatus.INTERRUPTED
        assert current.stop_reason is CampaignStopReason.PROCESS_INTERRUPTION
        audited = True
        return original(run_id)

    monkeypatch.setattr(store, "unresolved_call_states", assert_persisted_interruption)
    fixture_executor = campaign_factory.executor()

    async def accounted_executor(work):
        artifact = await fixture_executor(work)
        _record_artifact_call(store, artifact)
        return artifact

    resumed = await CampaignRunner(
        tmp_path,
        accounted_executor,
        run_store=store,
    ).resume(expected_identity=campaign_factory.freeze_identity())

    assert resumed.status is CampaignStatus.COMPLETE
    assert audited is True


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
async def test_usage_missing_precedes_model_drift_for_one_artifact(
    tmp_path: Path, campaign_factory
) -> None:
    executor = campaign_factory.executor(model_ids=["relay-a", "relay-b"])

    async def usage_missing_executor(work):
        artifact = await executor(work)
        if executor.calls == 2:
            payload = artifact.model_dump(mode="json")
            payload.update(
                usage={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "complete": False,
                },
                usage_source="missing",
                actual_cost_micro_cny=None,
                cost_is_lower_bound=True,
                diagnostic_only=True,
            )
            payload["result"]["usage"] = payload["usage"]
            payload["result"]["estimated_cost_micro_cny"] = None
            payload["result"]["cost_currency"] = None
            payload["result"]["price_config_id"] = None
            payload["artifact_sha256"] = artifact_fingerprint(payload)
            return RunArtifact.model_validate(payload)
        return artifact

    state = await CampaignRunner(tmp_path, usage_missing_executor).run(campaign_factory.plan())

    assert state.status is CampaignStatus.INCOMPLETE_USAGE
    assert state.stop_reason is CampaignStopReason.USAGE_MISSING


def test_campaign_plan_rejects_stability_item_before_dev_items(campaign_factory) -> None:
    payload = campaign_factory.plan().model_dump(mode="json")
    schedule = payload["schedule"]
    assert isinstance(schedule, list)
    schedule[0], schedule[-1] = schedule[-1], schedule[0]
    for order, item in enumerate(schedule):
        item["order"] = order
    payload["campaign_fingerprint"] = __import__(
        "evidence_route.evaluation.activity", fromlist=["campaign_fingerprint"]
    ).campaign_fingerprint(payload)

    with pytest.raises(ValueError, match="phase boundary"):
        type(campaign_factory.plan()).model_validate(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_name", ["plan.json", "campaign.json"])
async def test_fresh_campaign_rejects_half_created_journal(
    tmp_path: Path, campaign_factory, existing_name: str
) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / existing_name).write_text("{}\n", encoding="utf-8")
    executor = campaign_factory.executor()

    with pytest.raises(FileExistsError, match="incomplete campaign journal"):
        await CampaignRunner(tmp_path, executor).run(campaign_factory.plan())

    assert executor.calls == 0
    counterpart = {"plan.json": "campaign.json", "campaign.json": "plan.json"}[existing_name]
    assert not (tmp_path / counterpart).exists()


@pytest.mark.asyncio
async def test_runner_marks_budget_and_skips_tail(tmp_path: Path, campaign_factory) -> None:
    executor = campaign_factory.executor(exceed_budget_after=1)
    state = await CampaignRunner(tmp_path, executor).run(campaign_factory.plan())
    assert state.status is CampaignStatus.INCOMPLETE_BUDGET
    assert all(item.status is WorkStatus.NOT_RUN_BUDGET for item in state.items[1:])


@pytest.mark.asyncio
async def test_budget_overflow_after_persisted_call_keeps_current_diagnostic(
    tmp_path: Path, campaign_factory
) -> None:
    plan = campaign_factory.plan()
    store = SQLiteRunStore(
        tmp_path / "run-store.sqlite3",
        activity_id=plan.activity_id,
        cap_cny=1.0,
        pricing=PriceConfig(
            provider="fixture",
            currency="CNY",
            input_per_million=1_000_000,
            output_per_million=1_000_000,
            price_source="fixture",
            strict_evaluation=True,
        ),
    )
    underlying = campaign_factory.executor()

    async def executor(item):
        artifact = await underlying(item)
        call_id = artifact.call_ids[0]
        request_sha256 = hashlib.sha256(call_id.encode()).hexdigest()
        store.reserve_call(
            call_id,
            request_sha256=request_sha256,
            run_id=item.run_id,
            node="single",
            task_id="root",
            logical_attempt=0,
            max_input_tokens=1,
            max_output_tokens=0,
        )
        store.mark_sent(call_id)
        usage = artifact.usage.model_copy(update={"input_tokens": 2, "total_tokens": 2})
        store.complete_call(
            call_id,
            request_sha256=request_sha256,
            payload={"content": "persisted before budget stop"},
            usage=usage,
            usage_source="provider",
            requested_alias=artifact.requested_alias,
            response_model_id_raw=artifact.response_model_ids_raw[0],
            identity_verified=False,
        )
        raise AssertionError("complete_call must surface the budget overflow")

    campaign_root = tmp_path / "campaign"
    state = await CampaignRunner(campaign_root, executor, run_store=store).run(plan)

    current = state.items[0]
    assert state.status is CampaignStatus.INCOMPLETE_BUDGET
    assert state.stop_reason is CampaignStopReason.BUDGET
    assert current.status is WorkStatus.STOPPED
    assert current.stop_reason is CampaignStopReason.BUDGET
    assert current.artifact_sha256 is not None
    assert all(item.status is WorkStatus.NOT_RUN_BUDGET for item in state.items[1:])
    artifact = RunArtifact.model_validate_json(
        (campaign_root / current.artifact_relpath).read_text(encoding="utf-8")
    )
    assert artifact.diagnostic_only is True
    assert artifact.call_ids == store.summarize_run(current.run_id).call_ids
    assert artifact.result.errors[0] == CampaignStopReason.BUDGET.value.upper()


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
@pytest.mark.parametrize(
    ("error", "reason", "status"),
    [
        (
            UsageUnavailable("provider usage omitted"),
            CampaignStopReason.USAGE_MISSING,
            CampaignStatus.INCOMPLETE_USAGE,
        ),
        (
            BillingUncertain("transport handoff is unresolved"),
            CampaignStopReason.BILLING_UNCERTAIN,
            CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
        ),
    ],
)
async def test_runner_maps_typed_accounting_errors_to_safety_stops(
    tmp_path: Path, campaign_factory, error, reason, status
) -> None:
    async def executor(_item):
        raise error

    state = await CampaignRunner(tmp_path, executor).run(campaign_factory.plan())

    assert state.status is status
    assert state.stop_reason is reason
    assert state.items[0].status is WorkStatus.STOPPED
    assert state.items[0].stop_reason is reason
    assert state.items[0].artifact_sha256 is not None
    artifact = RunArtifact.model_validate_json(
        (tmp_path / state.items[0].artifact_relpath).read_text(encoding="utf-8")
    )
    assert artifact.diagnostic_only is True
    assert artifact.result.errors[0] == reason.value.upper()


@pytest.mark.asyncio
async def test_only_linked_dev_artifacts_become_stability_baselines(
    tmp_path: Path, campaign_factory
) -> None:
    state = await CampaignRunner(tmp_path, campaign_factory.executor()).run(campaign_factory.plan())
    linked = {link.claim_id for link in campaign_factory.plan().stability_repeat_zero_links}
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


@pytest.mark.asyncio
async def test_fresh_artifact_alias_mismatch_is_rejected_before_persisting(
    tmp_path: Path, campaign_factory
) -> None:
    underlying = campaign_factory.executor()

    async def executor(item):
        artifact = await underlying(item)
        payload = artifact.model_dump(mode="json")
        payload["requested_alias"] = "wrong-alias"
        payload["artifact_sha256"] = artifact_fingerprint(payload)
        return RunArtifact.model_validate(payload)

    with pytest.raises(ValueError, match="artifact identity"):
        await CampaignRunner(tmp_path, executor).run(campaign_factory.plan())

    state = CampaignState.model_validate_json(
        (tmp_path / "campaign.json").read_text(encoding="utf-8")
    )
    assert state.items[0].status is WorkStatus.RUNNING
    assert not (tmp_path / "artifacts" / f"{state.items[0].run_id}.json").exists()
