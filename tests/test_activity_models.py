from __future__ import annotations

import pytest
from pydantic import ValidationError

from evidence_route.contracts import ResultStatus, Strategy
from evidence_route.evaluation.activity import (
    CallBounds,
    CampaignItemState,
    CampaignStatus,
    CampaignStopReason,
    CampaignWorkItem,
    FreezeIdentity,
    FreezeMismatch,
    WorkStatus,
    campaign_fingerprint,
    derive_campaign_status,
    derive_run_id,
    result_status_to_work_status,
)

SHA = "a" * 64
GIT = "b" * 40


def freeze() -> FreezeIdentity:
    return FreezeIdentity(
        manifest_freeze_git_sha=GIT,
        dev_protocol_git_sha="c" * 40,
        calibration_runtime_manifest_sha256=SHA,
        dev_runtime_manifest_sha256="d" * 64,
        stability_runtime_manifest_sha256="e" * 64,
        corpus_preparation_receipt_sha256="f" * 64,
        prompt_bundle_sha256="1" * 64,
        config_sha256="2" * 64,
        pricing_sha256="3" * 64,
        endpoint_config_sha256="4" * 64,
        requirements_lock_sha256="5" * 64,
        requested_alias="relay-alias",
        seed=20260817,
    )


def bounds() -> CallBounds:
    return CallBounds(
        base_call_upper_bound=10,
        repair_upper_bound=20,
        fault_upper_bound=60,
        base_cost_micro_cny=100,
        repair_cost_micro_cny=200,
        fault_cost_micro_cny=600,
        reserve_basis_points=2000,
        startup_required_micro_cny=120,
    )


def item(*, order: int = 0, phase: str = "dev", repeat: int = 0) -> CampaignWorkItem:
    claim_id = f"claim-{order}"
    strategy = Strategy.ADAPTIVE
    campaign_id = "campaign"
    return CampaignWorkItem(
        order=order,
        phase=phase,
        claim_id=claim_id,
        strategy=strategy,
        repeat=repeat,
        run_id=derive_run_id(campaign_id, phase, claim_id, strategy, repeat),
    )


def test_freeze_identity_rejects_noncanonical_hashes() -> None:
    with pytest.raises(ValidationError):
        FreezeIdentity(**{**freeze().model_dump(), "config_sha256": "A" * 64})


def test_run_id_and_campaign_fingerprint_are_deterministic() -> None:
    run_id = derive_run_id("campaign", "dev", "claim-0", Strategy.ADAPTIVE, 0)
    assert run_id == derive_run_id("campaign", "dev", "claim-0", "adaptive", 0)
    assert len(run_id) == 64
    plan_payload = {
        "schema_version": "1",
        "activity_id": "activity",
        "campaign_id": "campaign",
        "freeze": freeze().model_dump(mode="json"),
        "cap_micro_cny": 1000,
        "call_bounds": bounds().model_dump(mode="json"),
        "schedule": [item()],
        "stability_repeat_zero_links": [],
    }
    first = campaign_fingerprint(plan_payload)
    second = campaign_fingerprint({**plan_payload, "campaign_fingerprint": "0" * 64})
    assert first == second


def test_result_status_mapping_and_campaign_completion() -> None:
    assert result_status_to_work_status(ResultStatus.COMPLETED) is WorkStatus.COMPLETED
    assert result_status_to_work_status(ResultStatus.PARTIAL) is WorkStatus.PARTIAL
    assert result_status_to_work_status(ResultStatus.FAILED) is WorkStatus.FAILED
    terminal = [
        CampaignItemState(
            run_id=derive_run_id("campaign", "dev", f"claim-{i}", "adaptive", 0),
            status=status,
            artifact_relpath=f"{i}.json",
            artifact_sha256=f"{i + 1:x}" * 64,
        )
        for i, status in enumerate(
            [WorkStatus.COMPLETED, WorkStatus.PARTIAL, WorkStatus.FAILED]
        )
    ]
    assert derive_campaign_status(terminal) is CampaignStatus.COMPLETE
    assert derive_campaign_status(
        [terminal[0], terminal[1].model_copy(update={"status": WorkStatus.NOT_RUN_BUDGET})]
    ) is CampaignStatus.INCOMPLETE_BUDGET


def test_safety_stop_reason_precedence() -> None:
    items = [
        CampaignItemState(
            run_id=derive_run_id("campaign", "dev", "claim", "adaptive", 0),
            status=WorkStatus.STOPPED,
            artifact_relpath="artifact.json",
            artifact_sha256="9" * 64,
            stop_reason=CampaignStopReason.MODEL_DRIFT,
        )
    ]
    assert derive_campaign_status(items, stop_reasons=[CampaignStopReason.BILLING_UNCERTAIN]) == (
        CampaignStatus.INCOMPLETE_COST_UNCERTAIN
    )


def test_freeze_mismatch_names_first_changed_field() -> None:
    error = FreezeMismatch("config_sha256", expected="a", actual="b")
    assert "config_sha256" in str(error)
