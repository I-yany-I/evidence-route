from __future__ import annotations

import pytest
from pydantic import ValidationError

from evidence_route.contracts import (
    ResultStatus,
    Strategy,
    Usage,
    Verdict,
    VerificationResult,
)
from evidence_route.evaluation.activity import (
    ActivityRecord,
    CallBounds,
    CampaignItemState,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    CampaignWorkItem,
    FreezeIdentity,
    FreezeMismatch,
    LatencyBreakdown,
    RunArtifact,
    WorkStatus,
    artifact_fingerprint,
    campaign_fingerprint,
    derive_campaign_status,
    derive_run_id,
    result_status_to_work_status,
)
from evidence_route.execution import build_run_artifact

SHA = "a" * 64
GIT = "b" * 40


def _latency() -> LatencyBreakdown:
    return LatencyBreakdown(
        fresh_end_to_end_ms=10,
        model_active_ms=8,
        retry_ms=0,
        queue_ms=2,
        checkpoint_downtime_ms=0,
        total_elapsed_ms=10,
        interruption_count=0,
    )


def _completed_artifact() -> RunArtifact:
    return build_run_artifact(
        activity_id="activity",
        campaign_id="campaign",
        phase="dev",
        run_id=SHA,
        claim_id="claim",
        strategy=Strategy.ALWAYS_SINGLE,
        repeat=0,
        result=VerificationResult(
            claim_id="claim",
            status=ResultStatus.COMPLETED,
            verdict=Verdict.SUPPORTED,
            confidence=0.9,
            rationale="fixture",
            initial_route="single",
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
        ),
        summary={
            "call_ids": ["call-0"],
            "usage": Usage(input_tokens=10, output_tokens=2, total_tokens=12, complete=True),
            "actual_cost_micro_cny": 42,
            "known_actual_cost_micro_cny": 42,
            "committed_cost_micro_cny": 42,
            "cost_is_lower_bound": False,
            "fresh_call_count": 1,
            "cache_hit_count": 0,
            "requested_aliases": ["alias"],
            "response_model_ids_raw": ["model-id"],
            "billing_uncertain": False,
        },
        requested_alias="alias",
        price_config_id="c" * 64,
        latency=_latency(),
    )


def _resign_artifact(artifact: RunArtifact, **result_updates: object) -> dict[str, object]:
    payload = artifact.model_dump(mode="json")
    result = payload["result"]
    assert isinstance(result, dict)
    result.update(result_updates)
    payload["artifact_sha256"] = artifact_fingerprint(payload)
    return payload


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
        for i, status in enumerate([WorkStatus.COMPLETED, WorkStatus.PARTIAL, WorkStatus.FAILED])
    ]
    assert derive_campaign_status(terminal) is CampaignStatus.COMPLETE
    assert (
        derive_campaign_status(
            [terminal[0], terminal[1].model_copy(update={"status": WorkStatus.NOT_RUN_BUDGET})]
        )
        is CampaignStatus.INCOMPLETE_BUDGET
    )


def test_user_paused_campaign_is_not_complete() -> None:
    item = CampaignItemState(
        run_id=derive_run_id("campaign", "dev", "claim", "adaptive", 0),
        status=WorkStatus.PENDING,
        artifact_relpath="artifact.json",
    )
    assert (
        derive_campaign_status([item], stop_reason=CampaignStopReason.USER_PAUSED)
        is CampaignStatus.PAUSED
    )


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


@pytest.mark.parametrize(
    ("model_type", "payload"),
    [
        (
            CampaignState,
            {
                "schema_version": "1",
                "activity_id": "activity",
                "campaign_id": "campaign",
                "status": CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
                "stop_reason": CampaignStopReason.BILLING_UNCERTAIN,
                "items": [],
                "stability_repeat_zero_artifact_sha256s": {},
                "observed_response_model_ids_raw": [],
                "billing_uncertain": False,
            },
        ),
        (
            ActivityRecord,
            {
                "schema_version": "1",
                "activity_id": "activity",
                "status": CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
                "calibration_status": CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
                "dev_status": CampaignStatus.PLANNED,
                "stability_status": CampaignStatus.PLANNED,
                "calibration_plan_sha256": "1" * 64,
                "calibration_state_sha256": "2" * 64,
                "calibration_artifacts": [
                    {"case_id": f"{index:064x}", "artifact_sha256": None} for index in range(32)
                ],
                "observed_response_model_ids_raw": [],
                "billing_uncertain": False,
                "stop_reason": CampaignStopReason.BILLING_UNCERTAIN,
            },
        ),
    ],
)
def test_billing_stop_reason_requires_billing_uncertain_flag(model_type, payload) -> None:
    with pytest.raises(ValidationError, match="billing"):
        model_type.model_validate(payload)


@pytest.mark.parametrize(
    ("model_type", "payload"),
    [
        (
            CampaignState,
            {
                "schema_version": "1",
                "activity_id": "activity",
                "campaign_id": "campaign",
                "status": CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
                "stop_reason": None,
                "items": [],
                "stability_repeat_zero_artifact_sha256s": {},
                "observed_response_model_ids_raw": [],
                "billing_uncertain": True,
            },
        ),
        (
            ActivityRecord,
            {
                "schema_version": "1",
                "activity_id": "activity",
                "status": CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
                "calibration_status": CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
                "dev_status": CampaignStatus.PLANNED,
                "stability_status": CampaignStatus.PLANNED,
                "calibration_plan_sha256": "1" * 64,
                "calibration_state_sha256": "2" * 64,
                "calibration_artifacts": [
                    {"case_id": f"{index:064x}", "artifact_sha256": None} for index in range(32)
                ],
                "observed_response_model_ids_raw": [],
                "billing_uncertain": True,
                "stop_reason": None,
            },
        ),
    ],
)
def test_billing_uncertain_flag_requires_billing_stop_reason(model_type, payload) -> None:
    with pytest.raises(ValidationError, match="billing"):
        model_type.model_validate(payload)


def test_run_artifact_rejects_resigned_result_usage_mismatch() -> None:
    artifact = _completed_artifact()
    mismatched_usage = Usage(
        input_tokens=11,
        output_tokens=2,
        total_tokens=13,
        complete=True,
    )

    with pytest.raises(ValidationError, match="result usage"):
        RunArtifact.model_validate(
            _resign_artifact(
                artifact,
                usage=mismatched_usage.model_dump(mode="json"),
            )
        )


@pytest.mark.parametrize(
    ("estimated_cost", "cost_currency", "price_config_id"),
    [
        (43, "CNY", "c" * 64),
        (None, None, None),
        (42, "CNY", ""),
    ],
)
def test_run_artifact_rejects_resigned_result_cost_mismatch(
    estimated_cost: int | None,
    cost_currency: str | None,
    price_config_id: str | None,
) -> None:
    artifact = _completed_artifact()

    with pytest.raises(ValidationError, match="result cost"):
        RunArtifact.model_validate(
            _resign_artifact(
                artifact,
                estimated_cost_micro_cny=estimated_cost,
                cost_currency=cost_currency,
                price_config_id=price_config_id,
            )
        )


def test_zero_call_pre_route_failure_keeps_zero_result_accounting() -> None:
    zero = Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=True)
    artifact = build_run_artifact(
        activity_id="activity",
        campaign_id="campaign",
        phase="dev",
        run_id=SHA,
        claim_id="claim",
        strategy=Strategy.ALWAYS_SINGLE,
        repeat=0,
        result=VerificationResult(
            claim_id="claim",
            status=ResultStatus.FAILED,
            rationale="probe unavailable",
            initial_route=None,
            failure_stage="pre_route",
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
            errors=["PROBE_RETRIEVAL_FAILED"],
        ),
        summary={
            "call_ids": [],
            "usage": zero,
            "actual_cost_micro_cny": 0,
            "known_actual_cost_micro_cny": 0,
            "committed_cost_micro_cny": 0,
            "cost_is_lower_bound": False,
            "cache_hit_count": 0,
            "requested_aliases": [],
            "response_model_ids_raw": [],
            "billing_uncertain": False,
        },
        requested_alias="alias",
        price_config_id="c" * 64,
        latency=_latency(),
    )

    assert artifact.usage == artifact.result.usage == zero
    assert artifact.actual_cost_micro_cny == artifact.result.estimated_cost_micro_cny == 0
    assert artifact.result.cost_currency == "CNY"
    assert artifact.result.price_config_id == "c" * 64


def test_incomplete_diagnostic_requires_matching_result_usage_without_cost() -> None:
    incomplete = Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False)
    artifact = build_run_artifact(
        activity_id="activity",
        campaign_id="campaign",
        phase="dev",
        run_id=SHA,
        claim_id="claim",
        strategy=Strategy.ALWAYS_SINGLE,
        repeat=0,
        result=VerificationResult(
            claim_id="claim",
            status=ResultStatus.PARTIAL,
            verdict=Verdict.NOT_ENOUGH_EVIDENCE,
            confidence=0.4,
            rationale="usage unavailable",
            initial_route="single",
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
        ),
        summary={
            "call_ids": ["call-0"],
            "usage": incomplete,
            "actual_cost_micro_cny": None,
            "known_actual_cost_micro_cny": 0,
            "committed_cost_micro_cny": 42,
            "cost_is_lower_bound": True,
            "cache_hit_count": 0,
            "requested_aliases": ["alias"],
            "response_model_ids_raw": ["model-id"],
            "billing_uncertain": False,
            "usage_sources": ["missing"],
        },
        requested_alias="alias",
        price_config_id="c" * 64,
        latency=_latency(),
    )

    assert artifact.diagnostic_only is True
    assert artifact.usage == artifact.result.usage == incomplete
    assert artifact.result.estimated_cost_micro_cny is None
    assert artifact.result.cost_currency is None
    assert artifact.result.price_config_id is None

    mismatched_usage = Usage(
        input_tokens=1,
        output_tokens=0,
        total_tokens=1,
        complete=False,
    )
    with pytest.raises(ValidationError, match="result usage"):
        RunArtifact.model_validate(
            _resign_artifact(
                artifact,
                usage=mismatched_usage.model_dump(mode="json"),
            )
        )


def test_diagnostic_exact_cost_binds_nested_result_cost() -> None:
    artifact = _completed_artifact()
    payload = artifact.model_dump(mode="json")
    payload["diagnostic_only"] = True
    result = payload["result"]
    assert isinstance(result, dict)
    result["estimated_cost_micro_cny"] = 43
    payload["artifact_sha256"] = artifact_fingerprint(payload)

    with pytest.raises(ValidationError, match="result cost"):
        RunArtifact.model_validate(payload)
