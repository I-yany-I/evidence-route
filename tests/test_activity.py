import pytest

from evidence_route.evaluation.activity import (
    CampaignStatus,
    CampaignStopReason,
    FreezeMismatch,
    derive_activity_status,
    derive_campaign_status,
)


@pytest.mark.parametrize(
    ("reason", "status"),
    [
        (CampaignStopReason.BILLING_UNCERTAIN, CampaignStatus.INCOMPLETE_COST_UNCERTAIN),
        (CampaignStopReason.USAGE_MISSING, CampaignStatus.INCOMPLETE_USAGE),
        (CampaignStopReason.MODEL_DRIFT, CampaignStatus.INCOMPLETE_MODEL_DRIFT),
        (CampaignStopReason.BUDGET, CampaignStatus.INCOMPLETE_BUDGET),
        (CampaignStopReason.INTERNAL_ERROR, CampaignStatus.FAILED),
        (CampaignStopReason.USER_CANCELLED, CampaignStatus.CANCELLED),
        (CampaignStopReason.PROCESS_INTERRUPTION, CampaignStatus.INTERRUPTED),
        (CampaignStopReason.USER_PAUSED, CampaignStatus.PAUSED),
    ],
)
def test_stop_reason_precedence_and_status_mapping(reason, status) -> None:
    assert derive_campaign_status([], stop_reason=reason) is status


def test_first_changed_freeze_field_is_reported() -> None:
    error = FreezeMismatch("config_sha256", expected="a", actual="b")
    assert error.field == "config_sha256"
    assert "config_sha256" in str(error)


def test_billing_uncertainty_precedes_process_interruption() -> None:
    assert (
        derive_activity_status(
            CampaignStatus.COMPLETE,
            CampaignStatus.INTERRUPTED,
            CampaignStatus.PLANNED,
            stop_reason=CampaignStopReason.PROCESS_INTERRUPTION,
            billing_uncertain=True,
        )
        is CampaignStatus.INCOMPLETE_COST_UNCERTAIN
    )


def test_user_pause_is_lower_priority_than_safety_stops() -> None:
    assert (
        derive_activity_status(
            CampaignStatus.COMPLETE,
            CampaignStatus.PAUSED,
            CampaignStatus.PLANNED,
            stop_reason=CampaignStopReason.USER_PAUSED,
        )
        is CampaignStatus.PAUSED
    )


@pytest.mark.parametrize(
    ("phase_status", "expected"),
    [
        (CampaignStatus.INCOMPLETE_USAGE, CampaignStatus.INCOMPLETE_USAGE),
        (CampaignStatus.INCOMPLETE_MODEL_DRIFT, CampaignStatus.INCOMPLETE_MODEL_DRIFT),
        (CampaignStatus.INCOMPLETE_BUDGET, CampaignStatus.INCOMPLETE_BUDGET),
        (CampaignStatus.FAILED, CampaignStatus.FAILED),
        (CampaignStatus.CANCELLED, CampaignStatus.CANCELLED),
    ],
)
def test_activity_phase_safety_status_precedes_process_interruption(
    phase_status: CampaignStatus,
    expected: CampaignStatus,
) -> None:
    assert (
        derive_activity_status(
            CampaignStatus.COMPLETE,
            phase_status,
            CampaignStatus.INTERRUPTED,
            stop_reason=CampaignStopReason.PROCESS_INTERRUPTION,
        )
        is expected
    )


def test_activity_internal_error_precedes_cancel_and_interruption() -> None:
    assert (
        derive_activity_status(
            CampaignStatus.COMPLETE,
            CampaignStatus.CANCELLED,
            CampaignStatus.FAILED,
        )
        is CampaignStatus.FAILED
    )
