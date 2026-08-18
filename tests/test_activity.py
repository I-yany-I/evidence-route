import pytest

from evidence_route.evaluation.activity import (
    CampaignStatus,
    CampaignStopReason,
    FreezeMismatch,
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
    ],
)
def test_stop_reason_precedence_and_status_mapping(reason, status) -> None:
    assert derive_campaign_status([], stop_reason=reason) is status


def test_first_changed_freeze_field_is_reported() -> None:
    error = FreezeMismatch("config_sha256", expected="a", actual="b")
    assert error.field == "config_sha256"
    assert "config_sha256" in str(error)
