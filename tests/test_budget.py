from decimal import Decimal

import pytest

from evidence_route.budget import (
    BudgetExceeded,
    InMemoryBudgetStore,
    PriceConfig,
    UsageUnavailable,
)
from evidence_route.contracts import Usage


def pricing() -> PriceConfig:
    return PriceConfig(
        provider="relay",
        currency="CNY",
        input_per_million=10.0,
        output_per_million=20.0,
        price_source="relay dashboard captured 2026-08-17",
        strict_evaluation=True,
    )


def test_cost_uses_separate_input_and_output_rates() -> None:
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=500_000,
        total_tokens=1_500_000,
        complete=True,
    )
    assert pricing().estimate(usage) == Decimal("20.0")


def test_strict_ledger_rejects_incomplete_usage() -> None:
    ledger = InMemoryBudgetStore(cap_cny=350.0, pricing=pricing())
    ledger.reserve_call("call-1", max_input_tokens=100, max_output_tokens=20)
    with pytest.raises(UsageUnavailable):
        ledger.record_call(
            "call-1", Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False)
        )
    assert "call-1" in ledger.reservations


def test_reservation_stops_before_crossing_cap() -> None:
    ledger = InMemoryBudgetStore(cap_cny=1.0, pricing=pricing())
    with pytest.raises(BudgetExceeded):
        ledger.reserve_call("call-1", max_input_tokens=100_000, max_output_tokens=100_000)


def test_recording_same_call_is_idempotent() -> None:
    store = InMemoryBudgetStore(cap_cny=350.0, pricing=pricing())
    store.reserve_call("call-1", max_input_tokens=100, max_output_tokens=20)
    usage = Usage(input_tokens=100, output_tokens=20, total_tokens=120, complete=True)
    store.record_call("call-1", usage)
    store.record_call("call-1", usage)
    assert store.recorded_call_ids == {"call-1"}
