from __future__ import annotations

import hashlib
import json
from decimal import ROUND_CEILING, Decimal
from threading import RLock

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidence_route.contracts import Usage


class BudgetExceeded(RuntimeError):
    """Raised when a new worst-case reservation would exceed the cap."""


class UsageUnavailable(RuntimeError):
    """Raised when a provider does not return complete token usage."""


class PriceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    currency: str
    input_per_million: float | None = Field(default=None, gt=0)
    output_per_million: float | None = Field(default=None, gt=0)
    price_source: str | None = None
    strict_evaluation: bool = False

    @model_validator(mode="after")
    def validate_strict_price(self) -> PriceConfig:
        if self.strict_evaluation and (
            self.input_per_million is None
            or self.output_per_million is None
            or not self.price_source
        ):
            raise ValueError("strict evaluation requires rates and price_source")
        return self

    @property
    def config_id(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def estimate(self, usage: Usage) -> Decimal:
        if not usage.complete:
            raise UsageUnavailable("provider usage is incomplete")
        if self.input_per_million is None or self.output_per_million is None:
            raise UsageUnavailable("pricing is not configured")
        input_cost = (
            Decimal(usage.input_tokens)
            * Decimal(str(self.input_per_million))
            / Decimal(1_000_000)
        )
        output_cost = (
            Decimal(usage.output_tokens)
            * Decimal(str(self.output_per_million))
            / Decimal(1_000_000)
        )
        return input_cost + output_cost

    def estimate_micro_cny(self, usage: Usage) -> int:
        return int(
            (self.estimate(usage) * Decimal(1_000_000)).to_integral_value(
                rounding=ROUND_CEILING
            )
        )


class InMemoryBudgetStore:
    """Deterministic test ledger; production calls use the SQLite store in Task 7."""

    def __init__(self, *, cap_cny: float, pricing: PriceConfig) -> None:
        if cap_cny <= 0:
            raise ValueError("cap_cny must be positive")
        self.cap_micro_cny = int(Decimal(str(cap_cny)) * Decimal(1_000_000))
        self.pricing = pricing
        self.reservations: dict[str, int] = {}
        self.actual_by_call: dict[str, int] = {}
        self._lock = RLock()

    @property
    def recorded_call_ids(self) -> set[str]:
        return set(self.actual_by_call)

    def reserve_call(
        self, call_id: str, *, max_input_tokens: int, max_output_tokens: int
    ) -> int:
        projected = self.pricing.estimate_micro_cny(
            Usage(
                input_tokens=max_input_tokens,
                output_tokens=max_output_tokens,
                total_tokens=max_input_tokens + max_output_tokens,
                complete=True,
            )
        )
        with self._lock:
            if call_id in self.actual_by_call or call_id in self.reservations:
                return self.reservations.get(call_id, self.actual_by_call.get(call_id, 0))
            committed = sum(self.actual_by_call.values()) + sum(self.reservations.values())
            if committed + projected > self.cap_micro_cny:
                raise BudgetExceeded("next call would exceed estimated_cost_cap")
            self.reservations[call_id] = projected
            return projected

    def record_call(self, call_id: str, usage: Usage) -> int:
        actual = self.pricing.estimate_micro_cny(usage)
        with self._lock:
            if call_id in self.actual_by_call:
                return self.actual_by_call[call_id]
            self.reservations.pop(call_id, None)
            self.actual_by_call[call_id] = actual
            return actual
