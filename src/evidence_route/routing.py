from __future__ import annotations

import json
from typing import Literal

from pydantic import ConfigDict, Field

from evidence_route.artifacts import BillingStateError
from evidence_route.budget import BudgetExceeded, UsageUnavailable
from evidence_route.config import GenerationSettings, RoutingSettings, stable_hash
from evidence_route.contracts import ClaimFeatures, RouteDecision, Strategy, StrictModel
from evidence_route.llm import BillingUncertain


class StructuredCallError(RuntimeError):
    """A router call failed in a way that permits a conservative fallback."""


class RouterPayload(StrictModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal["single", "multi"]
    reason_codes: list[str] = Field(min_length=1)
    explanation: str = Field(min_length=1, max_length=300)


class HybridRouter:
    def __init__(
        self,
        settings: RoutingSettings,
        generation: GenerationSettings,
        *,
        llm: object | None,
        deterministic_ambiguous: bool = False,
    ) -> None:
        self.settings = settings
        self.generation = generation
        self.llm = llm
        self.deterministic_ambiguous = deterministic_ambiguous
        self.config_hash = stable_hash(settings.model_dump(mode="json"))

    async def route(
        self,
        run_id: str,
        strategy: Strategy,
        features: ClaimFeatures,
        *,
        allow_repair: bool = True,
        force_llm: bool = False,
    ) -> RouteDecision:
        if strategy == Strategy.ALWAYS_MULTI:
            return self._decision("multi", "strategy", ["fixed_multi"])
        if strategy == Strategy.ALWAYS_SINGLE:
            return self._decision("single", "strategy", ["fixed_single"])
        if not force_llm and self._clear_multi(features):
            return self._decision("multi", "rule", ["compound_or_conflict"])
        if not force_llm and self._clear_single(features):
            return self._decision("single", "rule", ["atomic_with_sources"])
        if not force_llm and self.deterministic_ambiguous:
            return self._decision("multi", "rule", ["deterministic_ambiguous_multi"])
        if self.llm is None:
            return self._decision("multi", "fallback", ["router_fallback"])
        try:
            result = await self.llm.invoke(
                run_id=run_id,
                node="router",
                task_id="root",
                messages=self._messages(features),
                schema=RouterPayload,
                max_input_tokens=self.generation.router.max_input_tokens,
                max_output_tokens=self.generation.router.max_output_tokens,
                allow_repair=allow_repair,
            )
            payload = result.value
            return self._decision(payload.route, "llm", payload.reason_codes, payload.explanation)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(
                exc, (BudgetExceeded, UsageUnavailable, BillingUncertain, BillingStateError)
            ):
                raise
            return self._decision("multi", "fallback", ["router_fallback"])

    def _decision(
        self,
        route: Literal["single", "multi"],
        source: Literal["rule", "llm", "fallback", "strategy"],
        reason_codes: list[str],
        explanation: str | None = None,
    ) -> RouteDecision:
        return RouteDecision(
            route=route,
            source=source,
            reason_codes=reason_codes,
            explanation=explanation or ", ".join(reason_codes),
            config_hash=self.config_hash,
        )

    def _clear_multi(self, features: ClaimFeatures) -> bool:
        return features.atomic_clause_count >= self.settings.clear_multi_clauses or (
            features.atomic_clause_count >= 2
            and (
                features.has_comparison
                or features.time_scope_count >= 2
                or features.probe_conflict_hint
            )
        )

    def _clear_single(self, features: ClaimFeatures) -> bool:
        return (
            features.atomic_clause_count == 1
            and not features.has_comparison
            and not features.has_causal
            and not features.has_contrast
            and not features.probe_conflict_hint
            and features.probe_source_count >= self.settings.clear_single_min_sources
        )

    @staticmethod
    def _messages(features: ClaimFeatures) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Choose single or multi conservatively and return only the "
                    "supplied JSON schema."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    features.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
                ),
            },
        ]
