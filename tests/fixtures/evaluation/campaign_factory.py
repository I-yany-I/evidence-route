"""Small deterministic campaign fixtures used by runner tests.

The fixture deliberately returns immutable ``RunArtifact`` values rather than mocking the
runner's internals.  Tests can therefore exercise persistence, identity and status mapping using
the same validators as a real graph adapter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evidence_route.budget import BudgetExceeded
from evidence_route.contracts import ResultStatus, Usage, Verdict, VerificationResult
from evidence_route.evaluation.activity import (
    CallBounds,
    CampaignPlan,
    CampaignWorkItem,
    FreezeIdentity,
    RunArtifact,
    artifact_fingerprint,
    campaign_fingerprint,
)
from evidence_route.evaluation.runner import CampaignProcessInterruption, build_campaign_schedule

SHA = "a" * 64
GIT = "b" * 40


def _freeze() -> FreezeIdentity:
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


def _bounds() -> CallBounds:
    return CallBounds(
        base_call_upper_bound=1544,
        repair_upper_bound=3088,
        fault_upper_bound=9264,
        base_cost_micro_cny=1544,
        repair_cost_micro_cny=3088,
        fault_cost_micro_cny=9264,
        reserve_basis_points=2000,
        startup_required_micro_cny=1853,
    )


def _artifact(plan: CampaignPlan, item: CampaignWorkItem, model_id: str | None) -> RunArtifact:
    usage = Usage(input_tokens=1, output_tokens=0, total_tokens=1, complete=True)
    result = VerificationResult(
        claim_id=item.claim_id,
        status=ResultStatus.COMPLETED,
        verdict=Verdict.SUPPORTED,
        confidence=0.8,
        rationale="fixture",
        citations=[],
        available_evidence_ids=[],
        initial_route="single",
        usage=usage,
        latency_ms=1,
    )
    payload: dict[str, object] = {
        "schema_version": "1",
        "activity_id": plan.activity_id,
        "campaign_id": plan.campaign_id,
        "phase": item.phase,
        "run_id": item.run_id,
        "claim_id": item.claim_id,
        "strategy": item.strategy.value,
        "repeat": item.repeat,
        "result": result.model_dump(mode="json"),
        "call_ids": [f"call-{item.run_id}"],
        "usage": usage.model_dump(mode="json"),
        "usage_source": "provider",
        "actual_cost_micro_cny": 1,
        "known_actual_cost_micro_cny": 1,
        "committed_cost_micro_cny": 1,
        "cost_is_lower_bound": False,
        "fresh_call_count": 1,
        "cache_hit_count": 0,
        "requested_alias": plan.freeze.requested_alias,
        "response_model_ids_raw": [] if model_id is None else [model_id],
        "identity_verified": False,
        "billing_uncertain": False,
        "diagnostic_only": False,
        "latency": {
            "fresh_end_to_end_ms": 1,
            "model_active_ms": 1,
            "retry_ms": 0,
            "queue_ms": 0,
            "checkpoint_downtime_ms": 0,
            "total_elapsed_ms": 1,
            "interruption_count": 0,
        },
        "artifact_sha256": "0" * 64,
    }
    payload["artifact_sha256"] = artifact_fingerprint(payload)
    return RunArtifact.model_validate(payload)


class FixtureExecutor:
    def __init__(
        self,
        plan: CampaignPlan,
        *,
        fail_after: int | None = None,
        exceed_budget_after: int | None = None,
        model_ids: list[str] | None = None,
    ) -> None:
        self.plan = plan
        self.fail_after = fail_after
        self.exceed_budget_after = exceed_budget_after
        self.model_ids = model_ids or ["relay-a"]
        self.calls = 0
        self.completed: list[CampaignWorkItem] = []
        self.fresh_network_run_ids: set[str] = set()

    async def __call__(self, item: CampaignWorkItem) -> RunArtifact:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise CampaignProcessInterruption("injected interruption")
        if self.exceed_budget_after is not None and self.calls > self.exceed_budget_after:
            raise BudgetExceeded("fixture budget")
        model_id = self.model_ids[min(self.calls - 1, len(self.model_ids) - 1)]
        self.completed.append(item)
        self.fresh_network_run_ids.add(item.run_id)
        return _artifact(self.plan, item, model_id)


class CampaignFactory:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.activity_id = "gate-a"
        self.campaign_id = "campaign"
        claims = [{"claim_id": f"dev-{index}"} for index in range(21)]
        schedule, links = build_campaign_schedule(
            claims, seed=20260817, campaign_id=self.campaign_id, stability_claims=20
        )
        payload = {
            "schema_version": "1",
            "activity_id": self.activity_id,
            "campaign_id": self.campaign_id,
            "freeze": _freeze().model_dump(mode="json"),
            "cap_micro_cny": 1_000_000,
            "call_bounds": _bounds().model_dump(mode="json"),
            "schedule": [item.model_dump(mode="json") for item in schedule],
            "stability_repeat_zero_links": links,
        }
        payload["campaign_fingerprint"] = campaign_fingerprint(payload)
        self._plan = CampaignPlan.model_validate(payload)

    def plan(self) -> CampaignPlan:
        return self._plan

    def freeze_identity(self) -> FreezeIdentity:
        return self._plan.freeze

    def executor(self, **kwargs: object) -> FixtureExecutor:
        return FixtureExecutor(self._plan, **kwargs)


@pytest.fixture
def runtime_claims() -> list[dict[str, str]]:
    return [{"claim_id": f"dev-{index}"} for index in range(20)]


@pytest.fixture
def campaign_factory(tmp_path: Path) -> CampaignFactory:
    return CampaignFactory(tmp_path)
