"""Complete, deterministic report fixture built from persisted campaign artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evidence_route.artifacts import atomic_write_json
from evidence_route.contracts import ResultStatus, Strategy, Usage, Verdict, VerificationResult
from evidence_route.evaluation.activity import (
    ActivityRecord,
    CalibrationArtifactLink,
    CallBounds,
    CampaignItemState,
    CampaignPlan,
    CampaignState,
    CampaignStatus,
    FreezeIdentity,
    LatencyBreakdown,
    RunArtifact,
    WorkStatus,
    artifact_fingerprint,
    campaign_fingerprint,
    summarize_run_artifacts,
)
from evidence_route.evaluation.reporting import ReportInput
from evidence_route.evaluation.runner import build_campaign_schedule


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze() -> FreezeIdentity:
    return FreezeIdentity(
        manifest_freeze_git_sha="a" * 40,
        dev_protocol_git_sha="b" * 40,
        calibration_runtime_manifest_sha256="c" * 64,
        dev_runtime_manifest_sha256="d" * 64,
        stability_runtime_manifest_sha256="e" * 64,
        corpus_preparation_receipt_sha256="f" * 64,
        prompt_bundle_sha256="1" * 64,
        config_sha256="2" * 64,
        pricing_sha256="3" * 64,
        endpoint_config_sha256="4" * 64,
        requirements_lock_sha256="5" * 64,
        requested_alias="fixture-alias",
        seed=20260817,
    )


def _artifact(plan: CampaignPlan, item, index: int) -> RunArtifact:
    verdict = list(Verdict)[int(item.claim_id.split("-")[1]) % len(Verdict)]
    token_total = {
        Strategy.ALWAYS_SINGLE: 100,
        Strategy.ALWAYS_MULTI: 220,
        Strategy.ADAPTIVE: 130,
    }[item.strategy]
    if item.phase == "stability":
        token_total = 132 + item.repeat
    usage = Usage(
        input_tokens=token_total - 20,
        output_tokens=20,
        total_tokens=token_total,
        complete=True,
    )
    escalated = item.strategy is Strategy.ADAPTIVE and index % 11 == 0
    result = VerificationResult(
        claim_id=item.claim_id,
        status=ResultStatus.COMPLETED,
        verdict=verdict,
        confidence=0.82,
        rationale="fixture result",
        citations=[],
        available_evidence_ids=[],
        initial_route="multi" if item.strategy is Strategy.ALWAYS_MULTI else "single",
        escalated=escalated,
        usage=usage,
        estimated_cost_micro_cny=token_total,
        cost_currency="CNY",
        price_config_id="9" * 64,
        latency_ms=40 + index % 7,
    )
    call_id = hashlib.sha256(f"call\0{item.run_id}".encode()).hexdigest()
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
        "call_ids": [call_id],
        "usage": usage.model_dump(mode="json"),
        "usage_source": "provider",
        "actual_cost_micro_cny": token_total,
        "known_actual_cost_micro_cny": token_total,
        "committed_cost_micro_cny": token_total,
        "cost_is_lower_bound": False,
        "fresh_call_count": 1,
        "cache_hit_count": index % 3,
        "requested_alias": plan.freeze.requested_alias,
        "response_model_ids_raw": ["relay-model-a"],
        "identity_verified": False,
        "billing_uncertain": False,
        "diagnostic_only": False,
        "latency": LatencyBreakdown(
            fresh_end_to_end_ms=40 + index % 7,
            model_active_ms=30,
            retry_ms=4,
            queue_ms=6,
            checkpoint_downtime_ms=0,
            total_elapsed_ms=40 + index % 7,
            interruption_count=0,
        ).model_dump(mode="json"),
        "artifact_sha256": "0" * 64,
    }
    payload["artifact_sha256"] = artifact_fingerprint(payload)
    return RunArtifact.model_validate(payload)


class ReportInputFactory:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.activity_dir = root / "activity"
        self.artifact_dir = self.activity_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True)
        claims = [{"claim_id": f"dev-{index}"} for index in range(80)]
        schedule, links = build_campaign_schedule(
            claims,
            stability_runtime_claims=claims[:20],
            seed=20260817,
            campaign_id="dev-campaign",
        )
        plan_payload: dict[str, object] = {
            "schema_version": "1",
            "activity_id": "gate-a",
            "campaign_id": "dev-campaign",
            "freeze": _freeze().model_dump(mode="json"),
            "cap_micro_cny": 350_000_000,
            "call_bounds": CallBounds(
                base_call_upper_bound=1544,
                repair_upper_bound=3088,
                fault_upper_bound=9264,
                base_cost_micro_cny=1_000_000,
                repair_cost_micro_cny=2_000_000,
                fault_cost_micro_cny=6_000_000,
                reserve_basis_points=2000,
                startup_required_micro_cny=1_200_000,
            ).model_dump(mode="json"),
            "schedule": [item.model_dump(mode="json") for item in schedule],
            "stability_repeat_zero_links": links,
        }
        plan_payload["campaign_fingerprint"] = campaign_fingerprint(plan_payload)
        plan = CampaignPlan.model_validate(plan_payload)
        artifacts = [_artifact(plan, item, index) for index, item in enumerate(schedule)]
        states: list[CampaignItemState] = []
        for item, artifact in zip(schedule, artifacts, strict=True):
            relpath = f"artifacts/{artifact.run_id}.json"
            atomic_write_json(
                self.activity_dir / relpath,
                artifact.model_dump(mode="json"),
            )
            states.append(
                CampaignItemState(
                    run_id=item.run_id,
                    status=WorkStatus.COMPLETED,
                    artifact_relpath=relpath,
                    artifact_sha256=artifact.artifact_sha256,
                )
            )
        adaptive_sha = {
            item.claim_id: artifact.artifact_sha256
            for item, artifact in zip(schedule, artifacts, strict=True)
            if item.phase == "dev"
            and item.strategy is Strategy.ADAPTIVE
            and item.claim_id in {link["claim_id"] for link in links}
        }
        summary = summarize_run_artifacts(artifacts)
        state = CampaignState(
            schema_version="1",
            activity_id=plan.activity_id,
            campaign_id=plan.campaign_id,
            status=CampaignStatus.COMPLETE,
            items=states,
            stability_repeat_zero_artifact_sha256s=adaptive_sha,
            observed_response_model_ids_raw=["relay-model-a"],
            billing_uncertain=False,
            summary=summary,
        )
        atomic_write_json(self.activity_dir / "plan.json", plan.model_dump(mode="json"))
        atomic_write_json(self.activity_dir / "campaign.json", state.model_dump(mode="json"))
        activity = ActivityRecord(
            schema_version="1",
            activity_id=plan.activity_id,
            status=CampaignStatus.COMPLETE,
            calibration_status=CampaignStatus.COMPLETE,
            dev_status=CampaignStatus.COMPLETE,
            stability_status=CampaignStatus.COMPLETE,
            calibration_plan_sha256="6" * 64,
            calibration_state_sha256="7" * 64,
            calibration_artifacts=[
                CalibrationArtifactLink(
                    case_id=hashlib.sha256(f"case-{index}".encode()).hexdigest(),
                    artifact_sha256=hashlib.sha256(f"artifact-{index}".encode()).hexdigest(),
                )
                for index in range(32)
            ],
            freeze=plan.freeze,
            campaign_plan_sha256=_sha(self.activity_dir / "plan.json"),
            campaign_state_sha256=_sha(self.activity_dir / "campaign.json"),
            observed_response_model_ids_raw=["relay-model-a"],
            identity_verified=False,
            billing_uncertain=False,
            summary=summary,
        )
        atomic_write_json(self.activity_dir / "activity.json", activity.model_dump(mode="json"))
        self._write_gold()
        calibration_report = root / "reports" / "calibration" / "calibration_report.json"
        atomic_write_json(
            calibration_report,
            {"candidates": [{"rank": index} for index in range(54)], "selected": {"rank": 0}},
        )

    def _write_gold(self) -> None:
        payload = {
            "schema_version": "1",
            "dataset": "AVeriTeC",
            "revision": "8" * 40,
            "source_metadata_sha256": "a" * 64,
            "runtime_manifest_sha256": "d" * 64,
            "seed": 20260817,
            "items": [
                {
                    "claim_id": f"dev-{index}",
                    "original_id": index,
                    "claim": f"Claim {index}",
                    "label": list(Verdict)[index % len(Verdict)].value,
                    "questions": [],
                    "justification": "fixture",
                    "claim_types": [],
                }
                for index in range(80)
            ],
        }
        path = self.root / "gold.json"
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        path.write_bytes(encoded)
        path.with_suffix(".json.sha256").write_text(
            hashlib.sha256(encoded).hexdigest() + "\n",
            encoding="ascii",
        )

    def report_input(self) -> ReportInput:
        return ReportInput(
            repository_root=self.root,
            activity_dir=self.activity_dir,
            gold_manifest=self.root / "gold.json",
            nltk_data_root=self.root / "nltk",
        )

    def with_status(self, status: CampaignStatus) -> ReportInput:
        path = self.activity_dir / "activity.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = status.value
        if status is not CampaignStatus.COMPLETE:
            payload["dev_status"] = status.value
        atomic_write_json(path, payload)
        return self.report_input()


@pytest.fixture
def report_input_factory(tmp_path: Path) -> ReportInputFactory:
    return ReportInputFactory(tmp_path)


@pytest.fixture
def complete_report_input(report_input_factory: ReportInputFactory) -> ReportInput:
    return report_input_factory.report_input()
