from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_route.contracts import ResultStatus, Usage, Verdict, VerificationResult
from evidence_route.evaluation.activity import (
    FreezeMismatch,
    LatencyBreakdown,
    artifact_fingerprint,
)
from evidence_route.evaluation.experiment import (
    StabilityExperimentIdentity,
    create_experiment_identity,
    materialize_calibration_assets,
    validate_parent_baseline,
    verify_repeat_zero_reuse,
)
from evidence_route.evaluation.lifecycle import sha256_file


def _write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> dict[str, Path]:
    return {
        "report": _write(
            tmp_path / "parent-report.json",
            json.dumps({"activity_id": "gate-a", "campaign_id": "gate-a-dev"}),
        ),
        "manifest": _write(tmp_path / "manifest.json", '{"items": []}'),
        "pricing": _write(tmp_path / "pricing.yaml", "id: local\n"),
        "config": _write(tmp_path / "config.yaml", "llm: {}\n"),
        "prompt": _write(tmp_path / "prompts.py", "PROMPT_VERSION = 'v1'\n"),
        "stability": _write(tmp_path / "stability.json", '{"items": []}'),
    }


def _identity(tmp_path: Path) -> tuple[StabilityExperimentIdentity, dict[str, Path]]:
    inputs = _inputs(tmp_path)
    identity = create_experiment_identity(
        experiment_id="stability-hardening-20260901",
        activity_id="stability-hardening-20260901",
        campaign_id="stability-hardening-20260901",
        parent_activity_id="gate-a",
        parent_report=inputs["report"],
        parent_manifest=inputs["manifest"],
        pricing=inputs["pricing"],
        config=inputs["config"],
        prompt=inputs["prompt"],
        stability_manifest=inputs["stability"],
        repeat_schedule={"claim-a": [0, 1, 2]},
        requested_alias="gpt-5.6-sol",
        response_model_id="gpt-5.6-sol",
        identity_verified=False,
        output_dir=tmp_path / "experiment",
    )
    return identity, inputs


def test_create_experiment_identity_persists_parent_hashes_and_schedule(tmp_path: Path) -> None:
    identity, inputs = _identity(tmp_path)

    assert identity.parent_activity_id == "gate-a"
    assert identity.parent_report_sha256 == sha256_file(inputs["report"])
    assert identity.parent_manifest_sha256 == sha256_file(inputs["manifest"])
    assert identity.parent_pricing_sha256 == sha256_file(inputs["pricing"])
    assert identity.parent_config_sha256 == sha256_file(inputs["config"])
    assert identity.parent_prompt_sha256 == sha256_file(inputs["prompt"])
    assert identity.stability_manifest_sha256 == sha256_file(inputs["stability"])
    assert identity.repeat_schedule == {"claim-a": [0, 1, 2]}
    assert (tmp_path / "experiment" / "experiment.json").is_file()


def test_create_experiment_identity_uses_prompt_hash_from_parent_report(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    parent_prompt_hash = "a" * 64
    inputs["report"].write_text(
        json.dumps(
            {
                "activity_id": "gate-a",
                "campaign_id": "gate-a-dev",
                "reproducibility": {"prompt_bundle_sha256": parent_prompt_hash},
            }
        ),
        encoding="utf-8",
    )

    identity = create_experiment_identity(
        experiment_id="stability-hardening-20260901",
        activity_id="stability-hardening-20260901",
        campaign_id="stability-hardening-20260901",
        parent_activity_id="gate-a",
        parent_report=inputs["report"],
        parent_manifest=inputs["manifest"],
        pricing=inputs["pricing"],
        config=inputs["config"],
        prompt=inputs["prompt"],
        stability_manifest=inputs["stability"],
        repeat_schedule={"claim-a": [0, 1, 2]},
        requested_alias="gpt-5.6-sol",
        response_model_id="gpt-5.6-sol",
        identity_verified=False,
        output_dir=tmp_path / "experiment",
    )

    assert identity.parent_prompt_sha256 == parent_prompt_hash
    assert identity.prompt_sha256 == sha256_file(inputs["prompt"])


def test_create_experiment_identity_separates_parent_and_experiment_config(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    parent_config = _write(tmp_path / "parent-config.yaml", "llm: {model: baseline}\n")
    inputs["config"].write_text("llm: {model: hardened}\n", encoding="utf-8")

    identity = create_experiment_identity(
        experiment_id="stability-hardening-20260901",
        activity_id="stability-hardening-20260901",
        campaign_id="stability-hardening-20260901",
        parent_activity_id="gate-a",
        parent_report=inputs["report"],
        parent_manifest=inputs["manifest"],
        pricing=inputs["pricing"],
        parent_config=parent_config,
        config=inputs["config"],
        prompt=inputs["prompt"],
        stability_manifest=inputs["stability"],
        repeat_schedule={"claim-a": [0, 1, 2]},
        requested_alias="gpt-5.6-sol",
        response_model_id="gpt-5.6-sol",
        identity_verified=False,
        output_dir=tmp_path / "experiment",
    )

    assert identity.parent_config_sha256 == sha256_file(parent_config)
    assert identity.config_sha256 == sha256_file(inputs["config"])


def test_validate_parent_baseline_checks_parent_and_experiment_config(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    parent_config = _write(tmp_path / "parent-config.yaml", "llm: {model: baseline}\n")
    inputs["config"].write_text("llm: {model: hardened}\n", encoding="utf-8")
    identity = create_experiment_identity(
        experiment_id="stability-hardening-20260901",
        activity_id="stability-hardening-20260901",
        campaign_id="stability-hardening-20260901",
        parent_activity_id="gate-a",
        parent_report=inputs["report"],
        parent_manifest=inputs["manifest"],
        pricing=inputs["pricing"],
        parent_config=parent_config,
        config=inputs["config"],
        prompt=inputs["prompt"],
        stability_manifest=inputs["stability"],
        repeat_schedule={"claim-a": [0, 1, 2]},
        requested_alias="gpt-5.6-sol",
        response_model_id="gpt-5.6-sol",
        identity_verified=False,
        output_dir=tmp_path / "experiment",
    )

    validate_parent_baseline(
        identity,
        parent_report=inputs["report"],
        parent_manifest=inputs["manifest"],
        pricing=inputs["pricing"],
        parent_config=parent_config,
        config=inputs["config"],
        prompt=inputs["prompt"],
        stability_manifest=inputs["stability"],
    )

    parent_config.write_text("llm: {model: changed}\n", encoding="utf-8")
    with pytest.raises(FreezeMismatch):
        validate_parent_baseline(
            identity,
            parent_report=inputs["report"],
            parent_manifest=inputs["manifest"],
            pricing=inputs["pricing"],
            parent_config=parent_config,
            config=inputs["config"],
            prompt=inputs["prompt"],
            stability_manifest=inputs["stability"],
        )


@pytest.mark.parametrize(
    "key", ["report", "manifest", "pricing", "config", "prompt", "stability"]
)
def test_validate_parent_baseline_rejects_changed_parent_file(tmp_path: Path, key: str) -> None:
    identity, inputs = _identity(tmp_path)
    inputs[key].write_text(inputs[key].read_text(encoding="utf-8") + "changed", encoding="utf-8")

    with pytest.raises(FreezeMismatch):
        validate_parent_baseline(
            identity,
            parent_report=inputs["report"],
            parent_manifest=inputs["manifest"],
            pricing=inputs["pricing"],
            config=inputs["config"],
            prompt=inputs["prompt"],
            stability_manifest=inputs["stability"],
        )


def test_experiment_output_cannot_be_created_inside_published_parent_report_dir(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path / "reports")
    published = inputs["report"].parent / "evidence-route-gate-a-20260830-clean1"
    published.mkdir()
    parent_report = _write(published / "report.json", '{"activity_id": "gate-a"}')

    with pytest.raises(ValueError, match="published Gate A"):
        create_experiment_identity(
            experiment_id="bad",
            activity_id="bad",
            campaign_id="bad",
            parent_activity_id="gate-a",
            parent_report=parent_report,
            parent_manifest=inputs["manifest"],
            pricing=inputs["pricing"],
            config=inputs["config"],
            prompt=inputs["prompt"],
            stability_manifest=inputs["stability"],
            repeat_schedule={"claim-a": [0, 1, 2]},
            requested_alias="gpt-5.6-sol",
            response_model_id="gpt-5.6-sol",
            identity_verified=False,
            output_dir=published / "new-experiment",
        )


def test_repeat_zero_reuse_requires_recorded_artifact_hash(tmp_path: Path) -> None:
    identity, _ = _identity(tmp_path)
    identity = identity.model_copy(
        update={"repeat_zero_artifact_sha256s": {"claim-a": "a" * 64}}
    )
    artifact = _write(
        tmp_path / "claim-a.json",
        json.dumps({"artifact_sha256": "b" * 64}),
    )

    with pytest.raises(FreezeMismatch):
        verify_repeat_zero_reuse(identity, {"claim-a": artifact})


def test_repeat_zero_reuse_accepts_legacy_artifact_without_fallback_marker(
    tmp_path: Path,
) -> None:
    identity, _ = _identity(tmp_path)
    result = VerificationResult(
        claim_id="claim-a",
        status=ResultStatus.COMPLETED,
        verdict=Verdict.SUPPORTED,
        confidence=0.9,
        rationale="supported",
        initial_route="single",
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
        estimated_cost_micro_cny=0,
        cost_currency="CNY",
        price_config_id="local",
    )
    payload = {
        "schema_version": "1",
        "activity_id": "activity",
        "campaign_id": "campaign",
        "phase": "dev",
        "run_id": "a" * 64,
        "claim_id": "claim-a",
        "strategy": "always_single",
        "repeat": 0,
        "result": result.model_dump(mode="json"),
        "call_ids": ["call-a"],
        "usage": result.usage.model_dump(mode="json"),
        "usage_source": "provider",
        "actual_cost_micro_cny": 0,
        "known_actual_cost_micro_cny": 0,
        "committed_cost_micro_cny": 0,
        "cost_is_lower_bound": False,
        "fresh_call_count": 1,
        "cache_hit_count": 0,
        "requested_alias": "alias",
        "response_model_ids_raw": ["relay"],
        "identity_verified": False,
        "billing_uncertain": False,
        "diagnostic_only": False,
        "latency": LatencyBreakdown(
            fresh_end_to_end_ms=1,
            model_active_ms=1,
            retry_ms=0,
            queue_ms=0,
            checkpoint_downtime_ms=0,
            total_elapsed_ms=1,
            interruption_count=0,
        ).model_dump(mode="json"),
        "artifact_sha256": "0" * 64,
    }
    payload["result"].pop("fallback_used", None)
    payload["artifact_sha256"] = artifact_fingerprint(payload)
    artifact = _write(tmp_path / "claim-a.json", json.dumps(payload))
    identity = identity.model_copy(
        update={"repeat_zero_artifact_sha256s": {"claim-a": payload["artifact_sha256"]}}
    )

    verify_repeat_zero_reuse(identity, {"claim-a": artifact})


def test_materialize_calibration_assets_copies_parent_without_rewriting_it(
    tmp_path: Path,
) -> None:
    parent = Path("artifacts/evaluation/evidence-route-gate-a-20260830-clean1")
    target = tmp_path / "experiment"
    before = {
        name: (parent / name).read_bytes()
        for name in ("calibration-plan.json", "calibration-state.json", "calibration-replay.json")
    }
    parent_activity_before = (parent / "activity.json").read_bytes()

    materialize_calibration_assets(
        parent_activity_dir=parent,
        experiment_activity_dir=target,
        experiment_activity_id="stability-hardening-test",
        parent_activity_id="evidence-route-gate-a-20260830-clean1",
    )

    assert all((target / name).read_bytes() == payload for name, payload in before.items())
    assert (target / "calibration" / "runtime-cases.jsonl").is_file()
    activity = json.loads((target / "activity.json").read_text(encoding="utf-8"))
    assert activity["activity_id"] == "stability-hardening-test"
    assert activity["calibration_status"] == "complete"
    assert (parent / "activity.json").read_bytes() == parent_activity_before
