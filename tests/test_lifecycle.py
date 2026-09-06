from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from evidence_route.evaluation.activity import (
    CalibrationArtifactLink,
    CampaignStatus,
    CampaignStopReason,
    FreezeIdentity,
)
from evidence_route.evaluation.lifecycle import (
    ActivityFileError,
    build_freeze_identity,
    build_initial_activity,
    link_calibration_artifact,
    load_activity,
    mark_calibration_complete,
    persist_activity,
    resume_activity_after_billing_recovery,
    resume_activity_phase,
    transition_activity_phase,
    verify_current_freeze,
    verify_git_freeze,
)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _files(tmp_path: Path) -> dict[str, Path]:
    return {
        "calibration": _write(tmp_path / "calibration.json", '{"items":[1]}\n'),
        "dev": _write(tmp_path / "dev.json", '{"items":[2]}\n'),
        "stability": _write(tmp_path / "stability.json", '{"items":[3]}\n'),
        "receipt": _write(tmp_path / "receipt.json", '{"corpus":"frozen"}\n'),
        "retrieval_receipt": _write(
            tmp_path / "retrieval-receipt.json", '{"model":"frozen"}\n'
        ),
        "prompts": _write(tmp_path / "prompts.py", "PROMPT_VERSION = 'x'\n"),
        "config": _write(tmp_path / "config.yaml", "routing:\n  low_confidence: 0.65\n"),
        "pricing": _write(tmp_path / "pricing.yaml", "currency: CNY\ninput: 1\n"),
        "requirements": _write(tmp_path / "requirements.lock", "pydantic==2\n"),
    }


def test_build_freeze_identity_hashes_all_inputs_without_secret(tmp_path: Path) -> None:
    files = _files(tmp_path)
    identity = build_freeze_identity(
        calibration_manifest=files["calibration"],
        dev_manifest=files["dev"],
        stability_manifest=files["stability"],
        corpus_receipt=files["receipt"],
        prompt_bundle=files["prompts"],
        config_file=files["config"],
        pricing_file=files["pricing"],
        requirements_lock=files["requirements"],
        endpoint_config={
            "base_url": "https://relay.example/v1",
            "requested_alias": "relay-model",
            "api_key": "do-not-hash-this",
        },
        requested_alias="relay-model",
        seed=20260817,
        manifest_freeze_git_sha="a" * 40,
        dev_protocol_git_sha="b" * 40,
    )

    assert isinstance(identity, FreezeIdentity)
    assert identity.requested_alias == "relay-model"
    assert identity.seed == 20260817
    assert (
        identity.endpoint_config_sha256
        == hashlib.sha256(b'{"base_url":"https://relay.example/v1"}').hexdigest()
    )
    assert "do-not-hash-this" not in identity.model_dump_json()
    assert (
        identity.calibration_runtime_manifest_sha256
        == hashlib.sha256(files["calibration"].read_bytes()).hexdigest()
    )


def test_freeze_identity_optionally_binds_retrieval_model_receipt(tmp_path: Path) -> None:
    files = _files(tmp_path)
    common = {
        "calibration_manifest": files["calibration"],
        "dev_manifest": files["dev"],
        "stability_manifest": files["stability"],
        "corpus_receipt": files["receipt"],
        "prompt_bundle": files["prompts"],
        "config_file": files["config"],
        "pricing_file": files["pricing"],
        "requirements_lock": files["requirements"],
        "endpoint_config": {"base_url": "https://relay.example/v1"},
        "requested_alias": "relay-model",
        "seed": 20260817,
        "manifest_freeze_git_sha": "a" * 40,
        "dev_protocol_git_sha": "b" * 40,
    }

    legacy = build_freeze_identity(**common)
    bound = build_freeze_identity(
        **common,
        retrieval_model_receipt=files["retrieval_receipt"],
    )

    assert "retrieval_model_receipt_sha256" not in legacy.model_dump(mode="json")
    assert bound.retrieval_model_receipt_sha256 == hashlib.sha256(
        files["retrieval_receipt"].read_bytes()
    ).hexdigest()


def test_verify_current_freeze_reports_first_changed_field(tmp_path: Path) -> None:
    files = _files(tmp_path)
    expected = build_freeze_identity(
        calibration_manifest=files["calibration"],
        dev_manifest=files["dev"],
        stability_manifest=files["stability"],
        corpus_receipt=files["receipt"],
        prompt_bundle=files["prompts"],
        config_file=files["config"],
        pricing_file=files["pricing"],
        requirements_lock=files["requirements"],
        endpoint_config={"base_url": "https://relay.example/v1"},
        requested_alias="relay-model",
        seed=20260817,
        manifest_freeze_git_sha="a" * 40,
        dev_protocol_git_sha="b" * 40,
    )
    files["config"].write_text("routing:\n  low_confidence: 0.55\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config_sha256"):
        verify_current_freeze(
            expected,
            calibration_manifest=files["calibration"],
            dev_manifest=files["dev"],
            stability_manifest=files["stability"],
            corpus_receipt=files["receipt"],
            prompt_bundle=files["prompts"],
            config_file=files["config"],
            pricing_file=files["pricing"],
            requirements_lock=files["requirements"],
            endpoint_config={"base_url": "https://relay.example/v1"},
            requested_alias="relay-model",
            seed=20260817,
            manifest_freeze_git_sha="a" * 40,
            dev_protocol_git_sha="b" * 40,
        )


def test_verify_current_freeze_cannot_bypass_changed_file_with_old_digest(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    expected = build_freeze_identity(
        calibration_manifest=files["calibration"],
        dev_manifest=files["dev"],
        stability_manifest=files["stability"],
        corpus_receipt=files["receipt"],
        prompt_bundle=files["prompts"],
        config_file=files["config"],
        pricing_file=files["pricing"],
        requirements_lock=files["requirements"],
        endpoint_config={"base_url": "https://relay.example/v1"},
        requested_alias="relay-model",
        seed=20260817,
        manifest_freeze_git_sha="a" * 40,
        dev_protocol_git_sha="b" * 40,
    )
    old_config_sha = hashlib.sha256(files["config"].read_bytes()).hexdigest()
    files["config"].write_text("routing:\n  low_confidence: 0.55\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config_sha256"):
        verify_current_freeze(
            expected,
            calibration_manifest=files["calibration"],
            dev_manifest=files["dev"],
            stability_manifest=files["stability"],
            corpus_receipt=files["receipt"],
            prompt_bundle=files["prompts"],
            config_file=files["config"],
            config_sha256=old_config_sha,
            pricing_file=files["pricing"],
            requirements_lock=files["requirements"],
            endpoint_config={"base_url": "https://relay.example/v1"},
            requested_alias="relay-model",
            seed=20260817,
            manifest_freeze_git_sha="a" * 40,
            dev_protocol_git_sha="b" * 40,
        )


def test_activity_persistence_uses_sidecar_and_rejects_tampering(tmp_path: Path) -> None:
    activity = build_initial_activity(
        activity_id="activity-1",
        calibration_plan_sha256="a" * 64,
        calibration_state_sha256="b" * 64,
        case_ids=[f"{index:064x}" for index in range(32)],
    )
    path = tmp_path / "activity.json"
    digest = persist_activity(path, activity, fresh=True)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii") == digest + "\n"
    assert load_activity(path) == activity

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dev_status"] = CampaignStatus.RUNNING.value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ActivityFileError, match="sidecar SHA-256 mismatch"):
        load_activity(path)


def test_initial_activity_requires_exactly_ordered_unique_cases(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="32 unique"):
        build_initial_activity(
            activity_id="activity-1",
            calibration_plan_sha256="a" * 64,
            calibration_state_sha256="b" * 64,
            case_ids=["a" * 64] * 32,
        )


def test_activity_phase_transition_preserves_model_ids_and_stop_reason() -> None:
    activity = build_initial_activity(
        activity_id="activity-1",
        calibration_plan_sha256="a" * 64,
        calibration_state_sha256="b" * 64,
        case_ids=[f"{index:064x}" for index in range(32)],
    )
    for index, link in enumerate(activity.calibration_artifacts):
        activity = link_calibration_artifact(
            activity, case_id=link.case_id, artifact_sha256=f"{index + 1:064x}"
        )
    activity = mark_calibration_complete(activity)
    activity = transition_activity_phase(
        activity,
        phase="calibration",
        status=CampaignStatus.INCOMPLETE_USAGE,
        stop_reason=CampaignStopReason.USAGE_MISSING,
        billing_uncertain=False,
    )
    assert activity.status is CampaignStatus.INCOMPLETE_USAGE
    assert activity.stop_reason.value == "usage_missing"


def test_activity_phase_status_requires_matching_stop_reason() -> None:
    activity = build_initial_activity(
        activity_id="activity-1",
        calibration_plan_sha256="a" * 64,
        calibration_state_sha256="b" * 64,
        case_ids=[f"{index:064x}" for index in range(32)],
    )
    for index, link in enumerate(activity.calibration_artifacts):
        activity = link_calibration_artifact(
            activity, case_id=link.case_id, artifact_sha256=f"{index + 1:064x}"
        )
    activity = mark_calibration_complete(activity)

    with pytest.raises(ValueError, match="stop reason"):
        transition_activity_phase(
            activity,
            phase="dev",
            status=CampaignStatus.INCOMPLETE_BUDGET,
            stop_reason=CampaignStopReason.USAGE_MISSING,
        )


def test_activity_rejects_dev_transition_before_calibration() -> None:
    activity = build_initial_activity(
        activity_id="activity-1",
        calibration_plan_sha256="a" * 64,
        calibration_state_sha256="b" * 64,
        case_ids=[f"{index:064x}" for index in range(32)],
    )
    with pytest.raises(ValueError, match="calibration"):
        transition_activity_phase(activity, phase="dev", status=CampaignStatus.RUNNING)


def test_activity_rejects_dev_terminal_stop_before_calibration() -> None:
    activity = build_initial_activity(
        activity_id="activity-1",
        calibration_plan_sha256="a" * 64,
        calibration_state_sha256="b" * 64,
        case_ids=[f"{index:064x}" for index in range(32)],
    )
    with pytest.raises(ValueError, match="calibration"):
        transition_activity_phase(
            activity,
            phase="dev",
            status=CampaignStatus.INCOMPLETE_BUDGET,
            stop_reason=CampaignStopReason.BUDGET,
        )


@pytest.mark.parametrize(
    ("complete_calibration", "status", "stop_reason", "expected_error"),
    [
        (False, CampaignStatus.RUNNING, None, "calibration"),
        (False, CampaignStatus.INCOMPLETE_BUDGET, CampaignStopReason.BUDGET, "calibration"),
        (True, CampaignStatus.RUNNING, None, "dev"),
        (True, CampaignStatus.INCOMPLETE_BUDGET, CampaignStopReason.BUDGET, "dev"),
    ],
)
def test_activity_rejects_stability_transition_before_predecessors_complete(
    complete_calibration: bool,
    status: CampaignStatus,
    stop_reason: CampaignStopReason | None,
    expected_error: str,
) -> None:
    activity = build_initial_activity(
        activity_id="activity-1",
        calibration_plan_sha256="a" * 64,
        calibration_state_sha256="b" * 64,
        case_ids=[f"{index:064x}" for index in range(32)],
    )
    if complete_calibration:
        for index, link in enumerate(activity.calibration_artifacts):
            activity = link_calibration_artifact(
                activity,
                case_id=link.case_id,
                artifact_sha256=f"{index + 1:064x}",
            )
        activity = mark_calibration_complete(activity)

    with pytest.raises(ValueError, match=expected_error):
        transition_activity_phase(
            activity,
            phase="stability",
            status=status,
            stop_reason=stop_reason,
        )


def test_activity_billing_flag_requires_typed_billing_stop() -> None:
    activity = build_initial_activity(
        activity_id="activity-1",
        calibration_plan_sha256="a" * 64,
        calibration_state_sha256="b" * 64,
        case_ids=[f"{index:064x}" for index in range(32)],
    )
    with pytest.raises(ValueError, match="billing"):
        transition_activity_phase(
            activity,
            phase="calibration",
            status=CampaignStatus.RUNNING,
            billing_uncertain=True,
        )
    with pytest.raises(ValueError, match="billing"):
        transition_activity_phase(
            activity,
            phase="calibration",
            status=CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
            stop_reason=CampaignStopReason.BILLING_UNCERTAIN,
            billing_uncertain=False,
        )


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (CampaignStatus.INTERRUPTED, CampaignStopReason.PROCESS_INTERRUPTION),
        (CampaignStatus.PAUSED, CampaignStopReason.USER_PAUSED),
    ],
)
def test_resume_activity_phase_clears_resumable_pause(status, reason) -> None:
    activity = build_initial_activity(
        activity_id="activity-1",
        calibration_plan_sha256="a" * 64,
        calibration_state_sha256="b" * 64,
        case_ids=[f"{index:064x}" for index in range(32)],
    )
    interrupted = transition_activity_phase(
        activity,
        phase="calibration",
        status=status,
        stop_reason=reason,
    )
    resumed = resume_activity_phase(interrupted, phase="calibration")
    assert resumed.calibration_status is CampaignStatus.RUNNING
    assert resumed.stop_reason is None
    with pytest.raises(ValueError, match="can be resumed"):
        resume_activity_phase(activity, phase="calibration")


def test_resume_activity_after_billing_recovery_clears_only_billing_stop() -> None:
    activity = build_initial_activity(
        activity_id="activity-1",
        calibration_plan_sha256="a" * 64,
        calibration_state_sha256="b" * 64,
        case_ids=[f"{index:064x}" for index in range(32)],
    )
    calibrated = mark_calibration_complete(
        activity.model_copy(
            update={
                "calibration_artifacts": [
                    CalibrationArtifactLink(case_id=f"{index:064x}", artifact_sha256="c" * 64)
                    for index in range(32)
                ]
            },
            deep=True,
        )
    )
    stopped = transition_activity_phase(
        calibrated,
        phase="dev",
        status=CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
        stop_reason=CampaignStopReason.BILLING_UNCERTAIN,
        billing_uncertain=True,
    )

    resumed = resume_activity_after_billing_recovery(stopped, phase="dev")

    assert resumed.dev_status is CampaignStatus.RUNNING
    assert resumed.stop_reason is None
    assert resumed.billing_uncertain is False
    assert resumed.status is CampaignStatus.RUNNING


def test_resume_activity_after_authorized_recovery_clears_usage_stop() -> None:
    activity = build_initial_activity(
        activity_id="activity-1",
        calibration_plan_sha256="a" * 64,
        calibration_state_sha256="b" * 64,
        case_ids=[f"{index:064x}" for index in range(32)],
    )
    calibrated = mark_calibration_complete(
        activity.model_copy(
            update={
                "calibration_artifacts": [
                    CalibrationArtifactLink(case_id=f"{index:064x}", artifact_sha256="c" * 64)
                    for index in range(32)
                ]
            },
            deep=True,
        )
    )
    stopped = transition_activity_phase(
        calibrated,
        phase="dev",
        status=CampaignStatus.INCOMPLETE_USAGE,
        stop_reason=CampaignStopReason.USAGE_MISSING,
        billing_uncertain=False,
    )

    resumed = resume_activity_after_billing_recovery(stopped, phase="dev")

    assert resumed.dev_status is CampaignStatus.RUNNING
    assert resumed.stop_reason is None
    assert resumed.billing_uncertain is False
    assert resumed.status is CampaignStatus.RUNNING


def test_resume_activity_after_authorized_recovery_clears_internal_error_stop() -> None:
    activity = build_initial_activity(
        activity_id="activity-1",
        calibration_plan_sha256="a" * 64,
        calibration_state_sha256="b" * 64,
        case_ids=[f"{index:064x}" for index in range(32)],
    )
    calibrated = mark_calibration_complete(
        activity.model_copy(
            update={
                "calibration_artifacts": [
                    CalibrationArtifactLink(case_id=f"{index:064x}", artifact_sha256="c" * 64)
                    for index in range(32)
                ]
            },
            deep=True,
        )
    )
    stopped = transition_activity_phase(
        calibrated,
        phase="dev",
        status=CampaignStatus.FAILED,
        stop_reason=CampaignStopReason.INTERNAL_ERROR,
        billing_uncertain=False,
    )

    resumed = resume_activity_after_billing_recovery(stopped, phase="dev")

    assert resumed.dev_status is CampaignStatus.RUNNING
    assert resumed.stop_reason is None
    assert resumed.billing_uncertain is False
    assert resumed.status is CampaignStatus.RUNNING


def test_verify_current_freeze_allows_authorized_descendant_protocol_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _files(tmp_path)
    expected = build_freeze_identity(
        calibration_manifest=files["calibration"],
        dev_manifest=files["dev"],
        stability_manifest=files["stability"],
        corpus_receipt=files["receipt"],
        prompt_bundle=files["prompts"],
        config_file=files["config"],
        pricing_file=files["pricing"],
        requirements_lock=files["requirements"],
        endpoint_config={"base_url": "https://relay.example/v1"},
        requested_alias="relay-model",
        seed=20260817,
        manifest_freeze_git_sha="a" * 40,
        dev_protocol_git_sha="b" * 40,
    )
    actual = expected.model_copy(update={"dev_protocol_git_sha": "c" * 40})
    monkeypatch.setattr(
        "evidence_route.evaluation.lifecycle.build_freeze_identity",
        lambda **kwargs: actual,
    )
    monkeypatch.setattr(
        "evidence_route.evaluation.lifecycle.verify_git_freeze",
        lambda *args, **kwargs: "c" * 40,
    )

    assert verify_current_freeze(
        expected,
        repository_root=tmp_path,
        allow_descendant_git=True,
    ) == expected


def test_verify_git_freeze_requires_clean_worktree_and_ancestor(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    _write(tmp_path / "tracked.txt", "one\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "one"], cwd=tmp_path, check=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    _write(tmp_path / "tracked.txt", "dirty\n")
    with pytest.raises(ValueError, match="git_worktree_clean"):
        verify_git_freeze(tmp_path, dev_protocol_git_sha=base, manifest_freeze_git_sha=base)


def test_verify_git_freeze_rejects_untracked_worktree_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    _write(tmp_path / "tracked.txt", "one\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "one"], cwd=tmp_path, check=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    _write(tmp_path / "untracked.txt", "must be audited\n")
    with pytest.raises(ValueError, match="git_worktree_clean"):
        verify_git_freeze(tmp_path, dev_protocol_git_sha=base, manifest_freeze_git_sha=base)
