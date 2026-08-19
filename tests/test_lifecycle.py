from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from evidence_route.evaluation.activity import CampaignStatus, FreezeIdentity
from evidence_route.evaluation.lifecycle import (
    ActivityFileError,
    build_freeze_identity,
    build_initial_activity,
    load_activity,
    persist_activity,
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
    assert identity.endpoint_config_sha256 == hashlib.sha256(
        b'{"base_url":"https://relay.example/v1"}'
    ).hexdigest()
    assert "do-not-hash-this" not in identity.model_dump_json()
    assert identity.calibration_runtime_manifest_sha256 == hashlib.sha256(
        files["calibration"].read_bytes()
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
