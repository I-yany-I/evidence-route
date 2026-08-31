import hashlib
import json
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from evidence_route.artifacts import atomic_write_json
from evidence_route.evaluation import reporting
from evidence_route.evaluation.activity import (
    ActivityRecord,
    CampaignStatus,
    RunArtifact,
    artifact_fingerprint,
    summarize_run_artifacts,
)
from evidence_route.evaluation.calibration import runtime_case_fingerprint, state_fingerprint
from evidence_route.evaluation.lifecycle import persist_activity, sha256_file
from evidence_route.evaluation.reporting import (
    PublicationBlocked,
    PublicationGateResult,
    build_report_bundle,
)
from evidence_route.evaluation.stability import StabilityCategory
from evidence_route.llm import make_call_id

pytest_plugins = ["tests.fixtures.evaluation.report_factory"]


def _rewrite_campaign_as_zero_call_failures(report_input) -> None:
    activity_dir = report_input.activity_dir
    plan = json.loads((activity_dir / "plan.json").read_text(encoding="utf-8"))
    state_path = activity_dir / "campaign.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    artifacts: list[RunArtifact] = []
    artifact_sha_by_run_id: dict[str, str] = {}
    campaign_call_ids: set[str] = set()

    for item in state["items"]:
        artifact_path = activity_dir / item["artifact_relpath"]
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        campaign_call_ids.update(payload["call_ids"])
        price_config_id = payload["result"]["price_config_id"]
        payload["result"].update(
            {
                "status": "failed",
                "verdict": None,
                "confidence": None,
                "rationale": "probe retrieval failed before routing",
                "citations": [],
                "available_evidence_ids": [],
                "initial_route": None,
                "escalated": False,
                "failure_stage": "pre_route",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "complete": True,
                },
                "estimated_cost_micro_cny": 0,
                "cost_currency": "CNY",
                "price_config_id": price_config_id,
                "errors": ["PROBE_RETRIEVAL_FAILED"],
            }
        )
        payload.update(
            {
                "call_ids": [],
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "complete": True,
                },
                "usage_source": "not_applicable",
                "actual_cost_micro_cny": 0,
                "known_actual_cost_micro_cny": 0,
                "committed_cost_micro_cny": 0,
                "cost_is_lower_bound": False,
                "fresh_call_count": 0,
                "cache_hit_count": 0,
                "response_model_ids_raw": [],
                "billing_uncertain": False,
                "diagnostic_only": False,
            }
        )
        payload["artifact_sha256"] = artifact_fingerprint(payload)
        artifact = RunArtifact.model_validate(payload)
        atomic_write_json(artifact_path, artifact.model_dump(mode="json"))
        artifacts.append(artifact)
        artifact_sha_by_run_id[artifact.run_id] = artifact.artifact_sha256
        item["status"] = "failed"
        item["artifact_sha256"] = artifact.artifact_sha256

    state["stability_repeat_zero_artifact_sha256s"] = {
        link["claim_id"]: artifact_sha_by_run_id[link["dev_adaptive_run_id"]]
        for link in plan["stability_repeat_zero_links"]
    }
    state["summary"] = summarize_run_artifacts(artifacts).model_dump(mode="json")
    atomic_write_json(state_path, state)

    connection = sqlite3.connect(report_input.run_store)
    connection.executemany(
        "DELETE FROM calls WHERE call_id = ?",
        [(call_id,) for call_id in campaign_call_ids],
    )
    rows = connection.execute(
        "SELECT call_id, usage_json, actual_micro_cny, usage_source, requested_alias, "
        "response_model_id_raw, transport_attempts, cache_hits FROM calls ORDER BY rowid"
    ).fetchall()
    connection.commit()
    connection.close()

    usages = [json.loads(row[1]) for row in rows]
    actual_cost = sum(row[2] for row in rows)
    activity_path = activity_dir / "activity.json"
    activity = json.loads(activity_path.read_text(encoding="utf-8"))
    activity["campaign_state_sha256"] = sha256_file(state_path)
    activity["summary"].update(
        {
            "call_ids": [row[0] for row in rows],
            "usage": {
                "input_tokens": sum(usage["input_tokens"] for usage in usages),
                "output_tokens": sum(usage["output_tokens"] for usage in usages),
                "total_tokens": sum(usage["total_tokens"] for usage in usages),
                "complete": all(usage["complete"] for usage in usages),
            },
            "usage_sources": [row[3] for row in rows],
            "actual_cost_micro_cny": actual_cost,
            "known_actual_cost_micro_cny": actual_cost,
            "committed_cost_micro_cny": actual_cost,
            "cost_is_lower_bound": False,
            "fresh_call_count": len(rows),
            "cache_hit_count": sum(row[7] for row in rows),
            "transport_attempts": sum(row[6] for row in rows),
            "requested_aliases": sorted({row[4] for row in rows}),
            "response_model_ids_raw": sorted({row[5] for row in rows}),
            "billing_uncertain": False,
        }
    )
    persist_activity(activity_path, ActivityRecord.model_validate(activity))

    adaptive_digest = hashlib.sha256(
        json.dumps(
            sorted(
                (artifact.run_id, artifact.artifact_sha256)
                for artifact in artifacts
                if artifact.phase == "dev"
                and artifact.strategy.value == "adaptive"
                and artifact.repeat == 0
            ),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    official_path = activity_dir / "official" / "summary.json"
    official = json.loads(official_path.read_text(encoding="utf-8"))
    official["report_evidence"]["adaptive_artifacts_sha256"] = adaptive_digest
    atomic_write_json(official_path, official)


def _reseal_calibration_call_slot(
    report_input,
    *,
    run_field: str,
    original_node: str,
    tampered_node: str,
    tampered_attempt: int,
) -> None:
    activity_dir = report_input.activity_dir
    case_path = sorted((activity_dir / "calibration" / "cases").glob("*.json"))[0]
    case = json.loads(case_path.read_text(encoding="utf-8"))
    run_id = case[run_field]
    old_call_id = make_call_id(run_id, original_node, "root", 0)
    new_call_id = make_call_id(run_id, tampered_node, "root", tampered_attempt)

    connection = sqlite3.connect(report_input.run_store)
    updated = connection.execute(
        "UPDATE calls SET call_id = ?, node = ?, logical_attempt = ? WHERE call_id = ?",
        (new_call_id, tampered_node, tampered_attempt, old_call_id),
    ).rowcount
    assert updated == 1
    connection.commit()
    connection.close()

    case["call_ids"] = [
        new_call_id if call_id == old_call_id else call_id for call_id in case["call_ids"]
    ]
    request_sha256 = case["request_sha256_by_call_id"].pop(old_call_id)
    case["request_sha256_by_call_id"][new_call_id] = request_sha256
    case["artifact_sha256"] = runtime_case_fingerprint(case)
    atomic_write_json(case_path, case)

    state_path = activity_dir / "calibration-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state_item = next(item for item in state["items"] if item["case_id"] == case["case_id"])
    state_item["artifact_sha256"] = case["artifact_sha256"]
    state["state_sha256"] = state_fingerprint(state)
    atomic_write_json(state_path, state)

    activity_path = activity_dir / "activity.json"
    activity = json.loads(activity_path.read_text(encoding="utf-8"))
    activity["calibration_state_sha256"] = sha256_file(state_path)
    activity_link = next(
        link for link in activity["calibration_artifacts"] if link["case_id"] == case["case_id"]
    )
    activity_link["artifact_sha256"] = case["artifact_sha256"]
    activity["summary"]["call_ids"] = [
        new_call_id if call_id == old_call_id else call_id
        for call_id in activity["summary"]["call_ids"]
    ]
    persist_activity(activity_path, ActivityRecord.model_validate(activity))


def test_report_uses_exact_subset_name(complete_report_input) -> None:
    bundle = build_report_bundle(complete_report_input)
    assert "AVeriTeC dev balanced subset (n=80)" in bundle.markdown
    assert "full benchmark" not in bundle.markdown.lower()
    assert "leaderboard" not in bundle.markdown.lower()
    assert "Adaptive route mix:" in bundle.markdown


def test_resume_text_uses_measured_values_not_design_targets(complete_report_input) -> None:
    bundle = build_report_bundle(complete_report_input)
    adaptive = bundle.summary["strategies"]["adaptive"]
    assert f"{adaptive['token_reduction_pct']:.1f}%" in bundle.resume_snippet
    assert f"{adaptive['full_manifest_macro_f1']:.3f}" in bundle.resume_snippet
    assert "至少降低 40%" not in bundle.resume_snippet
    assert "官方 OpenAI" not in bundle.resume_snippet


def test_publish_forces_strict_git_and_official_evaluator_paths(
    report_input_factory, monkeypatch
) -> None:
    report_input = report_input_factory.report_input(allow_official_cache=True)

    with pytest.raises(PublicationBlocked, match="current_git_missing"):
        build_report_bundle(report_input, publish=True)

    # Isolate the official-evaluator switch from the Git gate.  A permissive ReportInput must
    # not let publish mode consume the fixture cache without rerunning the pinned evaluators.
    monkeypatch.setattr(
        "evidence_route.evaluation.reporting._publication_gate",
        lambda *args, **kwargs: PublicationGateResult(publishable=True),
    )
    evaluator_calls: list[bool] = []

    def fail_evaluator(*args, **kwargs):
        evaluator_calls.append(True)
        raise RuntimeError("pinned evaluator failed")

    monkeypatch.setattr(
        "evidence_route.evaluation.reporting.run_official_evaluators", fail_evaluator
    )

    with pytest.raises(PublicationBlocked, match="official_evaluator_unavailable"):
        build_report_bundle(report_input, publish=True)
    assert evaluator_calls == [True]


def test_permissive_diagnostic_bundle_cannot_be_written_as_final(
    report_input_factory,
) -> None:
    report_input = report_input_factory.report_input(allow_official_cache=True)
    bundle = build_report_bundle(report_input, publish=False)
    assert bundle.publication_gate.publishable is True

    with pytest.raises(PublicationBlocked, match="strict publication"):
        bundle.write(report_input.repository_root / "reports" / "final")


def test_publication_scorer_manifest_must_match_frozen_git_blob(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    manifest = tmp_path / "data" / "scorer_manifests" / "averitec_dev_gold.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"labels":["Supported"]}\n', encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze scorer manifest"], cwd=tmp_path, check=True)
    frozen_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()

    assert reporting._audit_scorer_manifest(  # type: ignore[attr-defined]
        tmp_path,
        manifest,
        dev_protocol_git_sha=frozen_sha,
    ) == []

    manifest.write_text('{"labels":["Refuted"]}\n', encoding="utf-8")
    assert "scorer_manifest_git_blob_mismatch" in reporting._audit_scorer_manifest(  # type: ignore[attr-defined]
        tmp_path,
        manifest,
        dev_protocol_git_sha=frozen_sha,
    )

    copied = tmp_path / "copied-gold.json"
    copied.write_bytes(manifest.read_bytes())
    assert "scorer_manifest_path_untrusted" in reporting._audit_scorer_manifest(  # type: ignore[attr-defined]
        tmp_path,
        copied,
        dev_protocol_git_sha=frozen_sha,
    )


def test_publication_git_audit_allows_generated_outputs_and_descendant_commit(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    source = tmp_path / "src" / "protocol.py"
    source.parent.mkdir(parents=True)
    source.write_text("FROZEN = True\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze protocol"], cwd=tmp_path, check=True)
    protocol_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()

    readme.write_text("generated results\n", encoding="utf-8")
    final_report = tmp_path / "reports" / "final" / "summary.json"
    final_report.parent.mkdir(parents=True)
    final_report.write_text("{}\n", encoding="utf-8")
    assert reporting._audit_publication_git(  # type: ignore[attr-defined]
        tmp_path,
        dev_protocol_git_sha=protocol_sha,
        manifest_freeze_git_sha=protocol_sha,
    ) == []

    subprocess.run(["git", "add", "README.md", "reports/final"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "publish generated report"], cwd=tmp_path, check=True)
    assert reporting._audit_publication_git(  # type: ignore[attr-defined]
        tmp_path,
        dev_protocol_git_sha=protocol_sha,
        manifest_freeze_git_sha=protocol_sha,
    ) == []

    source.write_text("FROZEN = False\n", encoding="utf-8")
    assert "current_git_worktree_dirty" in reporting._audit_publication_git(  # type: ignore[attr-defined]
        tmp_path,
        dev_protocol_git_sha=protocol_sha,
        manifest_freeze_git_sha=protocol_sha,
    )

    subprocess.run(["git", "add", "src/protocol.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "unauthorized source change"], cwd=tmp_path, check=True)
    assert "current_git_worktree_dirty" in reporting._audit_publication_git(  # type: ignore[attr-defined]
        tmp_path,
        dev_protocol_git_sha=protocol_sha,
        manifest_freeze_git_sha=protocol_sha,
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"M\0src/foo.py\0", ["src/foo.py"]),
        (b"R100\0src/old.py\0src/new.py\0", ["src/old.py", "src/new.py"]),
        (b"M  src/foo.py\0", ["src/foo.py"]),
        (b"R  src/old.py\0src/new.py\0", ["src/old.py", "src/new.py"]),
    ],
)
def test_git_status_paths_supports_name_status_and_porcelain(
    payload: bytes, expected: list[str]
) -> None:
    assert reporting._git_status_paths(payload) == expected  # type: ignore[attr-defined]


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.relative_to(root).as_posix() != "PREPARATION_RECEIPT.json"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def test_publication_rejects_tampered_nltk_tree(tmp_path: Path) -> None:
    source_spec = tmp_path / "data" / "sources" / "nltk_data.json"
    source_spec.parent.mkdir(parents=True)
    source_spec.write_text(
        json.dumps(
            {
                "repository": "nltk/nltk_data",
                "commit": "a" * 40,
                "files": {
                    "packages/tokenizers/punkt_tab.zip": {
                        "size": 3,
                        "sha256": "b" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    nltk_root = tmp_path / "data" / "external" / "nltk"
    asset = nltk_root / "tokenizers" / "punkt_tab" / "english" / "asset.tab"
    asset.parent.mkdir(parents=True)
    asset.write_text("one\n", encoding="utf-8")
    receipt = {
        "schema_version": "1",
        "repository": "nltk/nltk_data",
        "commit": "a" * 40,
        "files": {
            "packages/tokenizers/punkt_tab.zip": {
                "size": 3,
                "sha256": "b" * 64,
                "url": (
                    "https://raw.githubusercontent.com/nltk/nltk_data/"
                    + "a" * 40
                    + "/packages/tokenizers/punkt_tab.zip"
                ),
            }
        },
        "extracted_tree_sha256": _tree_sha256(nltk_root),
    }
    (nltk_root / "PREPARATION_RECEIPT.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    assert reporting._audit_nltk_data(  # type: ignore[attr-defined]
        tmp_path,
        nltk_root,
    ) == []

    asset.write_text("tampered\n", encoding="utf-8")
    assert "nltk_data_invalid" in reporting._audit_nltk_data(  # type: ignore[attr-defined]
        tmp_path,
        nltk_root,
    )


@pytest.mark.parametrize(
    "status", [status for status in CampaignStatus if status is not CampaignStatus.COMPLETE]
)
def test_every_non_complete_status_is_diagnostic_only(
    status, report_input_factory, tmp_path: Path
) -> None:
    report_input = report_input_factory.with_status(status)
    bundle = build_report_bundle(report_input, publish=False)
    diagnostic = tmp_path / "reports" / "incomplete" / status.value
    bundle.write(diagnostic)
    assert bundle.summary["publishable"] is False
    assert bundle.resume_snippet is None
    assert not (diagnostic / "resume_snippet.md").exists()

    final_dir = tmp_path / "reports" / "final"
    readme = tmp_path / "README.md"
    readme.write_text("unchanged", encoding="utf-8")
    with pytest.raises(PublicationBlocked, match=status.value):
        bundle.write(final_dir, readme=readme)
    with pytest.raises(PublicationBlocked, match=status.value):
        build_report_bundle(report_input, publish=True)
    assert not final_dir.exists()
    assert readme.read_text(encoding="utf-8") == "unchanged"


def test_regeneration_reads_artifacts_without_executor(
    complete_report_input, tmp_path: Path
) -> None:
    bundle = build_report_bundle(complete_report_input)
    bundle.write(tmp_path / "generated")
    assert (tmp_path / "generated" / "summary.json").is_file()
    assert (tmp_path / "generated" / "report.md").is_file()
    assert (tmp_path / "generated" / "resume_snippet.md").is_file()


def test_report_regeneration_is_byte_identical(complete_report_input, tmp_path: Path) -> None:
    first = build_report_bundle(complete_report_input)
    second = build_report_bundle(complete_report_input)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first.write(first_dir)
    second.write(second_dir)
    for name in ("summary.json", "report.md", "resume_snippet.md"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


def test_report_without_stability_diagnostics_removes_stale_files(
    complete_report_input, tmp_path: Path
) -> None:
    output_dir = tmp_path / "default"
    output_dir.mkdir()
    (output_dir / "stability_diagnostics.json").write_bytes(b"stale\n")
    (output_dir / "stability_diagnostics.md").write_bytes(b"stale\n")

    bundle = build_report_bundle(complete_report_input)
    bundle.write(output_dir)

    assert bundle.stability_diagnostics is None
    assert not (output_dir / "stability_diagnostics.json").exists()
    assert not (output_dir / "stability_diagnostics.md").exists()


def test_stability_diagnostics_are_provider_free_and_byte_deterministic(
    complete_report_input, tmp_path: Path, monkeypatch
) -> None:
    def fail_official(*args, **kwargs):
        raise AssertionError("stability diagnostics must not call an official evaluator")

    monkeypatch.setattr(reporting, "run_official_evaluators", fail_official)
    first = build_report_bundle(complete_report_input, stability_diagnostics=True)
    second = build_report_bundle(complete_report_input, stability_diagnostics=True)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first.write(first_dir)
    second.write(second_dir)

    assert first.stability_diagnostics is not None
    assert first.stability_diagnostics.records == sorted(
        first.stability_diagnostics.records, key=lambda record: record.claim_id
    )
    assert first.stability_diagnostics.claim_count == 20
    assert all(len(record.repeats) == 3 for record in first.stability_diagnostics.records)
    assert json.loads(
        (first_dir / "stability_diagnostics.json").read_text(encoding="utf-8")
    ) == first.stability_diagnostics.model_dump(mode="json")
    assert (
        first_dir / "stability_diagnostics.md"
    ).read_text(encoding="utf-8").splitlines()[0] == (
        "| claim_id | primary_category | categories | repeat_0_status/verdict | "
        "repeat_1_status/verdict | repeat_2_status/verdict |"
    )
    for name in (
        "summary.json",
        "report.md",
        "stability_diagnostics.json",
        "stability_diagnostics.md",
    ):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


def test_stability_diagnostics_preserve_missing_repeat_and_bad_ledger(
    complete_report_input, tmp_path: Path, monkeypatch
) -> None:
    invalid_input = replace(complete_report_input, run_store=tmp_path / "missing.sqlite3")
    bundle = build_report_bundle(invalid_input, stability_diagnostics=True)

    assert bundle.stability_diagnostics is not None
    record = bundle.stability_diagnostics.records[0]
    assert record.primary_category is None
    assert all(snapshot.transport_attempts is None for snapshot in record.repeats)

    original_load = reporting._load_artifacts
    missing_run_id = next(
        item.run_id
        for item in reporting._load_models(complete_report_input)[1].schedule
        if item.phase == "stability" and item.repeat == 2
    )

    def load_without_one_repeat(activity_dir, plan, state):
        artifacts, paths = original_load(activity_dir, plan, state)
        artifacts.pop(missing_run_id, None)
        paths.pop(missing_run_id, None)
        return artifacts, paths

    monkeypatch.setattr(reporting, "_load_artifacts", load_without_one_repeat)
    bundle = build_report_bundle(complete_report_input, stability_diagnostics=True)

    assert bundle.stability_diagnostics is not None
    assert bundle.stability_diagnostics.records[0].repeats[2].valid is False
    assert (
        bundle.stability_diagnostics.records[0].primary_category
        is StabilityCategory.INCOMPLETE_OR_FAILED
    )


def test_missing_calibration_case_blocks_publication(report_input_factory) -> None:
    report_input = report_input_factory.report_input()
    next((report_input.activity_dir / "calibration" / "cases").glob("*.json")).unlink()

    bundle = build_report_bundle(report_input, publish=False)

    assert bundle.publication_gate.publishable is False
    assert "calibration_case_missing" in bundle.publication_gate.reasons


def test_alias_drift_between_calibration_and_campaign_blocks_publication(
    report_input_factory,
) -> None:
    report_input = report_input_factory.report_input()
    plan_path = report_input.activity_dir / "calibration-plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["requested_alias"] = "different-alias"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    bundle = build_report_bundle(report_input, publish=False)

    assert bundle.publication_gate.publishable is False
    assert "calibration_evidence_invalid" in bundle.publication_gate.reasons


def test_diagnostic_report_cannot_write_below_final_directory(
    report_input_factory, tmp_path: Path
) -> None:
    report_input = report_input_factory.with_status(CampaignStatus.INTERRUPTED)
    bundle = build_report_bundle(report_input, publish=False)

    with pytest.raises(PublicationBlocked):
        bundle.write(tmp_path / "reports" / "final" / "nested")


def test_stale_calibration_report_from_another_activity_is_rejected(
    report_input_factory,
) -> None:
    report_input = report_input_factory.report_input()
    report_path = report_input.repository_root / "reports/calibration/calibration_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["activity_id"] = "other-activity"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    replay_path = report_input.activity_dir / "calibration-replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["calibration_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    replay_path.write_text(json.dumps(replay), encoding="utf-8")

    bundle = build_report_bundle(report_input, publish=False)

    assert bundle.publication_gate.publishable is False
    assert "calibration_replay_identity_mismatch" in bundle.publication_gate.reasons


def test_report_runs_and_persists_official_evaluators(report_input_factory, monkeypatch) -> None:
    report_input = report_input_factory.report_input()
    (report_input.activity_dir / "official" / "summary.json").unlink()
    calls = []

    def fake_official(selection, *, output_dir, nltk_data):
        calls.append((selection, output_dir, nltk_data))
        return {
            "shared_task_2024": {"available": True, "score": 0.5},
            "paper_2023_secondary": {"available": True, "score": 0.4},
        }

    monkeypatch.setattr(
        "evidence_route.evaluation.reporting.run_official_evaluators", fake_official
    )

    bundle = build_report_bundle(report_input)

    assert len(calls) == 1
    assert calls[0][0].completed_count == 80
    saved = json.loads(
        (report_input.activity_dir / "official" / "summary.json").read_text(encoding="utf-8")
    )
    assert saved == bundle.summary["official"]


def test_official_evaluator_failure_blocks_resume_claims(report_input_factory, monkeypatch) -> None:
    report_input = report_input_factory.report_input()
    (report_input.activity_dir / "official" / "summary.json").unlink()

    def fail_official(*args, **kwargs):
        raise RuntimeError("pinned evaluator failed")

    monkeypatch.setattr(
        "evidence_route.evaluation.reporting.run_official_evaluators", fail_official
    )

    bundle = build_report_bundle(report_input, publish=False)

    assert bundle.publication_gate.publishable is False
    assert "official_evaluator_unavailable" in bundle.publication_gate.reasons
    assert bundle.resume_snippet is None


def test_unexpected_activity_ledger_call_blocks_publication(
    report_input_factory,
) -> None:
    report_input = report_input_factory.report_input()
    connection = sqlite3.connect(report_input.run_store)
    connection.execute(
        "INSERT INTO calls (call_id, request_sha256, activity_id, run_id, node, task_id, "
        "logical_attempt, state, reserved_micro_cny, actual_micro_cny, payload_json, "
        "usage_json, usage_source, requested_alias, response_model_id_raw, identity_verified, "
        "transport_attempts, cache_hits, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "extra-call",
            "a" * 64,
            "gate-a",
            "extra-run",
            "router",
            "root",
            0,
            "completed",
            1,
            1,
            "{}",
            json.dumps(
                {
                    "input_tokens": 1,
                    "output_tokens": 0,
                    "total_tokens": 1,
                    "complete": True,
                }
            ),
            "provider",
            "fixture-alias",
            "relay-model-a",
            0,
            0,
            0,
            "fixture",
            "fixture",
        ),
    )
    connection.commit()
    connection.close()

    bundle = build_report_bundle(report_input, publish=False)

    assert "run_store_call_set_mismatch" in bundle.publication_gate.reasons


def test_all_zero_call_pre_route_failures_preserve_calibrated_model_identity(
    report_input_factory,
) -> None:
    report_input = report_input_factory.report_input(allow_official_cache=True)
    _rewrite_campaign_as_zero_call_failures(report_input)

    bundle = build_report_bundle(report_input, publish=False)

    assert bundle.publication_gate.publishable is True
    assert "response_model_id_count_not_one" not in bundle.publication_gate.reasons


@pytest.mark.parametrize(
    ("run_field", "original_node", "tampered_node", "tampered_attempt"),
    [
        ("router_run_id", "router", "single", 0),
        ("single_run_id", "single", "single", 1),
    ],
)
def test_resealed_calibration_call_slot_reassignment_blocks_publication(
    report_input_factory,
    run_field: str,
    original_node: str,
    tampered_node: str,
    tampered_attempt: int,
) -> None:
    report_input = report_input_factory.report_input(allow_official_cache=True)
    _reseal_calibration_call_slot(
        report_input,
        run_field=run_field,
        original_node=original_node,
        tampered_node=tampered_node,
        tampered_attempt=tampered_attempt,
    )

    bundle = build_report_bundle(report_input, publish=False)

    assert "run_store_accounting_invalid" in bundle.publication_gate.reasons


def test_ledger_model_drift_blocks_publication(report_input_factory) -> None:
    report_input = report_input_factory.report_input()
    connection = sqlite3.connect(report_input.run_store)
    connection.execute(
        "UPDATE calls SET response_model_id_raw = ? WHERE call_id = "
        "(SELECT call_id FROM calls LIMIT 1)",
        ("relay-model-b",),
    )
    connection.commit()
    connection.close()

    bundle = build_report_bundle(report_input, publish=False)

    assert bundle.publication_gate.publishable is False
    assert "run_store_accounting_invalid" in bundle.publication_gate.reasons


def test_completed_call_reservation_must_cover_actual_cost(report_input_factory) -> None:
    report_input = report_input_factory.report_input(allow_official_cache=True)
    connection = sqlite3.connect(report_input.run_store)
    connection.execute(
        "UPDATE calls SET reserved_micro_cny = 0 WHERE call_id = "
        "(SELECT call_id FROM calls LIMIT 1)"
    )
    connection.commit()
    connection.close()

    bundle = build_report_bundle(report_input, publish=False)

    assert bundle.publication_gate.publishable is False
    assert "run_store_accounting_invalid" in bundle.publication_gate.reasons


def test_expected_ledger_call_cannot_claim_another_activity(report_input_factory) -> None:
    report_input = report_input_factory.report_input(allow_official_cache=True)
    connection = sqlite3.connect(report_input.run_store)
    connection.execute(
        "UPDATE calls SET activity_id = ? WHERE call_id = (SELECT call_id FROM calls LIMIT 1)",
        ("other-activity",),
    )
    connection.commit()
    connection.close()

    bundle = build_report_bundle(report_input, publish=False)

    assert bundle.publication_gate.publishable is False
    assert "run_store_accounting_invalid" in bundle.publication_gate.reasons


@pytest.mark.parametrize(
    ("run_kind", "token_delta", "cost_delta"),
    [
        ("campaign", 1, 1),
        ("calibration", 1, 1),
    ],
)
def test_ledger_run_accounting_must_match_saved_evidence(
    report_input_factory,
    run_kind: str,
    token_delta: int,
    cost_delta: int,
) -> None:
    """Moving usage/cost between calls must not preserve a publishable aggregate."""

    report_input = report_input_factory.report_input(allow_official_cache=True)
    if run_kind == "campaign":
        evidence_paths = sorted((report_input.activity_dir / "artifacts").glob("*.json"))[:2]
        run_ids = [
            json.loads(path.read_text(encoding="utf-8"))["run_id"] for path in evidence_paths
        ]
    else:
        evidence_paths = sorted(
            (report_input.activity_dir / "calibration" / "cases").glob("*.json")
        )[:2]
        run_ids = [
            json.loads(path.read_text(encoding="utf-8"))["router_run_id"] for path in evidence_paths
        ]

    connection = sqlite3.connect(report_input.run_store)
    rows = [
        connection.execute(
            "SELECT call_id, usage_json, actual_micro_cny FROM calls "
            "WHERE run_id = ? ORDER BY call_id LIMIT 1",
            (run_id,),
        ).fetchone()
        for run_id in run_ids
    ]
    assert all(row is not None for row in rows)
    assert len(rows) == 2
    for index, (call_id, usage_json, actual_cost) in enumerate(rows):
        usage = json.loads(usage_json)
        direction = 1 if index == 0 else -1
        usage["input_tokens"] += direction * token_delta
        usage["total_tokens"] += direction * token_delta
        connection.execute(
            "UPDATE calls SET usage_json = ?, actual_micro_cny = ? WHERE call_id = ?",
            (
                json.dumps(usage),
                actual_cost + direction * cost_delta,
                call_id,
            ),
        )
    connection.commit()
    connection.close()

    bundle = build_report_bundle(report_input, publish=False)

    assert "run_store_evidence_accounting_mismatch" in bundle.publication_gate.reasons


def test_stale_official_cache_cannot_bypass_pinned_evaluator(
    report_input_factory, monkeypatch
) -> None:
    report_input = report_input_factory.report_input()

    def fail_official(*args, **kwargs):
        raise RuntimeError("pinned evaluator failed")

    monkeypatch.setattr(
        "evidence_route.evaluation.reporting.run_official_evaluators", fail_official
    )

    bundle = build_report_bundle(report_input, publish=False)

    assert bundle.publication_gate.publishable is False
    assert "official_evaluator_unavailable" in bundle.publication_gate.reasons


def test_activity_sidecar_is_required_for_publication(report_input_factory) -> None:
    report_input = report_input_factory.report_input()
    (report_input.activity_dir / "activity.json.sha256").unlink()

    with pytest.raises(ValueError, match="activity sidecar"):
        build_report_bundle(report_input, publish=False)


def test_unrelated_activity_ledger_call_blocks_publication(report_input_factory) -> None:
    report_input = report_input_factory.report_input()
    connection = sqlite3.connect(report_input.run_store)
    connection.execute(
        "INSERT INTO calls (call_id, request_sha256, activity_id, run_id, node, task_id, "
        "logical_attempt, state, reserved_micro_cny, actual_micro_cny, payload_json, "
        "usage_json, usage_source, requested_alias, response_model_id_raw, identity_verified, "
        "transport_attempts, cache_hits, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "other-activity-call",
            "b" * 64,
            "other-activity",
            "other-run",
            "router",
            "root",
            0,
            "completed",
            1,
            1,
            "{}",
            json.dumps(
                {
                    "input_tokens": 1,
                    "output_tokens": 0,
                    "total_tokens": 1,
                    "complete": True,
                }
            ),
            "provider",
            "fixture-alias",
            "relay-model-a",
            0,
            0,
            0,
            "fixture",
            "fixture",
        ),
    )
    connection.commit()
    connection.close()

    bundle = build_report_bundle(report_input, publish=False)

    assert "run_store_call_set_mismatch" in bundle.publication_gate.reasons


def test_runtime_manifest_sidecar_is_verified(report_input_factory) -> None:
    report_input = report_input_factory.report_input()
    report_input.runtime_manifest.with_suffix(".json.sha256").write_text(
        "0" * 64 + "\n", encoding="ascii"
    )

    bundle = build_report_bundle(report_input, publish=False)

    assert bundle.publication_gate.publishable is False
    assert any("runtime_manifest" in reason for reason in bundle.publication_gate.reasons)


def test_calibration_runtime_manifest_sidecar_is_verified(report_input_factory) -> None:
    report_input = report_input_factory.report_input(allow_official_cache=True)
    report_input.calibration_runtime_manifest.with_suffix(".json.sha256").write_text(
        "0" * 64 + "\n", encoding="ascii"
    )

    bundle = build_report_bundle(report_input, publish=False)

    assert bundle.publication_gate.publishable is False
    assert "calibration_runtime_manifest_invalid" in bundle.publication_gate.reasons


def test_stability_baseline_digest_map_is_verified(report_input_factory) -> None:
    report_input = report_input_factory.report_input()
    state_path = report_input.activity_dir / "campaign.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["stability_repeat_zero_artifact_sha256s"] = {}
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    activity_path = report_input.activity_dir / "activity.json"
    activity_payload = json.loads(activity_path.read_text(encoding="utf-8"))
    activity_payload["campaign_state_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    activity_path.write_text(json.dumps(activity_payload), encoding="utf-8")
    activity_path.with_suffix(".json.sha256").write_text(
        hashlib.sha256(activity_path.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )

    bundle = build_report_bundle(report_input, publish=False)

    assert "stability_baseline_links_mismatch" in bundle.publication_gate.reasons


def test_campaign_state_status_must_match_saved_artifact(report_input_factory) -> None:
    report_input = report_input_factory.report_input(allow_official_cache=True)
    state_path = report_input.activity_dir / "campaign.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["items"][0]["status"] = "partial"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    activity_path = report_input.activity_dir / "activity.json"
    activity_payload = json.loads(activity_path.read_text(encoding="utf-8"))
    activity_payload["campaign_state_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    activity_path.write_text(json.dumps(activity_payload), encoding="utf-8")
    activity_path.with_suffix(".json.sha256").write_text(
        hashlib.sha256(activity_path.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )

    with pytest.raises(ValueError, match="status"):
        build_report_bundle(report_input, publish=False)


def test_stopped_campaign_item_requires_diagnostic_artifact(report_input_factory) -> None:
    report_input = report_input_factory.report_input(allow_official_cache=True)
    state_path = report_input.activity_dir / "campaign.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["items"][0]["status"] = "stopped"
    payload["items"][0]["stop_reason"] = "internal_error"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    activity_path = report_input.activity_dir / "activity.json"
    activity_payload = json.loads(activity_path.read_text(encoding="utf-8"))
    activity_payload["campaign_state_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    activity_path.write_text(json.dumps(activity_payload), encoding="utf-8")
    activity_path.with_suffix(".json.sha256").write_text(
        hashlib.sha256(activity_path.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )

    with pytest.raises(ValueError, match="diagnostic"):
        build_report_bundle(report_input, publish=False)


def test_current_endpoint_drift_blocks_publication(report_input_factory, monkeypatch) -> None:
    report_input = report_input_factory.report_input()
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://changed.example/v1")

    bundle = build_report_bundle(report_input, publish=False)

    assert "current_endpoint_config_sha256_mismatch" in bundle.publication_gate.reasons


def test_strategy_summary_reports_measured_router_and_citation_rates(
    complete_report_input,
) -> None:
    bundle = build_report_bundle(complete_report_input)
    adaptive = bundle.summary["strategies"]["adaptive"]

    assert isinstance(adaptive["llm_router_rate"], float)
    assert isinstance(adaptive["citation_validity_rate"], float)


def test_completed_empty_evidence_is_citation_valid(report_input_factory) -> None:
    bundle = build_report_bundle(report_input_factory.report_input())
    for values in bundle.summary["strategies"].values():
        assert values["citation_validity_rate"] == 1.0
