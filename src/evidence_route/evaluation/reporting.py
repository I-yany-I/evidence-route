"""Deterministic reporting over immutable Gate A evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import Field

from evidence_route.artifacts import SQLiteRunStore
from evidence_route.contracts import ResultStatus, Strategy, StrictModel, Usage, Verdict
from evidence_route.evaluation.activity import (
    ActivityRecord,
    CampaignPlan,
    CampaignState,
    CampaignStatus,
    RunArtifact,
    campaign_fingerprint,
    result_status_to_work_status,
    verify_artifact_fingerprint,
)
from evidence_route.evaluation.calibration import (
    CalibrationReplay,
    load_calibration_cases,
    load_calibration_plan,
    load_calibration_state,
)
from evidence_route.evaluation.lifecycle import endpoint_config_hash, load_activity
from evidence_route.evaluation.metrics import (
    paired_bootstrap_difference,
    score_completed_conditionally,
    score_full_manifest,
    stability_consistency,
    summarize_operations,
)
from evidence_route.evaluation.official import run_official_evaluators, select_completed_triples
from evidence_route.evaluation.runner import build_campaign_schedule
from evidence_route.evaluation.runtime_manifest import load_runtime_manifest
from evidence_route.evaluation.scorer_manifest import align_runtime_and_gold, load_gold_manifest
from evidence_route.execution import load_price_config
from evidence_route.llm import ensure_v1, make_call_id

BENCHMARK_NAME = "AVeriTeC dev balanced subset (n=80)"
IDENTITY_DESCRIPTION = "OpenAI-compatible provider; relay-reported model ID; identity unverified"
_RESULTS_START = "<!-- EVIDENCE_ROUTE_RESULTS_START -->"
_RESULTS_END = "<!-- EVIDENCE_ROUTE_RESULTS_END -->"


class PublicationBlocked(RuntimeError):
    """Raised when diagnostic artifacts are requested for publication."""


@dataclass(frozen=True)
class ReportInput:
    repository_root: Path
    activity_dir: Path
    gold_manifest: Path
    nltk_data_root: Path
    # Optional explicit evidence paths. Defaults are resolved relative to the activity/repository
    # roots, preserving compatibility with the original four-field constructor.
    run_store: Path | None = None
    runtime_manifest: Path | None = None
    calibration_plan: Path | None = None
    calibration_state: Path | None = None
    calibration_report: Path | None = None
    calibrated_config: Path | None = None
    calibration_replay: Path | None = None
    official_output_dir: Path | None = None
    calibration_runtime_manifest: Path | None = None
    stability_runtime_manifest: Path | None = None
    corpus_preparation_receipt: Path | None = None
    prompt_bundle: Path | None = None
    pricing: Path | None = None
    requirements_lock: Path | None = None
    # These switches are intentionally opt-in for ephemeral offline fixtures.  The CLI keeps
    # strict publication defaults: a real report must audit Git and rerun pinned evaluators.
    require_git: bool = True
    allow_official_cache: bool = False


class PublicationGateResult(StrictModel):
    publishable: bool
    reasons: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ReportBundle:
    summary: dict[str, object]
    markdown: str
    resume_snippet: str | None
    representative_trace_sources: tuple[Path, ...]
    publication_gate: PublicationGateResult
    strict_publication_verified: bool = False

    def write(self, output_dir: Path, *, readme: Path | None = None) -> None:
        output_dir = Path(output_dir)
        publication_target = readme is not None or _is_final_output(output_dir)
        if publication_target and (
            not self.publication_gate.publishable or not self.strict_publication_verified
        ):
            details = "; ".join(self.publication_gate.reasons)
            message = "strict publication verification is required"
            raise PublicationBlocked(f"{message}; {details}" if details else message)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "summary.json", self.summary)
        _write_text(output_dir / "report.md", self.markdown)
        resume_path = output_dir / "resume_snippet.md"
        if self.resume_snippet is not None:
            _write_text(resume_path, self.resume_snippet)
        elif resume_path.exists():
            resume_path.unlink()
        trace_dir = output_dir / "traces"
        if self.representative_trace_sources:
            trace_dir.mkdir(parents=True, exist_ok=True)
            for source in self.representative_trace_sources:
                shutil.copyfile(source, trace_dir / source.name)
        if readme is not None:
            _update_readme(Path(readme), self.summary)


def _write_json(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    _write_text(path, encoded)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_final_output(path: Path) -> bool:
    """Return true for ``reports/final`` and any nested output below it."""

    resolved = Path(path).resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name.lower() == "final" and parent.parent.name.lower() == "reports":
            return True
    return False


def _git_output(repository_root: Path, args: list[str]) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _trusted_repository_path(repository_root: Path, path: Path, relative: str) -> bool:
    try:
        actual = Path(path).resolve().relative_to(Path(repository_root).resolve())
    except ValueError:
        return False
    return actual.as_posix() == relative


def _git_blob_matches(repository_root: Path, path: Path, relative: str, commit: str) -> bool:
    frozen_blob = _git_output(
        repository_root,
        ["rev-parse", "--verify", f"{commit}:{relative}"],
    ).decode("ascii", "strict").strip()
    current_blob = _git_output(
        repository_root,
        ["hash-object", f"--path={relative}", str(path)],
    ).decode("ascii", "strict").strip()
    return len(frozen_blob) == 40 and current_blob == frozen_blob


def _audit_scorer_manifest(
    repository_root: Path,
    path: Path,
    *,
    dev_protocol_git_sha: str,
) -> list[str]:
    """Require the evaluator manifest to be the frozen repository blob."""

    repository_root = Path(repository_root).resolve()
    path = Path(path).resolve()
    relative = "data/scorer_manifests/averitec_dev_gold.json"
    if not _trusted_repository_path(repository_root, path, relative) or not path.is_file():
        return ["scorer_manifest_path_untrusted"]
    try:
        if not _git_blob_matches(repository_root, path, relative, dev_protocol_git_sha):
            return ["scorer_manifest_git_blob_mismatch"]
    except (OSError, UnicodeError, subprocess.CalledProcessError):
        return ["scorer_manifest_git_blob_mismatch"]
    return []


def _git_status_paths(payload: bytes) -> list[str]:
    """Parse porcelain-v1 or name-status NUL output, retaining rename/copy sources."""

    tokens = payload.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        status: str
        path_bytes: bytes | None
        if len(token) >= 3 and token[2:3] == b" ":
            status = token[:2].decode("ascii", "replace")
            path_bytes = token[3:]
        else:
            status_text, separator, path_bytes = token.partition(b"\t")
            status = status_text.decode("ascii", "replace")
            if not separator:
                # ``git diff --name-status -z`` emits a bare status token followed
                # by one path token (and two path tokens for rename/copy records).
                if status_text[:1] in b"MADRCUT?!" and (
                    len(status_text) == 1 or status_text[1:].isdigit()
                ):
                    path_bytes = None
                else:
                    path_bytes = b""
        if path_bytes is None:
            if index >= len(tokens) or not tokens[index]:
                raise ValueError("git status output missing path")
            path_bytes = tokens[index]
            index += 1
        if not path_bytes:
            continue
        paths.append(path_bytes.decode("utf-8", "surrogateescape"))
        if status[:1] in {"R", "C"} or status[1:2] in {"R", "C"}:
            if index >= len(tokens) or not tokens[index]:
                raise ValueError("git rename/copy output missing destination")
            paths.append(tokens[index].decode("utf-8", "surrogateescape"))
            index += 1
    return paths


def _publication_path_allowed(path: str) -> bool:
    normalized = path.replace("\\", "/").removeprefix("./")
    return normalized == "README.md" or normalized.startswith("reports/final/")


def _audit_publication_git(
    repository_root: Path,
    *,
    dev_protocol_git_sha: str,
    manifest_freeze_git_sha: str,
) -> list[str]:
    """Audit frozen ancestry and limit all publication changes to generated outputs."""

    repository_root = Path(repository_root).resolve()
    try:
        head = _git_output(repository_root, ["rev-parse", "--verify", "HEAD"]).decode(
            "ascii", "strict"
        ).strip()
        for frozen_sha in (dev_protocol_git_sha, manifest_freeze_git_sha):
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", frozen_sha, head],
                cwd=repository_root,
                check=True,
                capture_output=True,
            )
        committed_paths = _git_status_paths(
            _git_output(
                repository_root,
                ["diff", "--name-status", "-z", f"{dev_protocol_git_sha}..{head}"],
            )
        )
        if any(not _publication_path_allowed(path) for path in committed_paths):
            return ["current_git_worktree_dirty"]
        worktree_paths = _git_status_paths(
            _git_output(
                repository_root,
                ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
            )
        )
        if any(not _publication_path_allowed(path) for path in worktree_paths):
            return ["current_git_worktree_dirty"]
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError):
        return ["current_git_freeze_mismatch"]
    return []


def _nltk_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = (
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() != "PREPARATION_RECEIPT.json"
    )
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _audit_nltk_data(
    repository_root: Path,
    nltk_root: Path,
    *,
    dev_protocol_git_sha: str | None = None,
) -> list[str]:
    """Validate the pinned NLTK source receipt and extracted tree."""

    repository_root = Path(repository_root).resolve()
    nltk_root = Path(nltk_root).resolve()
    expected_root = (repository_root / "data" / "external" / "nltk").resolve()
    if nltk_root != expected_root:
        return ["nltk_data_invalid"]
    source_path = repository_root / "data" / "sources" / "nltk_data.json"
    receipt_path = nltk_root / "PREPARATION_RECEIPT.json"
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(source, dict) or not isinstance(receipt, dict):
            raise ValueError("NLTK metadata must be objects")
        if source.get("schema_version", "1") != "1" or receipt.get("schema_version") != "1":
            raise ValueError("unsupported NLTK metadata schema")
        repository = source.get("repository")
        commit = source.get("commit")
        files = source.get("files")
        if (
            not isinstance(repository, str)
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(char not in "0123456789abcdef" for char in commit)
            or not isinstance(files, dict)
            or not files
        ):
            raise ValueError("invalid NLTK source metadata")
        if receipt.get("repository") != repository or receipt.get("commit") != commit:
            raise ValueError("NLTK receipt identity mismatch")
        receipt_files = receipt.get("files")
        if not isinstance(receipt_files, dict) or set(receipt_files) != set(files):
            raise ValueError("NLTK receipt file set mismatch")
        for relative, metadata in files.items():
            if (
                not isinstance(relative, str)
                or not relative.startswith("packages/")
                or not relative.endswith(".zip")
                or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts
                or not isinstance(metadata, dict)
            ):
                raise ValueError("invalid NLTK source file")
            actual = receipt_files.get(relative)
            if not isinstance(actual, dict):
                raise ValueError("invalid NLTK receipt file")
            size = metadata.get("size")
            digest = metadata.get("sha256")
            expected_url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
            if (
                not isinstance(size, int)
                or size < 0
                or not _is_sha256(digest)
                or actual.get("size") != size
                or actual.get("sha256") != digest
                or actual.get("url") != expected_url
            ):
                raise ValueError("NLTK receipt metadata mismatch")
        if receipt.get("extracted_tree_sha256") != _nltk_tree_sha256(nltk_root):
            raise ValueError("NLTK extracted tree mismatch")
        if dev_protocol_git_sha is not None:
            if not _git_blob_matches(
                repository_root,
                source_path,
                "data/sources/nltk_data.json",
                dev_protocol_git_sha,
            ):
                raise ValueError("NLTK source spec is not frozen")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError):
        return ["nltk_data_invalid"]
    return []


def _inside(root: Path, relpath: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / relpath).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"artifact path leaves activity directory: {relpath}") from exc
    if not resolved.is_file():
        raise ValueError(f"artifact file is missing: {relpath}")
    return resolved


def _load_models(
    report_input: ReportInput,
) -> tuple[ActivityRecord, CampaignPlan, CampaignState, Path, Path]:
    activity_dir = Path(report_input.activity_dir).resolve()
    activity_path = activity_dir / "activity.json"
    plan_path = activity_dir / "plan.json"
    state_path = activity_dir / "campaign.json"
    if not all(path.is_file() for path in (activity_path, plan_path, state_path)):
        raise ValueError(
            "activity directory must contain activity.json, plan.json, and campaign.json"
        )
    # Activity is the only lifecycle journal with a mandatory digest sidecar.  Loading it
    # through the lifecycle boundary prevents a raw JSON edit from becoming publishable.
    activity = load_activity(activity_path)
    plan = CampaignPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    state = CampaignState.model_validate_json(state_path.read_text(encoding="utf-8"))
    expected_fingerprint = campaign_fingerprint(plan)
    if plan.campaign_fingerprint != expected_fingerprint:
        raise ValueError("campaign plan fingerprint mismatch")
    if (activity.activity_id, state.activity_id, state.campaign_id) != (
        plan.activity_id,
        plan.activity_id,
        plan.campaign_id,
    ):
        raise ValueError("activity, plan, and state identities differ")
    expected_run_ids = [item.run_id for item in plan.schedule]
    actual_run_ids = [item.run_id for item in state.items]
    if actual_run_ids != expected_run_ids:
        raise ValueError("campaign state order differs from immutable plan")
    if len(state.items) != len(plan.schedule):
        raise ValueError("campaign state item count differs from immutable plan")
    for work, item in zip(plan.schedule, state.items, strict=True):
        if item.artifact_relpath != f"artifacts/{work.run_id}.json":
            raise ValueError("campaign state artifact path differs from immutable plan")
    return activity, plan, state, plan_path, state_path


def _load_artifacts(
    activity_dir: Path,
    plan: CampaignPlan,
    state: CampaignState,
) -> tuple[dict[str, RunArtifact], dict[str, Path]]:
    work_by_id = {item.run_id: item for item in plan.schedule}
    artifacts: dict[str, RunArtifact] = {}
    paths: dict[str, Path] = {}
    for item_state in state.items:
        if item_state.artifact_sha256 is None:
            continue
        path = _inside(activity_dir, item_state.artifact_relpath)
        artifact = RunArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        verify_artifact_fingerprint(artifact)
        work = work_by_id[item_state.run_id]
        if (
            artifact.artifact_sha256 != item_state.artifact_sha256
            or artifact.run_id != work.run_id
            or artifact.claim_id != work.claim_id
            or artifact.strategy is not work.strategy
            or artifact.phase != work.phase
            or artifact.repeat != work.repeat
            or artifact.activity_id != plan.activity_id
            or artifact.campaign_id != plan.campaign_id
        ):
            raise ValueError(f"artifact identity mismatch: {item_state.run_id}")
        expected_work_status = result_status_to_work_status(artifact.result.status)
        if item_state.status is not expected_work_status and item_state.status.value not in {
            "stopped",
            "interrupted",
        }:
            raise ValueError(f"campaign item status differs from artifact: {item_state.run_id}")
        if item_state.status.value == "stopped" and not artifact.diagnostic_only:
            raise ValueError(
                f"stopped campaign item lacks diagnostic artifact: {item_state.run_id}"
            )
        artifacts[artifact.run_id] = artifact
        paths[artifact.run_id] = path
    return artifacts, paths


def _evidence_path(explicit: Path | None, default: Path) -> Path:
    return Path(explicit).resolve() if explicit is not None else default.resolve()


def _load_call_nodes(path: Path | None) -> dict[str, str]:
    """Read the non-secret node label used to measure actual LLM-router calls."""

    if path is None or not path.is_file():
        return {}
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = connection.execute("SELECT call_id, node FROM calls").fetchall()
        return {str(call_id): str(node) for call_id, node in rows}
    except (OSError, sqlite3.DatabaseError):
        return {}
    finally:
        if "connection" in locals():
            connection.close()


def _audit_run_store(
    path: Path,
    *,
    activity: ActivityRecord,
    plan: CampaignPlan,
    calibration_cases: list[Any],
    artifacts: dict[str, RunArtifact],
    pricing_path: Path,
) -> list[str]:
    """Audit immutable call accounting without needing mutable pricing configuration."""

    if not path.is_file():
        return ["run_store_missing"]
    try:
        pricing = load_price_config(pricing_path)
        SQLiteRunStore.open_existing(
            path,
            activity_id=activity.activity_id,
            cap_cny=plan.cap_micro_cny / 1_000_000,
            pricing=pricing,
        )
    except (OSError, ValueError):
        return ["run_store_invalid"]
    expected_calls: dict[str, tuple[str, str | None, str, frozenset[str]]] = {}
    expected_runs: dict[str, tuple[Usage, int | None, frozenset[str] | None]] = {}
    calibration_allowed_slots: dict[str, frozenset[tuple[str, str, int]]] = {}
    calibration_required_slots: dict[str, frozenset[tuple[str, str, int]]] = {}
    duplicate_expected_call = False
    duplicate_expected_run = False
    for case in calibration_cases:
        case_models = {model for model in case.response_model_ids_raw if model.strip()}
        case_model = next(iter(case_models)) if len(case_models) == 1 else None
        case_run_ids = frozenset((case.router_run_id, case.single_run_id, case.multi_run_id))
        calibration_allowed_slots[case.router_run_id] = frozenset({("router", "root", 0)})
        calibration_allowed_slots[case.single_run_id] = frozenset(
            {("single", "root", attempt) for attempt in (0, 1)}
        )
        calibration_allowed_slots[case.multi_run_id] = frozenset(
            {
                *(
                    (node, "root", attempt)
                    for node in ("decomposer", "judge")
                    for attempt in (0, 1)
                ),
                *(("worker", f"t{task}", attempt) for task in range(3) for attempt in (0, 1)),
            }
        )
        calibration_required_slots[case.router_run_id] = frozenset({("router", "root", 0)})
        calibration_required_slots[case.single_run_id] = frozenset({("single", "root", 0)})
        calibration_required_slots[case.multi_run_id] = frozenset(
            {("decomposer", "root", 0), ("judge", "root", 0)}
        )
        for run_id, usage, cost in (
            (case.router_run_id, case.router_usage, case.router_actual_cost_micro_cny),
            (
                case.single_run_id,
                case.single_result.usage,
                case.single_result.estimated_cost_micro_cny,
            ),
            (
                case.multi_run_id,
                case.multi_result.usage,
                case.multi_result.estimated_cost_micro_cny,
            ),
        ):
            if run_id in expected_runs:
                duplicate_expected_run = True
            expected_runs[run_id] = (usage, cost, None)
        for call_id in case.call_ids:
            if call_id in expected_calls:
                duplicate_expected_call = True
            expected_calls[call_id] = (
                case.requested_alias,
                case_model,
                case.request_sha256_by_call_id[call_id],
                case_run_ids,
            )
    for artifact in artifacts.values():
        if artifact.run_id in expected_runs:
            duplicate_expected_run = True
        expected_runs[artifact.run_id] = (
            artifact.usage,
            artifact.actual_cost_micro_cny,
            frozenset(artifact.call_ids),
        )
        for call_id in artifact.call_ids:
            if call_id in expected_calls:
                duplicate_expected_call = True
            expected_calls[call_id] = (
                artifact.requested_alias,
                (
                    artifact.response_model_ids_raw[0]
                    if len(artifact.response_model_ids_raw) == 1
                    else None
                ),
                "",
                frozenset((artifact.run_id,)),
            )
    if duplicate_expected_call or duplicate_expected_run:
        return ["run_store_call_set_mismatch"]
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        metadata = connection.execute(
            "SELECT activity_id, cap_micro_cny FROM store_metadata WHERE singleton=1"
        ).fetchone()
        if metadata is None or metadata["activity_id"] != activity.activity_id:
            return ["run_store_identity_mismatch"]
        if metadata["cap_micro_cny"] != plan.cap_micro_cny:
            return ["run_store_cap_mismatch"]
        # Read the complete ledger, not only rows matching the requested activity.  An unrelated
        # row in a shared file is evidence that the report is not bound to one immutable ledger.
        rows = connection.execute(
            "SELECT call_id, request_sha256, activity_id, run_id, node, task_id, "
            "logical_attempt, state, reserved_micro_cny, actual_micro_cny, usage_json, "
            "usage_source, requested_alias, response_model_id_raw, identity_verified, "
            "transport_attempts, cache_hits "
            "FROM calls",
        ).fetchall()
    except (OSError, sqlite3.DatabaseError):
        return ["run_store_invalid"]
    finally:
        if "connection" in locals():
            connection.close()
    row_by_id = {str(row["call_id"]): row for row in rows}
    if len(row_by_id) != len(rows):
        return ["run_store_call_set_mismatch"]
    if set(expected_calls) != set(row_by_id):
        return ["run_store_call_set_mismatch"]
    if activity.summary is None or set(activity.summary.call_ids) != set(row_by_id):
        return ["activity_summary_call_set_mismatch"]
    input_tokens = output_tokens = total_tokens = actual_cost = 0
    cache_hits = transport_attempts = 0
    aliases: set[str] = set()
    models: set[str] = set()
    reservation_invalid = False
    observed_by_run: dict[str, dict[str, object]] = {}
    observed_calibration_slots: dict[str, set[tuple[str, str, int]]] = {}
    for call_id, (alias, model_id, request_sha256, run_ids) in expected_calls.items():
        row = row_by_id[call_id]
        try:
            usage = Usage.model_validate_json(row["usage_json"] or "null")
        except (TypeError, ValueError):
            return ["run_store_accounting_invalid"]
        if (
            row["state"] != "completed"
            or not isinstance(row["actual_micro_cny"], int)
            or not usage.complete
            or row["usage_source"] != "provider"
            or row["activity_id"] != activity.activity_id
            or row["requested_alias"] != alias
            or model_id is None
            or row["response_model_id_raw"] != model_id
            or row["run_id"] not in run_ids
            or row["identity_verified"] != 0
            or not _is_sha256(row["request_sha256"])
            or not isinstance(row["node"], str)
            or not row["node"]
            or not isinstance(row["task_id"], str)
            or not row["task_id"]
            or not isinstance(row["logical_attempt"], int)
            or make_call_id(row["run_id"], row["node"], row["task_id"], row["logical_attempt"])
            != call_id
            or (request_sha256 and row["request_sha256"] != request_sha256)
            or pricing.estimate_micro_cny(usage) != row["actual_micro_cny"]
        ):
            return ["run_store_accounting_invalid"]
        run_id = str(row["run_id"])
        allowed_slots = calibration_allowed_slots.get(run_id)
        if allowed_slots is not None:
            slot = (str(row["node"]), str(row["task_id"]), int(row["logical_attempt"]))
            observed_slots = observed_calibration_slots.setdefault(run_id, set())
            if slot not in allowed_slots or slot in observed_slots:
                return ["run_store_accounting_invalid"]
            observed_slots.add(slot)
        if (
            not isinstance(row["reserved_micro_cny"], int)
            or row["reserved_micro_cny"] < row["actual_micro_cny"]
        ):
            reservation_invalid = True
        run_accounting = observed_by_run.setdefault(
            str(row["run_id"]),
            {
                "call_ids": set(),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "actual_cost_micro_cny": 0,
            },
        )
        run_accounting["call_ids"].add(call_id)
        run_accounting["input_tokens"] += usage.input_tokens
        run_accounting["output_tokens"] += usage.output_tokens
        run_accounting["total_tokens"] += usage.total_tokens
        run_accounting["actual_cost_micro_cny"] += int(row["actual_micro_cny"])
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        total_tokens += usage.total_tokens
        actual_cost += int(row["actual_micro_cny"])
        cache_hits += int(row["cache_hits"])
        transport_attempts += int(row["transport_attempts"])
        aliases.add(str(row["requested_alias"]))
        models.add(str(row["response_model_id_raw"]))
    for run_id, required_slots in calibration_required_slots.items():
        observed_slots = observed_calibration_slots.get(run_id, set())
        if not required_slots.issubset(observed_slots) or any(
            logical_attempt == 1 and (node, task_id, 0) not in observed_slots
            for node, task_id, logical_attempt in observed_slots
        ):
            return ["run_store_accounting_invalid"]
    for run_id, (expected_usage, expected_cost, expected_call_ids) in expected_runs.items():
        observed = observed_by_run.get(
            run_id,
            {
                "call_ids": set(),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "actual_cost_micro_cny": 0,
            },
        )
        if (
            observed["input_tokens"] != expected_usage.input_tokens
            or observed["output_tokens"] != expected_usage.output_tokens
            or observed["total_tokens"] != expected_usage.total_tokens
            or observed["actual_cost_micro_cny"] != expected_cost
            or (expected_call_ids is not None and observed["call_ids"] != set(expected_call_ids))
        ):
            return ["run_store_evidence_accounting_mismatch"]
    if reservation_invalid:
        return ["run_store_accounting_invalid"]
    summary = activity.summary
    expected_run_ids = [
        run_id
        for case in calibration_cases
        for run_id in (case.router_run_id, case.single_run_id, case.multi_run_id)
    ]
    expected_run_ids.extend(item.run_id for item in plan.schedule)
    if (
        summary.run_ids != expected_run_ids
        or summary.usage.model_dump(mode="json")
        != {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "complete": True,
        }
        or summary.actual_cost_micro_cny != actual_cost
        or summary.known_actual_cost_micro_cny != actual_cost
        or summary.committed_cost_micro_cny != actual_cost
        or summary.fresh_call_count != len(rows)
        or summary.cache_hit_count != cache_hits
        or summary.transport_attempts != transport_attempts
        or summary.usage_sources != ["provider"] * len(rows)
        or set(summary.requested_aliases) != aliases
        or set(summary.response_model_ids_raw) != models
        or summary.cost_is_lower_bound
        or summary.billing_uncertain
    ):
        return ["activity_summary_accounting_mismatch"]
    return []


def _audit_calibration_evidence(
    report_input: ReportInput,
    activity: ActivityRecord,
    campaign_plan: CampaignPlan,
    artifacts: dict[str, RunArtifact],
) -> list[str]:
    """Bind calibration collection, replay and paid campaign to one activity freeze."""

    activity_dir = Path(report_input.activity_dir).resolve()
    repository_root = Path(report_input.repository_root).resolve()
    plan_path = _evidence_path(
        report_input.calibration_plan, activity_dir / "calibration-plan.json"
    )
    state_path = _evidence_path(
        report_input.calibration_state, activity_dir / "calibration-state.json"
    )
    report_path = _evidence_path(
        report_input.calibration_report,
        repository_root / "reports" / "calibration" / "calibration_report.json",
    )
    replay_path = _evidence_path(
        report_input.calibration_replay, activity_dir / "calibration-replay.json"
    )
    config_path = _evidence_path(
        report_input.calibrated_config, repository_root / "configs" / "calibrated.yaml"
    )
    calibration_runtime_path = _evidence_path(
        report_input.calibration_runtime_manifest,
        repository_root / "data/manifests/averitec_calibration_runtime.json",
    )
    if not plan_path.is_file():
        return ["calibration_plan_missing"]
    if not state_path.is_file():
        return ["calibration_state_missing"]
    if not report_path.is_file():
        return ["calibration_report_missing"]
    if not replay_path.is_file():
        return ["calibration_replay_metadata_missing"]
    if not config_path.is_file():
        return ["calibrated_config_missing"]
    try:
        calibration_plan = load_calibration_plan(plan_path)
        calibration_runtime = load_runtime_manifest(
            calibration_runtime_path, allowed_root=calibration_runtime_path.parent
        )
        calibration_state = load_calibration_state(state_path, expected_plan=calibration_plan)
        if any(
            not (activity_dir / item.artifact_relpath).is_file() for item in calibration_state.items
        ):
            return ["calibration_case_missing"]
        calibration_cases = load_calibration_cases(
            activity_dir, calibration_plan, calibration_state, require_complete=True
        )
        calibration_report = CalibrationReplay.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
        replay_metadata = json.loads(replay_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if "runtime manifest" in str(exc).lower() or "sidecar" in str(exc).lower():
            return ["calibration_runtime_manifest_invalid"]
        return ["calibration_evidence_invalid"]
    freeze = campaign_plan.freeze
    reasons: list[str] = []
    if calibration_plan.activity_id != activity.activity_id:
        reasons.append("calibration_activity_id_mismatch")
    if _sha256(calibration_runtime_path) != calibration_plan.runtime_manifest_sha256 or [
        item.claim_id for item in calibration_runtime.items
    ] != [item.claim_id for item in calibration_plan.items]:
        reasons.append("calibration_runtime_manifest_mismatch")
    if activity.calibration_plan_sha256 != _sha256(plan_path):
        reasons.append("calibration_plan_sha256_mismatch")
    if activity.calibration_state_sha256 != _sha256(state_path):
        reasons.append("calibration_state_sha256_mismatch")
    state_links = [(item.case_id, item.artifact_sha256) for item in calibration_state.items]
    if state_links != [
        (link.case_id, link.artifact_sha256) for link in activity.calibration_artifacts
    ] or state_links != [(case.case_id, case.artifact_sha256) for case in calibration_cases]:
        reasons.append("calibration_artifact_links_mismatch")
    frozen_values = (
        calibration_plan.manifest_freeze_git_sha,
        calibration_plan.runtime_manifest_sha256,
        calibration_plan.corpus_preparation_receipt_sha256,
        calibration_plan.prompt_bundle_sha256,
        calibration_plan.pricing_sha256,
        calibration_plan.endpoint_config_sha256,
        calibration_plan.requirements_lock_sha256,
        calibration_plan.requested_alias,
        calibration_plan.seed,
    )
    campaign_values = (
        freeze.manifest_freeze_git_sha,
        freeze.calibration_runtime_manifest_sha256,
        freeze.corpus_preparation_receipt_sha256,
        freeze.prompt_bundle_sha256,
        freeze.pricing_sha256,
        freeze.endpoint_config_sha256,
        freeze.requirements_lock_sha256,
        freeze.requested_alias,
        freeze.seed,
    )
    if frozen_values != campaign_values:
        reasons.append("calibration_freeze_mismatch")
    if (
        calibration_report.activity_id != activity.activity_id
        or calibration_report.runtime_manifest_sha256 != calibration_plan.runtime_manifest_sha256
        or calibration_report.prompt_bundle_sha256 != calibration_plan.prompt_bundle_sha256
        or calibration_report.calibrated_config_sha256 != freeze.config_sha256
        or _sha256(config_path) != freeze.config_sha256
        or replay_metadata.get("activity_id") != activity.activity_id
        or replay_metadata.get("plan_fingerprint") != calibration_plan.plan_fingerprint
        or replay_metadata.get("runtime_manifest_sha256")
        != calibration_plan.runtime_manifest_sha256
        or replay_metadata.get("calibration_report_sha256") != _sha256(report_path)
        or replay_metadata.get("calibrated_config_sha256") != freeze.config_sha256
        or replay_metadata.get("selected_config_hash") != calibration_report.selected.config_hash
    ):
        reasons.append("calibration_replay_identity_mismatch")
    aliases = {case.requested_alias for case in calibration_cases}
    aliases.update(artifact.requested_alias for artifact in artifacts.values())
    if activity.summary is not None:
        aliases.update(activity.summary.requested_aliases)
    models = {
        *(model for case in calibration_cases for model in case.response_model_ids_raw),
        *(model for artifact in artifacts.values() for model in artifact.response_model_ids_raw),
        *activity.observed_response_model_ids_raw,
    }
    if aliases != {freeze.requested_alias}:
        reasons.append("requested_alias_drift")
    if len(models) != 1:
        reasons.append("response_model_id_drift")
    run_store = _evidence_path(
        report_input.run_store, repository_root / "artifacts" / "gate-a-run-store.sqlite3"
    )
    reasons.extend(
        _audit_run_store(
            run_store,
            activity=activity,
            plan=campaign_plan,
            calibration_cases=calibration_cases,
            artifacts=artifacts,
            pricing_path=_evidence_path(
                report_input.pricing, repository_root / "configs/pricing.local.yaml"
            ),
        )
    )
    return reasons


def _audit_current_freeze_inputs(report_input: ReportInput, plan: CampaignPlan) -> list[str]:
    repository_root = Path(report_input.repository_root).resolve()
    freeze = plan.freeze
    pricing_env = os.environ.get("EVIDENCE_ROUTE_PRICE_FILE")
    inputs = (
        (
            "calibration_runtime_manifest",
            _evidence_path(
                report_input.calibration_runtime_manifest,
                repository_root / "data/manifests/averitec_calibration_runtime.json",
            ),
            freeze.calibration_runtime_manifest_sha256,
        ),
        (
            "stability_runtime_manifest",
            _evidence_path(
                report_input.stability_runtime_manifest,
                repository_root / "data/manifests/averitec_stability_runtime.json",
            ),
            freeze.stability_runtime_manifest_sha256,
        ),
        (
            "corpus_preparation_receipt",
            _evidence_path(
                report_input.corpus_preparation_receipt,
                repository_root / "data/processed/averitec/preparation_receipt.json",
            ),
            freeze.corpus_preparation_receipt_sha256,
        ),
        (
            "prompt_bundle",
            _evidence_path(
                report_input.prompt_bundle,
                repository_root / "src/evidence_route/prompts.py",
            ),
            freeze.prompt_bundle_sha256,
        ),
        (
            "pricing",
            _evidence_path(
                report_input.pricing,
                Path(pricing_env)
                if pricing_env
                else repository_root / "configs/pricing.local.yaml",
            ),
            freeze.pricing_sha256,
        ),
        (
            "requirements_lock",
            _evidence_path(report_input.requirements_lock, repository_root / "requirements.lock"),
            freeze.requirements_lock_sha256,
        ),
    )
    reasons = [
        f"current_{label}_missing" if not path.is_file() else f"current_{label}_sha256_mismatch"
        for label, path, expected in inputs
        if not path.is_file() or _sha256(path) != expected
    ]
    if report_input.require_git:
        git_marker = repository_root / ".git"
        if not git_marker.exists():
            reasons.append("current_git_missing")
        else:
            reasons.extend(
                _audit_publication_git(
                    repository_root,
                    dev_protocol_git_sha=freeze.dev_protocol_git_sha,
                    manifest_freeze_git_sha=freeze.manifest_freeze_git_sha,
                )
            )
        reasons.extend(
            _audit_scorer_manifest(
                repository_root,
                Path(report_input.gold_manifest),
                dev_protocol_git_sha=freeze.dev_protocol_git_sha,
            )
        )
        reasons.extend(
            _audit_nltk_data(
                repository_root,
                Path(report_input.nltk_data_root),
                dev_protocol_git_sha=freeze.dev_protocol_git_sha,
            )
        )
    # Reports deliberately do not require credentials.  When an endpoint/alias is present in
    # the environment, however, audit it against the frozen digest instead of silently ignoring
    # an operator's changed target.
    current_base_url = os.environ.get("EVIDENCE_ROUTE_BASE_URL")
    current_alias = os.environ.get("EVIDENCE_ROUTE_MODEL")
    if current_base_url:
        actual_endpoint = endpoint_config_hash({"base_url": ensure_v1(current_base_url)})
        if actual_endpoint != freeze.endpoint_config_sha256:
            reasons.append("current_endpoint_config_sha256_mismatch")
    if current_alias and current_alias != freeze.requested_alias:
        reasons.append("current_requested_alias_mismatch")
    return reasons


def _strategy_summary(
    strategy: Strategy,
    gold: dict[str, Verdict],
    artifacts: list[RunArtifact],
    *,
    call_nodes: dict[str, str] | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    call_nodes = call_nodes or {}
    results = {
        artifact.claim_id: artifact.result
        for artifact in artifacts
        if artifact.phase == "dev" and artifact.strategy is strategy and artifact.repeat == 0
    }
    full = score_full_manifest(gold, results)
    conditional = score_completed_conditionally(gold, results)
    full_payload = full.model_dump(mode="json")
    operations = []
    for artifact in artifacts:
        if not (artifact.phase == "dev" and artifact.strategy is strategy and artifact.repeat == 0):
            continue
        route_source = (
            "llm"
            if any(call_nodes.get(call_id) == "router" for call_id in artifact.call_ids)
            else "strategy"
            if strategy in {Strategy.ALWAYS_SINGLE, Strategy.ALWAYS_MULTI}
            else "rule_or_fallback"
        )
        result = artifact.result
        citation_valid: bool | None = None
        if result.status in {ResultStatus.COMPLETED, ResultStatus.PARTIAL}:
            available = set(result.available_evidence_ids)
            seen: set[str] = set()
            citation_valid = True
            for citation in result.citations:
                if (
                    citation.evidence_id not in available
                    or citation.evidence_id in seen
                    or not citation.claim_unit_ids
                    or not citation.question.strip()
                    or not citation.answer.strip()
                    or not citation.quote.strip()
                ):
                    citation_valid = False
                seen.add(citation.evidence_id)
            # Empty citations are a valid completed result for the official adapter; when
            # citations exist, all structural fields above must be valid.
            if not result.citations:
                citation_valid = True
        operation = {
            "usage": artifact.usage.model_dump(mode="json"),
            "estimated_cost_micro_cny": artifact.actual_cost_micro_cny,
            "fresh_end_to_end_ms": artifact.latency.fresh_end_to_end_ms,
            "route": result.initial_route,
            "route_source": route_source,
            "escalated": result.escalated,
            "cache_hit_count": artifact.cache_hit_count,
            "errors": result.errors,
        }
        if citation_valid is not None:
            operation["citation_valid"] = citation_valid
        operations.append(operation)
    operation_summary = summarize_operations(operations)
    route_not_reached = sum(
        artifact.result.initial_route is None
        for artifact in artifacts
        if artifact.phase == "dev" and artifact.strategy is strategy and artifact.repeat == 0
    )
    route_denominator = max(0, len(operations) - route_not_reached)
    route_counts = Counter(
        artifact.result.initial_route
        for artifact in artifacts
        if artifact.phase == "dev"
        and artifact.strategy is strategy
        and artifact.repeat == 0
        and artifact.result.initial_route is not None
    )
    status_counts = Counter(result.status.value for result in results.values())
    errors: Counter[str] = Counter()
    for result in results.values():
        errors.update(result.errors)
    payload: dict[str, object] = {
        "full_manifest_macro_f1": full.macro_f1,
        "full_manifest_accuracy": full.accuracy,
        "completed_conditional_macro_f1": conditional.macro_f1,
        "completed_conditional_accuracy": conditional.accuracy,
        "full_manifest_per_class": full_payload["per_class"],
        "full_manifest_confusion_labels": full_payload["confusion_labels"],
        "full_manifest_confusion_matrix": full_payload["confusion_matrix"],
        "partial_count": full.partial_count,
        "failed_count": full.failed_count,
        "missing_count": full.missing_count,
        "completion_rate": full.completion_rate,
        "completed_count": full.completed_count,
        "manifest_count": full.sample_count,
        "official_completed_subset_size": conditional.sample_count,
        "input_tokens": operation_summary.input_tokens,
        "output_tokens": operation_summary.output_tokens,
        "total_tokens": operation_summary.total_tokens,
        "mean_tokens_per_claim": operation_summary.mean_tokens_per_claim,
        "actual_cost_micro_cny": operation_summary.exact_cost_micro_cny,
        "fresh_latency_p50_ms": operation_summary.latency_p50_ms,
        "fresh_latency_p95_ms": operation_summary.latency_p95_ms,
        "cache_hit_count": operation_summary.cache_hit_count,
        "route_counts": dict(route_counts),
        "route_not_reached_count": route_not_reached,
        "route_not_reached_rate": route_not_reached / len(operations) if operations else 0.0,
        "single_route_rate": (
            route_counts["single"] / route_denominator if route_denominator else 0.0
        ),
        "multi_route_rate": (
            route_counts["multi"] / route_denominator if route_denominator else 0.0
        ),
        "escalation_rate": operation_summary.escalation_rate,
        "llm_router_rate": operation_summary.llm_router_rate,
        "citation_validity_rate": operation_summary.citation_validity_rate,
        "status_counts": dict(status_counts),
        "failure_reasons": dict(errors),
        "token_reduction_pct": 0.0,
        "cost_reduction_pct": 0.0,
    }
    return payload, full.scored_predictions


def _audit_campaign_schedule(report_input: ReportInput, plan: CampaignPlan) -> list[str]:
    """Rebuild the immutable schedule from the claim-only manifests and frozen seed."""

    root = Path(report_input.repository_root).resolve()
    dev_path = _evidence_path(
        report_input.runtime_manifest, root / "data/manifests/averitec_dev_runtime.json"
    )
    stability_path = _evidence_path(
        report_input.stability_runtime_manifest,
        root / "data/manifests/averitec_stability_runtime.json",
    )
    try:
        dev = load_runtime_manifest(dev_path, allowed_root=dev_path.parent)
        stability = load_runtime_manifest(stability_path, allowed_root=stability_path.parent)
    except (OSError, UnicodeDecodeError, ValueError):
        return ["runtime_manifest_invalid"]
    reasons: list[str] = []
    if _sha256(dev_path) != plan.freeze.dev_runtime_manifest_sha256:
        reasons.append("dev_runtime_manifest_sha256_mismatch")
    if _sha256(stability_path) != plan.freeze.stability_runtime_manifest_sha256:
        reasons.append("stability_runtime_manifest_sha256_mismatch")
    if len(dev.items) != 80 or len(stability.items) != 20:
        reasons.append("runtime_manifest_cohort_size_mismatch")
        return reasons
    try:
        expected_schedule, expected_links = build_campaign_schedule(
            dev.items,
            stability_runtime_claims=stability.items,
            seed=plan.freeze.seed,
            campaign_id=plan.campaign_id,
        )
    except (TypeError, ValueError):
        return [*reasons, "campaign_schedule_invalid"]
    actual_items = [
        (item.order, item.phase, item.claim_id, item.strategy.value, item.repeat, item.run_id)
        for item in plan.schedule
    ]
    expected_items = [
        (item.order, item.phase, item.claim_id, item.strategy.value, item.repeat, item.run_id)
        for item in expected_schedule
    ]
    if actual_items != expected_items:
        reasons.append("campaign_schedule_mismatch")
    actual_links = [
        (link.claim_id, link.dev_adaptive_run_id) for link in plan.stability_repeat_zero_links
    ]
    if actual_links != [(link["claim_id"], link["dev_adaptive_run_id"]) for link in expected_links]:
        reasons.append("stability_links_mismatch")
    return reasons


def _publication_gate(
    report_input: ReportInput,
    activity: ActivityRecord,
    plan: CampaignPlan,
    state: CampaignState,
    plan_path: Path,
    state_path: Path,
    artifacts: dict[str, RunArtifact],
) -> PublicationGateResult:
    reasons: list[str] = []
    if activity.status is not CampaignStatus.COMPLETE:
        reasons.append(f"activity_status={activity.status.value}")
    for phase, status in (
        ("calibration", activity.calibration_status),
        ("dev", activity.dev_status),
        ("stability", activity.stability_status),
    ):
        if status is not CampaignStatus.COMPLETE:
            reasons.append(f"{phase}_status={status.value}")
    if state.status is not CampaignStatus.COMPLETE:
        reasons.append(f"campaign_status={state.status.value}")
    if len(activity.calibration_artifacts) != 32 or any(
        link.artifact_sha256 is None for link in activity.calibration_artifacts
    ):
        reasons.append("calibration_artifacts_not_closed")
    dev_items = [item for item in plan.schedule if item.phase == "dev"]
    stability_items = [item for item in plan.schedule if item.phase == "stability"]
    if len(dev_items) != 240:
        reasons.append(f"dev_item_count={len(dev_items)}")
    if len(stability_items) != 40:
        reasons.append(f"stability_item_count={len(stability_items)}")
    if len(artifacts) != len(plan.schedule):
        reasons.append(f"artifact_count={len(artifacts)}")
    if activity.campaign_plan_sha256 != _sha256(plan_path):
        reasons.append("campaign_plan_sha256_mismatch")
    if activity.campaign_state_sha256 != _sha256(state_path):
        reasons.append("campaign_state_sha256_mismatch")
    if activity.freeze != plan.freeze:
        reasons.append("freeze_identity_mismatch")
    if activity.billing_uncertain or state.billing_uncertain:
        reasons.append("billing_uncertain")
    if activity.summary is None or not activity.summary.usage.complete:
        reasons.append("activity_usage_incomplete")
    if activity.summary is not None:
        if len(activity.summary.requested_aliases) != 1:
            reasons.append("requested_alias_count_not_one")
        if activity.summary.billing_uncertain:
            reasons.append("summary_billing_uncertain")
    campaign_model_ids = {
        model_id
        for artifact in artifacts.values()
        for model_id in artifact.response_model_ids_raw
        if model_id.strip()
    }
    activity_model_ids = set(activity.observed_response_model_ids_raw)
    state_model_ids = set(state.observed_response_model_ids_raw)
    ledger_model_ids = (
        set(activity.summary.response_model_ids_raw) if activity.summary is not None else set()
    )
    if (
        len(campaign_model_ids) > 1
        or len(activity_model_ids) != 1
        or state_model_ids != activity_model_ids
        or ledger_model_ids != activity_model_ids
        or not campaign_model_ids.issubset(activity_model_ids)
    ):
        reasons.append("response_model_id_count_not_one")
    if any(
        not artifact.usage.complete
        or artifact.actual_cost_micro_cny is None
        or artifact.billing_uncertain
        or artifact.diagnostic_only
        for artifact in artifacts.values()
    ):
        reasons.append("artifact_accounting_incomplete")
    item_by_run_id = {item.run_id: item for item in state.items}
    expected_baselines: dict[str, str] = {}
    for link in plan.stability_repeat_zero_links:
        dev_item = item_by_run_id.get(link.dev_adaptive_run_id)
        if dev_item is None or dev_item.artifact_sha256 is None:
            continue
        expected_baselines[link.claim_id] = dev_item.artifact_sha256
    if state.stability_repeat_zero_artifact_sha256s != expected_baselines:
        reasons.append("stability_baseline_links_mismatch")
    reasons.extend(_audit_campaign_schedule(report_input, plan))
    runtime_path = _evidence_path(
        report_input.runtime_manifest,
        Path(report_input.repository_root) / "data" / "manifests" / "averitec_dev_runtime.json",
    )
    if not runtime_path.is_file():
        reasons.append("dev_runtime_manifest_missing")
    elif _sha256(runtime_path) != plan.freeze.dev_runtime_manifest_sha256:
        reasons.append("dev_runtime_manifest_sha256_mismatch")
    reasons.extend(_audit_calibration_evidence(report_input, activity, plan, artifacts))
    reasons.extend(_audit_current_freeze_inputs(report_input, plan))
    return PublicationGateResult(publishable=not reasons, reasons=reasons)


def _official_summary(
    report_input: ReportInput,
    gold_manifest: Any,
    artifacts: dict[str, RunArtifact],
    strategies: dict[str, dict[str, object]],
    *,
    enabled: bool,
) -> dict[str, object]:
    adaptive = strategies[Strategy.ADAPTIVE.value]
    runtime_path = _evidence_path(
        report_input.runtime_manifest,
        Path(report_input.repository_root) / "data" / "manifests" / "averitec_dev_runtime.json",
    )
    if not enabled:
        unavailable = {
            "available": False,
            "reason": "publication evidence gate did not pass",
            "completed_count": adaptive["official_completed_subset_size"],
            "full_count": adaptive["manifest_count"],
            "completion_rate": adaptive["completion_rate"],
        }
        return {
            "shared_task_2024": unavailable,
            "paper_2023_secondary": dict(unavailable),
        }
    if not runtime_path.is_file():
        unavailable = {
            "available": False,
            "reason": "dev runtime manifest was not supplied",
            "completed_count": adaptive["official_completed_subset_size"],
            "full_count": adaptive["manifest_count"],
            "completion_rate": adaptive["completion_rate"],
        }
        return {
            "shared_task_2024": unavailable,
            "paper_2023_secondary": dict(unavailable),
        }
    activity_dir = Path(report_input.activity_dir).resolve()
    output_dir = _evidence_path(report_input.official_output_dir, activity_dir / "official")
    adaptive_artifacts_sha256 = hashlib.sha256(
        json.dumps(
            sorted(
                (artifact.run_id, artifact.artifact_sha256)
                for artifact in artifacts.values()
                if artifact.phase == "dev"
                and artifact.strategy is Strategy.ADAPTIVE
                and artifact.repeat == 0
            ),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    evidence = {
        "activity_id": json.loads((activity_dir / "activity.json").read_text(encoding="utf-8"))[
            "activity_id"
        ],
        "runtime_manifest_sha256": _sha256(runtime_path),
        "gold_manifest_sha256": _sha256(Path(report_input.gold_manifest)),
        "adaptive_artifacts_sha256": adaptive_artifacts_sha256,
    }
    summary_path = output_dir / "summary.json"
    if report_input.allow_official_cache and summary_path.is_file():
        try:
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("report_evidence") == evidence
            and all(
                isinstance(cached.get(name), dict) and cached[name].get("available") is True
                for name in ("shared_task_2024", "paper_2023_secondary")
            )
        ):
            return cached
    try:
        runtime = load_runtime_manifest(runtime_path, allowed_root=runtime_path.parent)
        aligned = align_runtime_and_gold(runtime, gold_manifest)
        adaptive_results = {
            artifact.claim_id: artifact.result
            for artifact in artifacts.values()
            if artifact.phase == "dev"
            and artifact.strategy is Strategy.ADAPTIVE
            and artifact.repeat == 0
        }
        selection = select_completed_triples(aligned, adaptive_results)
        payload = run_official_evaluators(
            selection,
            output_dir=output_dir,
            nltk_data=report_input.nltk_data_root,
        )
        payload["report_evidence"] = evidence
        _write_json(summary_path, payload)
        return payload
    except (OSError, ValueError, RuntimeError) as exc:
        unavailable = {
            "available": False,
            "reason": str(exc),
            "completed_count": adaptive["official_completed_subset_size"],
            "full_count": adaptive["manifest_count"],
            "completion_rate": adaptive["completion_rate"],
        }
        return {
            "shared_task_2024": unavailable,
            "paper_2023_secondary": dict(unavailable),
        }


def _representative_sources(
    artifacts: dict[str, RunArtifact], paths: dict[str, Path]
) -> tuple[Path, ...]:
    categories = {
        "single": lambda artifact: (
            artifact.phase == "dev"
            and artifact.result.status is ResultStatus.COMPLETED
            and artifact.result.initial_route == "single"
            and not artifact.result.escalated
        ),
        "multi": lambda artifact: (
            artifact.phase == "dev"
            and artifact.result.status is ResultStatus.COMPLETED
            and artifact.result.initial_route == "multi"
        ),
        "escalation": lambda artifact: artifact.phase == "dev" and artifact.result.escalated,
        "failure": lambda artifact: (
            artifact.phase == "dev"
            and artifact.result.status in {ResultStatus.PARTIAL, ResultStatus.FAILED}
        ),
    }
    selected: list[Path] = []
    seen: set[Path] = set()
    for predicate in categories.values():
        matches = sorted(
            (artifact for artifact in artifacts.values() if predicate(artifact)),
            key=lambda artifact: artifact.run_id,
        )
        if matches:
            path = paths[matches[0].run_id]
            if path not in seen:
                seen.add(path)
                selected.append(path)
    return tuple(selected)


def _render_markdown(summary: dict[str, object]) -> str:
    strategies = summary["strategies"]
    lines = [
        "# EvidenceRoute Gate A Evaluation",
        "",
        "## Scope",
        "",
        f"{summary['benchmark_name']}. This is a frozen balanced subset evaluation and must not "
        "be generalized beyond this cohort.",
        "",
        "## Reproducibility",
        "",
        f"- Model identity: {summary['reproducibility']['model_identity']}",
        f"- Activity status: {summary['activity_status']}",
        "",
        "## Full-Manifest Results",
        "",
        "| Strategy | Macro-F1 | Accuracy | Completion |",
        "|---|---:|---:|---:|",
    ]
    for name in ("always_single", "always_multi", "adaptive"):
        values = strategies[name]
        lines.append(
            f"| {name} | {values['full_manifest_macro_f1']:.3f} | "
            f"{values['full_manifest_accuracy']:.3f} | {values['completion_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Completed-Only Official Results",
            "",
            "Official evaluator numbers are completed-only and always retain the full-manifest "
            "completion denominator.",
            "",
            "Shared-task evaluator: "
            f"{json.dumps(summary['official']['shared_task_2024'], sort_keys=True)}.",
            "Paper evaluator: "
            f"{json.dumps(summary['official']['paper_2023_secondary'], sort_keys=True)}.",
            "",
            "## Cost/Latency",
            "",
            f"Adaptive tokens: {strategies['adaptive']['total_tokens']}; exact cost: "
            f"{strategies['adaptive']['actual_cost_micro_cny']}; P50/P95 fresh latency: "
            f"{strategies['adaptive']['fresh_latency_p50_ms']}/"
            f"{strategies['adaptive']['fresh_latency_p95_ms']} ms; "
            f"token reduction versus always_multi: "
            f"{strategies['adaptive']['token_reduction_pct']:.1f}%.",
            "",
            "## Routing",
            "",
            f"Adaptive route-not-reached: "
            f"{strategies['adaptive']['route_not_reached_count']}; LLM-router rate: "
            f"{strategies['adaptive']['llm_router_rate']:.1%}; citation validity: "
            f"{strategies['adaptive']['citation_validity_rate']!s}.",
            "Adaptive route mix: "
            f"single {strategies['adaptive']['single_route_rate']:.1%}, "
            f"multi {strategies['adaptive']['multi_route_rate']:.1%}, "
            f"escalated {strategies['adaptive']['escalation_rate']:.1%}.",
            "",
            "## Stability",
            "",
            f"Three-run verdict consistency: {summary['stability']['rate']:.1%} "
            f"({summary['stability']['consistent']}/{summary['stability']['total']}).",
            "",
            "## Failure Analysis",
            "",
            "Adaptive status counts: "
            f"{json.dumps(strategies['adaptive']['status_counts'], sort_keys=True)}.",
            "",
            "## Limitations",
            "",
            "The 3 percentage-point quality tolerance and 85% stability threshold are engineering "
            "gates, not statistical non-inferiority claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def _resume_snippet(summary: dict[str, object]) -> str:
    adaptive = summary["strategies"]["adaptive"]
    return (
        "- EvidenceRoute（事实核查 Agent）：在同一冻结证据与模型配置下实现 single/multi/"
        "adaptive 路由；在 AVeriTeC dev balanced subset (n=80) 上，adaptive 的 full-manifest "
        f"macro-F1 为 {adaptive['full_manifest_macro_f1']:.3f}，相对 always_multi 降低 "
        f"{adaptive['token_reduction_pct']:.1f}% Token，完成率 {adaptive['completion_rate']:.1%}；"
        "使用 OpenAI-compatible provider，模型 ID 由中转服务报告且 identity unverified。\n"
    )


def _update_readme(path: Path, summary: dict[str, object]) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(_RESULTS_START) != 1 or text.count(_RESULTS_END) != 1:
        raise ValueError("README must contain exactly one EvidenceRoute result marker pair")
    start, remainder = text.split(_RESULTS_START, 1)
    _old, end = remainder.split(_RESULTS_END, 1)
    adaptive = summary["strategies"]["adaptive"]
    generated = (
        f"\n- Benchmark: {summary['benchmark_name']}\n"
        f"- Adaptive full-manifest macro-F1: {adaptive['full_manifest_macro_f1']:.3f}\n"
        f"- Adaptive completion rate: {adaptive['completion_rate']:.1%}\n"
        f"- Token reduction vs always_multi: {adaptive['token_reduction_pct']:.1f}%\n"
    )
    _write_text(path, start + _RESULTS_START + generated + _RESULTS_END + end)


def build_report_bundle(
    report_input: ReportInput,
    *,
    publish: bool = False,
) -> ReportBundle:
    if publish:
        # Offline fixtures may relax these switches for diagnostic regeneration.  A publish
        # request is a strict boundary, so callers cannot turn publication into a cache-only or
        # no-Git path by constructing a permissive ReportInput themselves.
        report_input = replace(
            report_input,
            require_git=True,
            allow_official_cache=False,
        )
    activity, plan, state, plan_path, state_path = _load_models(report_input)
    artifacts, artifact_paths = _load_artifacts(Path(report_input.activity_dir), plan, state)
    gold_manifest = load_gold_manifest(
        report_input.gold_manifest,
        allowed_root=Path(report_input.gold_manifest).parent,
    )
    gold = {item.claim_id: item.label for item in gold_manifest.items}
    strategy_payloads: dict[str, dict[str, object]] = {}
    predictions: dict[str, dict[str, str]] = {}
    artifact_values = list(artifacts.values())
    call_nodes = _load_call_nodes(
        _evidence_path(
            report_input.run_store,
            Path(report_input.repository_root) / "artifacts/gate-a-run-store.sqlite3",
        )
    )
    for strategy in Strategy:
        payload, scored = _strategy_summary(strategy, gold, artifact_values, call_nodes=call_nodes)
        strategy_payloads[strategy.value] = payload
        predictions[strategy.value] = scored
    multi = strategy_payloads[Strategy.ALWAYS_MULTI.value]
    adaptive = strategy_payloads[Strategy.ADAPTIVE.value]
    multi_tokens = int(multi["total_tokens"])
    adaptive_tokens = int(adaptive["total_tokens"])
    adaptive["token_reduction_pct"] = (
        100.0 * (multi_tokens - adaptive_tokens) / multi_tokens if multi_tokens else 0.0
    )
    multi_cost = multi["actual_cost_micro_cny"]
    adaptive_cost = adaptive["actual_cost_micro_cny"]
    if isinstance(multi_cost, int) and isinstance(adaptive_cost, int) and multi_cost:
        adaptive["cost_reduction_pct"] = 100.0 * (multi_cost - adaptive_cost) / multi_cost

    gold_values = [gold[claim_id].value for claim_id in gold]
    paired: dict[str, object] = {}
    for fixed in (Strategy.ALWAYS_SINGLE.value, Strategy.ALWAYS_MULTI.value):
        paired[fixed] = paired_bootstrap_difference(
            gold_values,
            [predictions[Strategy.ADAPTIVE.value][claim_id] for claim_id in gold],
            [predictions[fixed][claim_id] for claim_id in gold],
            samples=10_000,
            seed=plan.freeze.seed,
        ).model_dump(mode="json")

    stability_runs: dict[str, list[Any]] = {}
    for link in plan.stability_repeat_zero_links:
        baseline = artifacts.get(link.dev_adaptive_run_id)
        repeats = sorted(
            (
                artifact
                for artifact in artifact_values
                if artifact.phase == "stability" and artifact.claim_id == link.claim_id
            ),
            key=lambda artifact: artifact.repeat,
        )
        if baseline is not None and len(repeats) == 2:
            stability_runs[link.claim_id] = [
                baseline.result,
                repeats[0].result,
                repeats[1].result,
            ]
    stability = (
        stability_consistency(stability_runs).model_dump(mode="json")
        if stability_runs
        else {"total": 0, "consistent": 0, "rate": 0.0, "wilson_low": 0.0, "wilson_high": 0.0}
    )
    gate = _publication_gate(
        report_input,
        activity,
        plan,
        state,
        plan_path,
        state_path,
        artifacts,
    )
    official = _official_summary(
        report_input,
        gold_manifest,
        artifacts,
        strategy_payloads,
        enabled=gate.publishable,
    )
    if gate.publishable and any(
        not isinstance(official.get(name), dict) or official[name].get("available") is not True
        for name in ("shared_task_2024", "paper_2023_secondary")
    ):
        gate = PublicationGateResult(
            publishable=False,
            reasons=[*gate.reasons, "official_evaluator_unavailable"],
        )
    summary: dict[str, object] = {
        "benchmark_name": BENCHMARK_NAME,
        "publishable": gate.publishable,
        "activity_status": activity.status.value,
        "phase_statuses": {
            "calibration": activity.calibration_status.value,
            "dev": activity.dev_status.value,
            "stability": activity.stability_status.value,
        },
        "strategies": strategy_payloads,
        "paired_bootstrap": paired,
        "stability": stability,
        "official": official,
        "reproducibility": {
            "activity_id": activity.activity_id,
            "campaign_id": plan.campaign_id,
            "campaign_fingerprint": plan.campaign_fingerprint,
            "manifest_freeze_git_sha": plan.freeze.manifest_freeze_git_sha,
            "dev_protocol_git_sha": plan.freeze.dev_protocol_git_sha,
            "calibration_runtime_manifest_sha256": plan.freeze.calibration_runtime_manifest_sha256,
            "dev_runtime_manifest_sha256": plan.freeze.dev_runtime_manifest_sha256,
            "stability_runtime_manifest_sha256": plan.freeze.stability_runtime_manifest_sha256,
            "corpus_preparation_receipt_sha256": plan.freeze.corpus_preparation_receipt_sha256,
            "prompt_bundle_sha256": plan.freeze.prompt_bundle_sha256,
            "config_sha256": plan.freeze.config_sha256,
            "pricing_sha256": plan.freeze.pricing_sha256,
            "endpoint_config_sha256": plan.freeze.endpoint_config_sha256,
            "requirements_lock_sha256": plan.freeze.requirements_lock_sha256,
            "seed": plan.freeze.seed,
            "requested_alias": plan.freeze.requested_alias,
            "relay_reported_model_ids_raw": activity.observed_response_model_ids_raw,
            "model_identity": IDENTITY_DESCRIPTION,
        },
        "publication_gate": gate.model_dump(mode="json"),
        "limitations": [
            "Balanced frozen subset; not the full AVeriTeC benchmark or leaderboard.",
            "Relay-reported model identity is self-reported and unverified.",
            "Engineering gates are not statistical non-inferiority claims.",
        ],
    }
    if publish and not gate.publishable:
        raise PublicationBlocked("; ".join(gate.reasons))
    markdown = _render_markdown(summary)
    resume = _resume_snippet(summary) if gate.publishable else None
    sources = _representative_sources(artifacts, artifact_paths)
    return ReportBundle(
        summary=summary,
        markdown=markdown,
        resume_snippet=resume,
        representative_trace_sources=sources,
        publication_gate=gate,
        strict_publication_verified=publish and gate.publishable,
    )


__all__ = [
    "BENCHMARK_NAME",
    "PublicationBlocked",
    "PublicationGateResult",
    "ReportBundle",
    "ReportInput",
    "build_report_bundle",
]
