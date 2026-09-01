"""Immutable identity and parent checks for isolated stability experiments."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from evidence_route.artifacts import atomic_write_json
from evidence_route.contracts import StrictModel
from evidence_route.evaluation.activity import (
    ActivityRecord,
    CalibrationArtifactLink,
    FreezeMismatch,
    RunArtifact,
    artifact_fingerprint,
)
from evidence_route.evaluation.calibration import (
    CalibrationItemStatus,
    load_calibration_plan,
    load_calibration_state,
)
from evidence_route.evaluation.lifecycle import (
    build_initial_activity,
    load_activity,
    mark_calibration_complete,
    persist_activity,
    sha256_file,
)

EXPERIMENT_METADATA_NAME = "experiment.json"
PUBLISHED_GATE_A_DIRECTORY = "evidence-route-gate-a-20260830-clean1"


class StabilityExperimentIdentity(StrictModel):
    experiment_id: str = Field(min_length=1)
    activity_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    parent_activity_id: str = Field(min_length=1)
    parent_campaign_id: str | None = Field(default=None, min_length=1)
    parent_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_pricing_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stability_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repeat_schedule: dict[str, list[int]]
    repeat_zero_artifact_sha256s: dict[str, str] = Field(default_factory=dict)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_alias: str = Field(min_length=1)
    response_model_id: str = Field(min_length=1)
    identity_verified: bool
    created_at: str = Field(min_length=1)

    @field_validator("repeat_schedule")
    @classmethod
    def validate_repeat_schedule(cls, value: dict[str, list[int]]) -> dict[str, list[int]]:
        if not value or any(not claim_id for claim_id in value):
            raise ValueError("repeat_schedule must contain claim IDs")
        for claim_id, repeats in value.items():
            if repeats != sorted(set(repeats)) or any(
                repeat not in (0, 1, 2) for repeat in repeats
            ):
                raise ValueError(f"repeat schedule is invalid for {claim_id!r}")
        return value

    @field_validator("repeat_zero_artifact_sha256s")
    @classmethod
    def validate_artifact_links(cls, value: dict[str, str]) -> dict[str, str]:
        for claim_id, digest in value.items():
            if not claim_id or len(digest) != 64 or digest != digest.lower():
                raise ValueError("repeat-zero artifact links must use lowercase SHA-256 values")
            int(digest, 16)
        return value

    @model_validator(mode="after")
    def validate_link_schedule(self) -> StabilityExperimentIdentity:
        scheduled_claims = set(self.repeat_schedule)
        if set(self.repeat_zero_artifact_sha256s) - scheduled_claims:
            raise ValueError("repeat-zero artifact links contain an unscheduled claim")
        if any(
            0 not in self.repeat_schedule[claim_id]
            for claim_id in self.repeat_zero_artifact_sha256s
        ):
            raise ValueError("repeat-zero artifact links require repeat 0 in the schedule")
        return self


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"experiment parent report is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("experiment parent report must contain a JSON object")
    return value


def _find_value(value: object, key: str) -> object | None:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for nested in value.values():
            found = _find_value(nested, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_value(nested, key)
            if found is not None:
                return found
    return None


def _parent_identity(report: Path) -> tuple[str | None, str | None, dict[str, str]]:
    payload = _read_json(report)
    activity_id = _find_value(payload, "activity_id")
    campaign_id = _find_value(payload, "campaign_id")
    links = _find_value(payload, "stability_repeat_zero_artifact_sha256s")
    normalized_links: dict[str, str] = {}
    if isinstance(links, Mapping):
        normalized_links = {str(key): str(value) for key, value in links.items()}
    return (
        activity_id if isinstance(activity_id, str) else None,
        campaign_id if isinstance(campaign_id, str) else None,
        normalized_links,
    )


def _parent_prompt_hash(report: Path, fallback_prompt: Path) -> str:
    payload = _read_json(report)
    value = _find_value(payload, "prompt_bundle_sha256")
    if value is None:
        return sha256_file(fallback_prompt)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("parent report prompt hash is invalid")
    return value


def _reject_published_target(parent_report: Path, output_dir: Path) -> None:
    report_dir = parent_report.resolve().parent
    target = output_dir.resolve()
    if PUBLISHED_GATE_A_DIRECTORY in {part.name for part in target.parents} | {target.name}:
        raise ValueError("experiment output cannot be inside the published Gate A directory")
    if report_dir.name == PUBLISHED_GATE_A_DIRECTORY:
        try:
            target.relative_to(report_dir)
        except ValueError:
            return
        raise ValueError("experiment output cannot be inside the published Gate A directory")
    if target == report_dir:
        raise ValueError("experiment output must be distinct from the parent report directory")


def _validate_file_hash(path: Path, expected: str, field: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise FreezeMismatch(field, expected=expected, actual=actual)


def create_experiment_identity(
    *,
    experiment_id: str,
    activity_id: str,
    campaign_id: str,
    parent_activity_id: str,
    parent_report: Path,
    parent_manifest: Path,
    pricing: Path,
    config: Path,
    parent_config: Path | None = None,
    prompt: Path,
    stability_manifest: Path,
    repeat_schedule: Mapping[str, list[int]],
    requested_alias: str,
    response_model_id: str,
    identity_verified: bool,
    output_dir: Path,
    repeat_zero_artifact_sha256s: Mapping[str, str] | None = None,
) -> StabilityExperimentIdentity:
    parent_report = Path(parent_report)
    parent_manifest = Path(parent_manifest)
    pricing = Path(pricing)
    parent_config = Path(parent_config) if parent_config is not None else Path(config)
    config = Path(config)
    prompt = Path(prompt)
    stability_manifest = Path(stability_manifest)
    output_dir = Path(output_dir)
    for path, label in (
        (parent_report, "parent report"),
        (parent_manifest, "parent manifest"),
        (pricing, "pricing"),
        (parent_config, "parent config"),
        (config, "config"),
        (prompt, "prompt"),
        (stability_manifest, "stability manifest"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} input is missing: {path}")
    _reject_published_target(parent_report, output_dir)
    report_activity_id, parent_campaign_id, report_links = _parent_identity(parent_report)
    if report_activity_id is not None and report_activity_id != parent_activity_id:
        raise FreezeMismatch(
            "parent_activity_id", expected=parent_activity_id, actual=report_activity_id
        )
    links = dict(repeat_zero_artifact_sha256s or report_links)
    parent_prompt_hash = _parent_prompt_hash(parent_report, prompt)
    identity = StabilityExperimentIdentity(
        experiment_id=experiment_id,
        activity_id=activity_id,
        campaign_id=campaign_id,
        parent_activity_id=parent_activity_id,
        parent_campaign_id=parent_campaign_id,
        parent_report_sha256=sha256_file(parent_report),
        parent_manifest_sha256=sha256_file(parent_manifest),
        parent_pricing_sha256=sha256_file(pricing),
        parent_config_sha256=sha256_file(parent_config),
        parent_prompt_sha256=parent_prompt_hash,
        stability_manifest_sha256=sha256_file(stability_manifest),
        repeat_schedule={str(key): list(value) for key, value in repeat_schedule.items()},
        repeat_zero_artifact_sha256s=links,
        config_sha256=sha256_file(config),
        prompt_sha256=sha256_file(prompt),
        requested_alias=requested_alias,
        response_model_id=response_model_id,
        identity_verified=identity_verified,
        created_at=datetime.now(UTC).isoformat(),
    )
    metadata_path = output_dir / EXPERIMENT_METADATA_NAME
    if metadata_path.exists():
        existing = load_experiment_identity(output_dir)
        if existing.model_dump(mode="json", exclude={"created_at"}) != identity.model_dump(
            mode="json", exclude={"created_at"}
        ):
            raise FileExistsError(
                f"experiment identity already exists with different frozen inputs: {metadata_path}"
            )
        return existing
    atomic_write_json(metadata_path, identity.model_dump(mode="json"))
    return identity


def load_experiment_identity(experiment_dir: Path) -> StabilityExperimentIdentity:
    path = Path(experiment_dir) / EXPERIMENT_METADATA_NAME
    try:
        return StabilityExperimentIdentity.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"experiment identity is missing or invalid: {path}") from exc


def validate_experiment_target(parent_report: Path, experiment_dir: Path) -> None:
    """Validate an experiment destination without creating files in it."""

    _reject_published_target(Path(parent_report), Path(experiment_dir))


def validate_parent_baseline(
    identity: StabilityExperimentIdentity,
    *,
    parent_report: Path,
    parent_manifest: Path,
    pricing: Path,
    config: Path,
    parent_config: Path | None = None,
    prompt: Path,
    stability_manifest: Path | None = None,
) -> None:
    parent_config = Path(parent_config) if parent_config is not None else Path(config)
    for path, expected, field in (
        (Path(parent_report), identity.parent_report_sha256, "parent_report_sha256"),
        (Path(parent_manifest), identity.parent_manifest_sha256, "parent_manifest_sha256"),
        (Path(pricing), identity.parent_pricing_sha256, "parent_pricing_sha256"),
        (parent_config, identity.parent_config_sha256, "parent_config_sha256"),
    ):
        _validate_file_hash(path, expected, field)
    reported_parent_prompt = _parent_prompt_hash(Path(parent_report), Path(prompt))
    if reported_parent_prompt != identity.parent_prompt_sha256:
        raise FreezeMismatch(
            "parent_prompt_sha256",
            expected=identity.parent_prompt_sha256,
            actual=reported_parent_prompt,
        )
    _validate_file_hash(Path(prompt), identity.prompt_sha256, "prompt_sha256")
    _validate_file_hash(Path(config), identity.config_sha256, "config_sha256")
    if stability_manifest is not None:
        _validate_file_hash(
            Path(stability_manifest),
            identity.stability_manifest_sha256,
            "stability_manifest_sha256",
        )
    report_activity_id, parent_campaign_id, _ = _parent_identity(Path(parent_report))
    if report_activity_id != identity.parent_activity_id:
        raise FreezeMismatch(
            "parent_activity_id", expected=identity.parent_activity_id, actual=report_activity_id
        )
    if (
        identity.parent_campaign_id is not None
        and parent_campaign_id != identity.parent_campaign_id
    ):
        raise FreezeMismatch(
            "parent_campaign_id", expected=identity.parent_campaign_id, actual=parent_campaign_id
        )


def verify_repeat_zero_reuse(
    identity: StabilityExperimentIdentity,
    baseline_artifacts: Mapping[str, Path],
) -> None:
    expected_links = identity.repeat_zero_artifact_sha256s
    if set(baseline_artifacts) != set(expected_links):
        raise FreezeMismatch(
            "repeat_zero_artifact_links",
            expected=sorted(expected_links),
            actual=sorted(baseline_artifacts),
        )
    for claim_id, expected in expected_links.items():
        path = Path(baseline_artifacts[claim_id])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            actual = payload.get("artifact_sha256") if isinstance(payload, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FreezeMismatch(
                "repeat_zero_artifact_sha256", expected=expected, actual=None
            ) from exc
        if actual != expected:
            raise FreezeMismatch("repeat_zero_artifact_sha256", expected=expected, actual=actual)
        try:
            artifact = RunArtifact.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"repeat-zero artifact is invalid: {path}") from exc
        if artifact_fingerprint(artifact) != expected:
            raise FreezeMismatch(
                "repeat_zero_artifact_fingerprint",
                expected=expected,
                actual=artifact_fingerprint(artifact),
            )


def materialize_calibration_assets(
    *,
    parent_activity_dir: Path,
    experiment_activity_dir: Path,
    experiment_activity_id: str,
    parent_activity_id: str,
) -> ActivityRecord:
    """Copy a complete calibration journal into an isolated experiment activity.

    The calibration plan/state and case artifacts retain their original bytes and identity. The
    generated activity journal receives the new experiment activity ID, so the campaign can use
    its own artifact directory while the parent directory remains read-only.
    """

    parent = Path(parent_activity_dir).resolve()
    target = Path(experiment_activity_dir).resolve()
    if parent == target:
        raise ValueError("experiment activity directory must be distinct from parent activity")
    source_activity = load_activity(parent / "activity.json")
    if source_activity.activity_id != parent_activity_id:
        raise FreezeMismatch(
            "parent_activity_id", expected=parent_activity_id, actual=source_activity.activity_id
        )
    if source_activity.calibration_status.value != "complete":
        raise ValueError("parent calibration activity is not complete")

    names = ("calibration-plan.json", "calibration-state.json", "calibration-replay.json")
    for name in names:
        source = parent / name
        if not source.is_file():
            raise ValueError(f"parent calibration asset is missing: {source}")
    source_plan = load_calibration_plan(parent / "calibration-plan.json")
    source_state = load_calibration_state(
        parent / "calibration-state.json", expected_plan=source_plan
    )
    if any(item.status is not CalibrationItemStatus.COMPLETE for item in source_state.items):
        raise ValueError("parent calibration state is incomplete")
    source_cases = parent / "calibration"
    if not source_cases.is_dir():
        raise ValueError(f"parent calibration cases are missing: {source_cases}")

    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = parent / name
        destination = target / name
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise FreezeMismatch(
                    f"experiment_{name}",
                    expected=sha256_file(source),
                    actual=sha256_file(destination),
                )
        else:
            shutil.copy2(source, destination)
    destination_cases = target / "calibration"
    if destination_cases.exists():
        if not destination_cases.is_dir():
            raise ValueError(f"experiment calibration path is not a directory: {destination_cases}")
        source_files = sorted(path.relative_to(source_cases) for path in source_cases.rglob("*"))
        target_files = sorted(
            path.relative_to(destination_cases) for path in destination_cases.rglob("*")
        )
        if source_files != target_files:
            raise FreezeMismatch(
                "experiment_calibration_cases", expected=source_files, actual=target_files
            )
        for relative in source_files:
            source = source_cases / relative
            destination = destination_cases / relative
            if source.is_file() and source.read_bytes() != destination.read_bytes():
                raise FreezeMismatch(
                    "experiment_calibration_case",
                    expected=sha256_file(source),
                    actual=sha256_file(destination),
                )
    else:
        shutil.copytree(source_cases, destination_cases)

    activity_path = target / "activity.json"
    if activity_path.exists():
        activity = load_activity(activity_path)
        if activity.activity_id != experiment_activity_id:
            raise FreezeMismatch(
                "experiment_activity_id",
                expected=experiment_activity_id,
                actual=activity.activity_id,
            )
        if activity.calibration_status.value != "complete":
            raise ValueError("experiment calibration activity is not complete")
        return activity

    activity = build_initial_activity(
        activity_id=experiment_activity_id,
        calibration_plan_sha256=sha256_file(target / "calibration-plan.json"),
        calibration_state_sha256=sha256_file(target / "calibration-state.json"),
        case_ids=[item.case_id for item in source_plan.items],
    )
    activity = activity.model_copy(
        update={
            "calibration_artifacts": [
                CalibrationArtifactLink(
                    case_id=item.case_id,
                    artifact_sha256=item.artifact_sha256,
                )
                for item in source_state.items
            ],
            "observed_response_model_ids_raw": source_activity.observed_response_model_ids_raw,
        },
        deep=True,
    )
    activity = mark_calibration_complete(activity)
    persist_activity(activity_path, activity, fresh=True)
    return activity


__all__ = [
    "EXPERIMENT_METADATA_NAME",
    "StabilityExperimentIdentity",
    "create_experiment_identity",
    "load_experiment_identity",
    "materialize_calibration_assets",
    "validate_experiment_target",
    "validate_parent_baseline",
    "verify_repeat_zero_reuse",
]
