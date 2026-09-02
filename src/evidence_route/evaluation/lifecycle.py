"""Audited freeze and activity-file lifecycle helpers.

The production services use this module at phase boundaries.  It intentionally has no
transport or scorer dependencies: all inputs are paths, hashes, and non-secret endpoint
metadata, which keeps the runtime process independent from gold data and credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from evidence_route.config import stable_hash
from evidence_route.evaluation.activity import (
    ActivityRecord,
    CalibrationArtifactLink,
    CampaignStatus,
    CampaignStopReason,
    FreezeIdentity,
    FreezeMismatch,
    compare_freeze_identity,
    derive_activity_status,
    derive_campaign_status,
)

_SHA256_RE = r"^[0-9a-f]{64}$"
_GIT_RE = r"^[0-9a-f]{40}$"
_SENSITIVE_KEY_PARTS = ("authorization", "api_key", "apikey", "token", "secret", "password")
_ENDPOINT_IDENTITY_KEYS = frozenset({"requested_alias", "model", "model_alias"})


class ActivityFileError(ValueError):
    """Raised when an activity JSON/sidecar pair is absent or inconsistent."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    resolved = Path(path)
    try:
        payload = resolved.read_bytes()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"freeze input is unreadable: {resolved}") from exc
    return sha256_bytes(payload)


def _require_sha(value: str, *, label: str, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{label} must be a lowercase SHA-{length * 4}")
    alphabet = "0123456789abcdef"
    if any(char not in alphabet for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-{length * 4}")
    return value


def _safe_endpoint_value(value: object) -> object:
    """Remove secret-bearing endpoint fields before hashing or serialising."""

    if isinstance(value, Mapping):
        safe: dict[str, object] = {}
        for key, nested in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if lowered in _ENDPOINT_IDENTITY_KEYS:
                # The requested alias is persisted as its own freeze field.  Keeping it
                # out of the endpoint digest prevents two identity fields from drifting
                # independently while preserving the relay URL as the endpoint identity.
                continue
            if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
                continue
            safe[key_text] = _safe_endpoint_value(nested)
        return safe
    if isinstance(value, (list, tuple)):
        return [_safe_endpoint_value(item) for item in value]
    # SecretStr and similar wrappers should never leak their value through repr/str.
    if value.__class__.__name__ in {"SecretStr", "SecretBytes"}:
        return "[REDACTED]"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def endpoint_config_hash(
    endpoint_config: Mapping[str, object] | None = None,
    *,
    base_url: str | None = None,
    requested_alias: str | None = None,
) -> str:
    """Hash only non-secret endpoint identity fields.

    The API key is deliberately dropped rather than replaced.  This makes the digest stable
    across credential rotation while still binding the relay URL and requested model alias.
    """

    payload: dict[str, object] = {}
    if endpoint_config is not None:
        safe = _safe_endpoint_value(endpoint_config)
        if not isinstance(safe, dict):
            raise ValueError("endpoint_config must be a mapping")
        payload.update(safe)
    if base_url is not None:
        payload["base_url"] = base_url
    return stable_hash(payload)


def _manifest_hash(value: Path | str | object, *, label: str) -> str:
    if isinstance(value, (str, Path)):
        if (
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            and not Path(value).is_file()
        ):
            return value
        return sha256_file(Path(value))
    digest = getattr(value, "_manifest_sha256", None)
    if isinstance(digest, str):
        return _require_sha(digest, label=label)
    raise ValueError(f"{label} must be a manifest path or a loaded manifest with a digest")


def _digest_input(
    value: Path | str | None,
    explicit_digest: str | None,
    *,
    label: str,
) -> str:
    """Accept either a file path or a previously audited lowercase digest."""

    if value is None:
        if explicit_digest is None:
            raise ValueError(f"{label} input is required")
        return _require_sha(explicit_digest, label=label)
    if (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
        and not Path(value).is_file()
    ):
        actual = value
    else:
        actual = sha256_file(value)
    if explicit_digest is not None:
        expected = _require_sha(explicit_digest, label=label)
        if actual != expected:
            raise FreezeMismatch(label, expected=expected, actual=actual)
    return actual


def build_freeze_identity(
    *,
    calibration_manifest: Path | str | object,
    dev_manifest: Path | str | object,
    stability_manifest: Path | str | object,
    corpus_receipt: Path | str | None = None,
    prompt_bundle: Path | str | None = None,
    config_file: Path | str | None = None,
    pricing_file: Path | str | None = None,
    requirements_lock: Path | str | None = None,
    corpus_preparation_receipt_sha256: str | None = None,
    prompt_bundle_sha256: str | None = None,
    config_sha256: str | None = None,
    pricing_sha256: str | None = None,
    requirements_lock_sha256: str | None = None,
    endpoint_config: Mapping[str, object] | None = None,
    base_url: str | None = None,
    requested_alias: str,
    seed: int,
    manifest_freeze_git_sha: str,
    dev_protocol_git_sha: str,
) -> FreezeIdentity:
    """Compute every immutable identity field from current local inputs.

    File digests are over raw bytes, including manifest sidecar-independent bytes.  No secret
    value is included in the returned model or in any exception text.
    """

    _require_sha(manifest_freeze_git_sha, label="manifest_freeze_git_sha", length=40)
    _require_sha(dev_protocol_git_sha, label="dev_protocol_git_sha", length=40)
    if not isinstance(requested_alias, str) or not requested_alias.strip():
        raise ValueError("requested_alias must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return FreezeIdentity(
        manifest_freeze_git_sha=manifest_freeze_git_sha,
        dev_protocol_git_sha=dev_protocol_git_sha,
        calibration_runtime_manifest_sha256=_manifest_hash(
            calibration_manifest, label="calibration manifest"
        ),
        dev_runtime_manifest_sha256=_manifest_hash(dev_manifest, label="dev manifest"),
        stability_runtime_manifest_sha256=_manifest_hash(
            stability_manifest, label="stability manifest"
        ),
        corpus_preparation_receipt_sha256=_digest_input(
            corpus_receipt,
            corpus_preparation_receipt_sha256,
            label="corpus_preparation_receipt_sha256",
        ),
        prompt_bundle_sha256=_digest_input(
            prompt_bundle, prompt_bundle_sha256, label="prompt_bundle_sha256"
        ),
        config_sha256=_digest_input(config_file, config_sha256, label="config_sha256"),
        pricing_sha256=_digest_input(pricing_file, pricing_sha256, label="pricing_sha256"),
        # The requested alias is stored as its own freeze field.  Keep the endpoint digest
        # compatible with the provider-smoke digest, which binds only endpoint configuration.
        endpoint_config_sha256=endpoint_config_hash(endpoint_config, base_url=base_url),
        requirements_lock_sha256=_digest_input(
            requirements_lock, requirements_lock_sha256, label="requirements_lock_sha256"
        ),
        requested_alias=requested_alias,
        seed=seed,
    )


def verify_current_freeze(
    expected: FreezeIdentity,
    **kwargs: object,
) -> FreezeIdentity:
    """Recompute and compare the current freeze, returning it on success.

    ``compare_freeze_identity`` walks model declaration order, so the first changed field is
    deterministic and suitable for a resume audit log.
    """

    allow_descendant_git = bool(kwargs.pop("allow_descendant_git", False))
    build_kwargs = dict(kwargs)
    build_kwargs.pop("repository_root", None)
    actual = build_freeze_identity(**build_kwargs)  # type: ignore[arg-type]
    comparable = (
        actual.model_copy(update={"dev_protocol_git_sha": expected.dev_protocol_git_sha})
        if allow_descendant_git
        else actual
    )
    compare_freeze_identity(expected, comparable)
    repository_root = kwargs.get("repository_root")
    if repository_root is not None:
        verify_git_freeze(
            Path(repository_root),
            dev_protocol_git_sha=expected.dev_protocol_git_sha,
            manifest_freeze_git_sha=expected.manifest_freeze_git_sha,
            allow_descendant_git=allow_descendant_git,
        )
    return comparable


def _git(repository_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("git freeze audit failed") from exc
    return completed.stdout.strip()


def git_head(repository_root: Path | str) -> str:
    """Return the exact repository HEAD used by a lifecycle freeze."""

    value = _git(Path(repository_root), "rev-parse", "HEAD")
    _require_sha(value, label="git HEAD", length=40)
    return value


def verify_git_freeze(
    repository_root: Path | str,
    *,
    dev_protocol_git_sha: str,
    manifest_freeze_git_sha: str,
    allow_descendant_git: bool = False,
) -> str:
    """Require a clean tracked worktree at the protocol commit and an ancestor manifest commit."""

    _require_sha(dev_protocol_git_sha, label="dev_protocol_git_sha", length=40)
    _require_sha(manifest_freeze_git_sha, label="manifest_freeze_git_sha", length=40)
    root = Path(repository_root).resolve()
    status = _git(root, "status", "--porcelain")
    if status:
        raise FreezeMismatch("git_worktree_clean", expected=True, actual=False)
    head = git_head(root)
    if head != dev_protocol_git_sha and not allow_descendant_git:
        raise FreezeMismatch("dev_protocol_git_sha", expected=dev_protocol_git_sha, actual=head)
    ancestry_checks = [
        ("manifest_freeze_git_sha", manifest_freeze_git_sha),
        *(
            [("dev_protocol_git_sha", dev_protocol_git_sha)]
            if allow_descendant_git
            else []
        ),
    ]
    for label, ancestor in ancestry_checks:
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", ancestor, head],
                cwd=root,
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise FreezeMismatch(label, expected=ancestor, actual=head) from exc
    return head


def _canonical_activity_bytes(activity: ActivityRecord) -> bytes:
    payload = activity.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def activity_digest(activity: ActivityRecord) -> str:
    return sha256_bytes(_canonical_activity_bytes(activity))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def persist_activity(
    path: Path | str,
    activity: ActivityRecord,
    *,
    fresh: bool = False,
) -> str:
    """Atomically persist ``activity.json`` and its lowercase SHA-256 sidecar."""

    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if fresh and (target.exists() or sidecar.exists()):
        raise FileExistsError(f"activity already exists: {target}")
    payload = _canonical_activity_bytes(activity)
    digest = sha256_bytes(payload)
    _atomic_bytes(target, payload)
    _atomic_bytes(sidecar, (digest + "\n").encode("ascii"))
    return digest


def load_activity(path: Path | str, *, require_sidecar: bool = True) -> ActivityRecord:
    target = Path(path)
    if not target.is_file():
        raise ActivityFileError(f"activity file is missing: {target}")
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise ActivityFileError(f"activity file is unreadable: {target}") from exc
    digest = sha256_bytes(payload)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if require_sidecar:
        if not sidecar.is_file():
            raise ActivityFileError(f"activity sidecar is missing: {sidecar}")
        try:
            recorded = sidecar.read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            raise ActivityFileError(f"activity sidecar is unreadable: {sidecar}") from exc
        if recorded != digest + "\n":
            raise ActivityFileError("sidecar SHA-256 mismatch")
    try:
        activity = ActivityRecord.model_validate_json(payload)
    except Exception as exc:
        raise ActivityFileError(f"activity JSON is invalid: {target}") from exc
    return activity


def update_activity(
    path: Path | str,
    updater: Callable[[ActivityRecord], ActivityRecord],
) -> tuple[ActivityRecord, str]:
    """Load, transform, validate, and atomically replace one activity record."""

    current = load_activity(path)
    updated = updater(current.model_copy(deep=True))
    if not isinstance(updated, ActivityRecord):
        raise TypeError("activity updater must return ActivityRecord")
    digest = persist_activity(path, updated)
    return updated, digest


def build_initial_activity(
    *,
    activity_id: str,
    calibration_plan_sha256: str,
    calibration_state_sha256: str,
    case_ids: Sequence[str],
    freeze: FreezeIdentity | None = None,
) -> ActivityRecord:
    """Create the pre-calibration activity journal with 32 ordered empty links."""

    if len(case_ids) != 32 or len(set(case_ids)) != 32:
        raise ValueError("activity requires 32 unique calibration case IDs")
    for index, case_id in enumerate(case_ids):
        _require_sha(case_id, label=f"case_ids[{index}]")
    _require_sha(calibration_plan_sha256, label="calibration_plan_sha256")
    _require_sha(calibration_state_sha256, label="calibration_state_sha256")
    return ActivityRecord(
        schema_version="1",
        activity_id=activity_id,
        status=CampaignStatus.PLANNED,
        calibration_status=CampaignStatus.PLANNED,
        dev_status=CampaignStatus.PLANNED,
        stability_status=CampaignStatus.PLANNED,
        calibration_plan_sha256=calibration_plan_sha256,
        calibration_state_sha256=calibration_state_sha256,
        calibration_artifacts=[
            CalibrationArtifactLink(case_id=case_id, artifact_sha256=None) for case_id in case_ids
        ],
        freeze=freeze,
        campaign_plan_sha256=None,
        campaign_state_sha256=None,
        observed_response_model_ids_raw=[],
        identity_verified=False,
        billing_uncertain=False,
        stop_reason=None,
        summary=None,
    )


def link_calibration_artifact(
    activity: ActivityRecord,
    *,
    case_id: str,
    artifact_sha256: str,
) -> ActivityRecord:
    """Close exactly one calibration link after its artifact has been atomically written."""

    _require_sha(artifact_sha256, label="artifact_sha256")
    updated = activity.model_copy(deep=True)
    for index, link in enumerate(updated.calibration_artifacts):
        if link.case_id == case_id:
            if any(
                previous.artifact_sha256 is None
                for previous in updated.calibration_artifacts[:index]
            ):
                raise ValueError("calibration artifacts must close in plan order")
            if link.artifact_sha256 is not None and link.artifact_sha256 != artifact_sha256:
                raise FreezeMismatch(
                    "calibration_artifacts", expected=link.artifact_sha256, actual=artifact_sha256
                )
            link.artifact_sha256 = artifact_sha256
            return ActivityRecord.model_validate(updated.model_dump(mode="python"))
    raise ValueError(f"unknown calibration case: {case_id}")


def mark_calibration_complete(activity: ActivityRecord) -> ActivityRecord:
    """Transition calibration only after all 32 links are closed."""

    if any(link.artifact_sha256 is None for link in activity.calibration_artifacts):
        raise ValueError("requires 32 completed calibration artifacts")
    updated = activity.model_copy(update={"calibration_status": CampaignStatus.COMPLETE}, deep=True)
    updated.status = derive_activity_status(
        updated.calibration_status,
        updated.dev_status,
        updated.stability_status,
        stop_reason=updated.stop_reason,
        billing_uncertain=updated.billing_uncertain,
    )
    return ActivityRecord.model_validate(updated.model_dump(mode="python"))


def transition_activity_phase(
    activity: ActivityRecord,
    *,
    phase: str,
    status: CampaignStatus,
    stop_reason: CampaignStopReason | None = None,
    billing_uncertain: bool | None = None,
    observed_response_model_ids_raw: Sequence[str] | None = None,
    summary: object | None = None,
) -> ActivityRecord:
    """Apply one phase status and recompute the publication status atomically in memory."""

    if phase not in {"calibration", "dev", "stability"}:
        raise ValueError("activity phase must be calibration, dev, or stability")
    if not isinstance(status, CampaignStatus):
        status = CampaignStatus(status)
    current = getattr(activity, f"{phase}_status")
    terminal_statuses = {
        CampaignStatus.PAUSED,
        CampaignStatus.INTERRUPTED,
        CampaignStatus.INCOMPLETE_BUDGET,
        CampaignStatus.INCOMPLETE_MODEL_DRIFT,
        CampaignStatus.INCOMPLETE_USAGE,
        CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
        CampaignStatus.CANCELLED,
        CampaignStatus.FAILED,
    }
    if (
        current is CampaignStatus.COMPLETE
        and status is not CampaignStatus.COMPLETE
        and not (status in terminal_statuses and stop_reason is not None)
    ):
        raise ValueError(f"{phase} phase is already terminal")
    if (
        current
        in {
            CampaignStatus.INCOMPLETE_BUDGET,
            CampaignStatus.INCOMPLETE_MODEL_DRIFT,
            CampaignStatus.INCOMPLETE_USAGE,
            CampaignStatus.INCOMPLETE_COST_UNCERTAIN,
            CampaignStatus.CANCELLED,
            CampaignStatus.FAILED,
        }
        and status is not current
    ):
        raise ValueError(f"{phase} phase is already terminal")
    if phase in {"dev", "stability"} and status is not CampaignStatus.PLANNED:
        if activity.calibration_status is not CampaignStatus.COMPLETE:
            raise ValueError("calibration must be complete before dev or stability")
        if phase == "stability" and activity.dev_status is not CampaignStatus.COMPLETE:
            raise ValueError("dev must be complete before stability")
    effective_stop_reason = stop_reason or activity.stop_reason
    if effective_stop_reason is CampaignStopReason.BILLING_UNCERTAIN:
        if billing_uncertain is False:
            raise ValueError("billing stop requires billing_uncertain=True")
        if billing_uncertain is not True:
            raise ValueError("billing stop requires billing_uncertain=True")
    elif billing_uncertain is True:
        raise ValueError("billing_uncertain=True requires a billing stop reason")
    if activity.stop_reason is not None and status in {
        CampaignStatus.PLANNED,
        CampaignStatus.RUNNING,
        CampaignStatus.COMPLETE,
    }:
        raise ValueError("activity has a terminal stop reason")
    if status in terminal_statuses and stop_reason is None:
        raise ValueError("terminal activity status requires a typed stop reason")
    if stop_reason is not None:
        expected_status = derive_campaign_status([], stop_reason=stop_reason)
        if status is not expected_status:
            raise ValueError("activity phase status does not match stop reason")
    updates: dict[str, object] = {f"{phase}_status": status}
    if stop_reason is not None:
        if activity.stop_reason is not None and activity.stop_reason is not stop_reason:
            raise FreezeMismatch(
                "stop_reason", expected=activity.stop_reason.value, actual=stop_reason.value
            )
        updates["stop_reason"] = stop_reason
    if billing_uncertain is not None:
        updates["billing_uncertain"] = billing_uncertain
    if observed_response_model_ids_raw is not None:
        updates["observed_response_model_ids_raw"] = list(
            dict.fromkeys(value for value in observed_response_model_ids_raw if value.strip())
        )
    if summary is not None:
        updates["summary"] = summary
    updated = activity.model_copy(update=updates, deep=True)
    updated.status = derive_activity_status(
        updated.calibration_status,
        updated.dev_status,
        updated.stability_status,
        stop_reason=updated.stop_reason,
        billing_uncertain=updated.billing_uncertain,
    )
    return ActivityRecord.model_validate(updated.model_dump(mode="python"))


def resume_activity_phase(activity: ActivityRecord, *, phase: str) -> ActivityRecord:
    """Atomically clear a resumable process-interruption or user-pause marker."""

    if phase not in {"calibration", "dev", "stability"}:
        raise ValueError("activity phase must be calibration, dev, or stability")
    resumable = {
        CampaignStopReason.PROCESS_INTERRUPTION: CampaignStatus.INTERRUPTED,
        CampaignStopReason.USER_PAUSED: CampaignStatus.PAUSED,
    }
    expected_status = resumable.get(activity.stop_reason)
    if expected_status is None:
        raise ValueError("only process_interruption or user_paused can be resumed")
    if getattr(activity, f"{phase}_status") is not expected_status:
        raise ValueError("phase is not resumable")
    updated = activity.model_copy(
        update={
            f"{phase}_status": CampaignStatus.RUNNING,
            "stop_reason": None,
            "billing_uncertain": False,
        },
        deep=True,
    )
    updated.status = derive_activity_status(
        updated.calibration_status,
        updated.dev_status,
        updated.stability_status,
        billing_uncertain=False,
    )
    return ActivityRecord.model_validate(updated.model_dump(mode="python"))


def resume_activity_after_billing_recovery(
    activity: ActivityRecord, *, phase: str
) -> ActivityRecord:
    """Clear an authorized billing/usage stop after an explicit ledger decision."""

    if phase not in {"calibration", "dev", "stability"}:
        raise ValueError("activity phase must be calibration, dev, or stability")
    if activity.stop_reason is CampaignStopReason.BILLING_UNCERTAIN:
        if not activity.billing_uncertain:
            raise ValueError("billing recovery requires billing_uncertain=True")
        expected_status = CampaignStatus.INCOMPLETE_COST_UNCERTAIN
    elif activity.stop_reason is CampaignStopReason.USAGE_MISSING:
        if activity.billing_uncertain:
            raise ValueError("usage recovery cannot carry billing_uncertain=True")
        expected_status = CampaignStatus.INCOMPLETE_USAGE
    elif activity.stop_reason is CampaignStopReason.INTERNAL_ERROR:
        if activity.billing_uncertain:
            raise ValueError("internal recovery cannot carry billing_uncertain=True")
        expected_status = CampaignStatus.FAILED
    else:
        raise ValueError("activity is not stopped for an authorized recovery")
    if getattr(activity, f"{phase}_status") is not expected_status:
        raise ValueError("phase is not stopped for the expected recovery reason")
    updated = activity.model_copy(
        update={
            f"{phase}_status": CampaignStatus.RUNNING,
            "stop_reason": None,
            "billing_uncertain": False,
        },
        deep=True,
    )
    updated.status = derive_activity_status(
        updated.calibration_status,
        updated.dev_status,
        updated.stability_status,
        billing_uncertain=False,
    )
    return ActivityRecord.model_validate(updated.model_dump(mode="python"))


def seal_activity(
    activity: ActivityRecord,
    *,
    freeze: FreezeIdentity,
    campaign_plan_sha256: str,
    campaign_state_sha256: str,
) -> ActivityRecord:
    """Attach immutable campaign identity after calibration replay has completed."""

    if activity.calibration_status is not CampaignStatus.COMPLETE:
        raise ValueError("cannot seal activity before calibration is complete")
    _require_sha(campaign_plan_sha256, label="campaign_plan_sha256")
    _require_sha(campaign_state_sha256, label="campaign_state_sha256")
    if activity.freeze is not None:
        compare_freeze_identity(activity.freeze, freeze)
    if (
        activity.campaign_plan_sha256 is not None
        and activity.campaign_plan_sha256 != campaign_plan_sha256
    ):
        raise FreezeMismatch(
            "campaign_plan_sha256",
            expected=activity.campaign_plan_sha256,
            actual=campaign_plan_sha256,
        )
    if (
        activity.campaign_state_sha256 is not None
        and activity.campaign_state_sha256 != campaign_state_sha256
    ):
        raise FreezeMismatch(
            "campaign_state_sha256",
            expected=activity.campaign_state_sha256,
            actual=campaign_state_sha256,
        )
    updated = activity.model_copy(
        update={
            "freeze": freeze,
            "campaign_plan_sha256": campaign_plan_sha256,
            "campaign_state_sha256": campaign_state_sha256,
        },
        deep=True,
    )
    return ActivityRecord.model_validate(updated.model_dump(mode="python"))


__all__ = [
    "ActivityFileError",
    "activity_digest",
    "build_freeze_identity",
    "build_initial_activity",
    "endpoint_config_hash",
    "git_head",
    "link_calibration_artifact",
    "load_activity",
    "mark_calibration_complete",
    "persist_activity",
    "resume_activity_after_billing_recovery",
    "resume_activity_phase",
    "seal_activity",
    "sha256_bytes",
    "sha256_file",
    "update_activity",
    "transition_activity_phase",
    "verify_current_freeze",
    "verify_git_freeze",
]
