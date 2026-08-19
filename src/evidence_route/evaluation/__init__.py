"""Offline scoring and evaluation boundaries for EvidenceRoute."""

from evidence_route.evaluation.lifecycle import (
    ActivityFileError,
    activity_digest,
    build_freeze_identity,
    build_initial_activity,
    endpoint_config_hash,
    link_calibration_artifact,
    load_activity,
    mark_calibration_complete,
    persist_activity,
    seal_activity,
    transition_activity_phase,
    update_activity,
    verify_current_freeze,
    verify_git_freeze,
)
from evidence_route.evaluation.metrics import (
    NO_PREDICTION,
    paired_bootstrap_difference,
    score_full_manifest,
    wilson_interval,
)
from evidence_route.evaluation.runtime_manifest import (
    RuntimeClaim,
    RuntimeManifest,
    load_runtime_manifest,
)

__all__ = [
    "NO_PREDICTION",
    "ActivityFileError",
    "RuntimeClaim",
    "RuntimeManifest",
    "activity_digest",
    "build_freeze_identity",
    "build_initial_activity",
    "endpoint_config_hash",
    "link_calibration_artifact",
    "load_activity",
    "load_runtime_manifest",
    "mark_calibration_complete",
    "persist_activity",
    "paired_bootstrap_difference",
    "seal_activity",
    "score_full_manifest",
    "transition_activity_phase",
    "update_activity",
    "verify_current_freeze",
    "verify_git_freeze",
    "wilson_interval",
]
