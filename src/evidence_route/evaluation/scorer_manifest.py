"""Gold/scorer manifest loading kept outside the runtime import boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from evidence_route.contracts import StrictModel, Verdict
from evidence_route.evaluation.runtime_manifest import (
    RuntimeClaim,
    RuntimeManifest,
    _resolve_inside,
    _verify_sidecar,
)


class GoldClaim(StrictModel):
    claim_id: str = Field(min_length=1)
    original_id: int = Field(ge=0)
    claim: str = Field(min_length=1)
    label: Verdict
    questions: list[dict[str, object]]
    justification: str
    claim_types: list[str] = Field(default_factory=list)


class GoldManifest(StrictModel):
    schema_version: str = Field(pattern=r"^1$")
    dataset: str = Field(pattern=r"^AVeriTeC$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    items: list[GoldClaim] = Field(min_length=1)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"gold manifest is not valid UTF-8 JSON: {path}") from exc


def load_gold_manifest(path: Path | str, *, allowed_root: Path | str) -> GoldManifest:
    resolved = _resolve_inside(Path(path), Path(allowed_root), label="scorer manifest")
    _verify_sidecar(resolved)
    try:
        manifest = GoldManifest.model_validate(_read_json(resolved))
    except Exception as exc:
        raise ValueError(f"invalid gold manifest: {exc}") from exc
    ids = [item.claim_id for item in manifest.items]
    if len(ids) != len(set(ids)):
        raise ValueError("gold manifest contains duplicate claim IDs")
    return manifest


def align_runtime_and_gold(
    runtime: RuntimeManifest,
    gold: GoldManifest,
    *,
    runtime_manifest_sha256: str | None = None,
) -> list[tuple[RuntimeClaim, GoldClaim]]:
    """Cryptographically align runtime claims with scorer-only gold rows."""

    runtime_manifest_sha256 = runtime_manifest_sha256 or runtime._manifest_sha256
    if runtime_manifest_sha256 is None:
        raise ValueError("runtime manifest digest is required for gold alignment")
    if runtime_manifest_sha256 != gold.runtime_manifest_sha256:
        raise ValueError("gold manifest is bound to a different runtime manifest")
    if (runtime.revision, runtime.seed, runtime.source_metadata_sha256) != (
        gold.revision,
        gold.seed,
        gold.source_metadata_sha256,
    ):
        raise ValueError("runtime and gold manifest identities differ")
    if len(runtime.items) != len(gold.items):
        raise ValueError("runtime and gold manifest lengths differ")
    aligned: list[tuple[RuntimeClaim, GoldClaim]] = []
    for runtime_item, gold_item in zip(runtime.items, gold.items, strict=True):
        if (
            runtime_item.claim_id,
            runtime_item.original_id,
            runtime_item.claim,
        ) != (gold_item.claim_id, gold_item.original_id, gold_item.claim):
            raise ValueError("runtime and gold claim order or text differs")
        aligned.append((runtime_item, gold_item))
    return aligned


__all__ = ["GoldClaim", "GoldManifest", "align_runtime_and_gold", "load_gold_manifest"]
