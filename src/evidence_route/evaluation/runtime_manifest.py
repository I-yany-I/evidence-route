"""Claim-only runtime manifest loading and integrity checks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from evidence_route.contracts import StrictModel

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = frozenset({"label", "questions", "justification", "gold", "claim_types"})


class RuntimeClaim(StrictModel):
    claim_id: str = Field(min_length=1)
    original_id: int = Field(ge=0)
    claim: str = Field(min_length=1)
    claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["train", "dev"]
    corpus_relpath: str = Field(pattern=r"^(train|dev)-\d+\.jsonl$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_bytes: int = Field(gt=0)
    corpus_records: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_identity(self) -> RuntimeClaim:
        if self.claim_id != f"{self.split}-{self.original_id}":
            raise ValueError("claim_id does not match split and original_id")
        if self.corpus_relpath != self.claim_id + ".jsonl":
            raise ValueError("corpus_relpath does not match claim_id")
        actual = hashlib.sha256(self.claim.encode("utf-8")).hexdigest()
        if self.claim_sha256 != actual:
            raise ValueError("claim SHA-256 mismatch")
        return self


class RuntimeManifest(StrictModel):
    _manifest_sha256: str | None = PrivateAttr(default=None)
    schema_version: Literal["1"]
    dataset: Literal["AVeriTeC"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    items: list[RuntimeClaim] = Field(min_length=1)


def manifest_digest(path: Path) -> str:
    """Return the digest of the raw manifest bytes, excluding its sidecar."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve_inside(path: Path, allowed_root: Path, *, label: str) -> Path:
    root = Path(allowed_root).resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} outside allowed runtime root: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return resolved


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        text = sidecar.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"manifest sidecar is unreadable: {sidecar}") from exc
    if not re.fullmatch(r"[0-9a-f]{64}\n", text):
        raise ValueError("manifest sidecar must contain one lowercase SHA-256 plus newline")
    actual = manifest_digest(path)
    if text[:-1] != actual:
        raise ValueError("sidecar SHA-256 mismatch")
    return actual


def _reject_forbidden(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in _FORBIDDEN_KEYS:
                raise ValueError(f"runtime manifest contains forbidden field: {key}")
            _reject_forbidden(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_forbidden(nested)


def _reject_duplicates(manifest: RuntimeManifest) -> None:
    ids = [item.claim_id for item in manifest.items]
    if len(ids) != len(set(ids)):
        raise ValueError("runtime manifest contains duplicate claim IDs")


def _verify_corpora(manifest: RuntimeManifest, corpus_root: Path) -> None:
    root = Path(corpus_root).resolve()
    for item in manifest.items:
        path = (root / item.corpus_relpath).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"corpus path outside allowed corpus root: {item.corpus_relpath}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"corpus file is missing: {item.corpus_relpath}")
        payload = path.read_bytes()
        if len(payload) != item.corpus_bytes:
            raise ValueError(f"corpus byte count mismatch: {item.claim_id}")
        if hashlib.sha256(payload).hexdigest() != item.corpus_sha256:
            raise ValueError(f"corpus SHA-256 mismatch: {item.claim_id}")
        records = sum(1 for line in payload.splitlines() if line.strip())
        if records != item.corpus_records:
            raise ValueError(f"corpus record count mismatch: {item.claim_id}")


def load_runtime_manifest(
    path: Path | str, *, allowed_root: Path | str, corpus_root: Path | str | None = None
) -> RuntimeManifest:
    """Load a claim-only manifest after path, sidecar, schema and corpus checks."""

    resolved = _resolve_inside(Path(path), Path(allowed_root), label="runtime manifest")
    digest = _verify_sidecar(resolved)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"runtime manifest is not valid UTF-8 JSON: {resolved}") from exc
    _reject_forbidden(raw)
    try:
        manifest = RuntimeManifest.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"invalid runtime manifest: {exc}") from exc
    _reject_duplicates(manifest)
    manifest._manifest_sha256 = digest
    if corpus_root is not None:
        _verify_corpora(manifest, Path(corpus_root))
    return manifest
