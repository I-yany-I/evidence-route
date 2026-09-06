"""Versioned construction of frozen-corpus evidence providers."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evidence_route.config import EvidenceSettings
from evidence_route.providers.averitec import AveritecFrozenProvider
from evidence_route.providers.averitec_v2 import AveritecHybridProvider
from evidence_route.providers.dense import (
    FastEmbedEncoder,
    RetrievalModelError,
    verify_model_receipt,
)

EncoderFactory = Callable[[Path, Path], Any]


def _configured_model_paths(
    model_root: Path | None,
    model_receipt: Path | None,
) -> tuple[Path, Path]:
    root_value = model_root or os.environ.get("EVIDENCE_ROUTE_RETRIEVAL_MODEL_ROOT")
    receipt_value = model_receipt or os.environ.get("EVIDENCE_ROUTE_RETRIEVAL_MODEL_RECEIPT")
    if root_value is None or receipt_value is None:
        raise RetrievalModelError("hybrid retrieval requires model root and receipt paths")
    return Path(root_value), Path(receipt_value)


def configured_model_receipt_path(
    settings: EvidenceSettings,
    *,
    model_root: Path | None = None,
    model_receipt: Path | None = None,
) -> Path | None:
    if settings.retrieval_mode == "sentence_bm25_v1":
        return None
    _, receipt_path = _configured_model_paths(model_root, model_receipt)
    return receipt_path


def build_evidence_provider(
    corpus_dir: Path,
    settings: EvidenceSettings,
    *,
    model_root: Path | None = None,
    model_receipt: Path | None = None,
    encoder_factory: EncoderFactory = FastEmbedEncoder.from_local,
) -> AveritecFrozenProvider | AveritecHybridProvider:
    """Build only the provider named by validated retrieval settings."""

    if settings.retrieval_mode == "sentence_bm25_v1":
        return AveritecFrozenProvider(corpus_dir)

    root, receipt_path = _configured_model_paths(model_root, model_receipt)
    try:
        actual_receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RetrievalModelError(f"model receipt is unreadable: {receipt_path}") from exc
    if actual_receipt_hash != settings.dense_model_receipt_sha256:
        raise RetrievalModelError("model receipt hash mismatch")

    receipt = verify_model_receipt(root, receipt_path)
    if (
        receipt.get("model_id") != settings.dense_model_id
        or receipt.get("model_revision") != settings.dense_model_revision
    ):
        raise RetrievalModelError("model receipt identity differs from retrieval settings")
    encoder = encoder_factory(root, receipt_path)
    if getattr(encoder, "model_id", None) != settings.dense_model_id:
        raise RetrievalModelError("loaded encoder identity differs from retrieval settings")
    return AveritecHybridProvider(
        corpus_dir,
        settings=settings.model_dump(mode="python"),
        encoder=encoder,
    )


__all__ = ["build_evidence_provider", "configured_model_receipt_path"]
