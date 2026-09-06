"""Interfaces for deterministic local dense retrieval encoders."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from threading import RLock
from typing import Protocol

import numpy as np


class DenseEncoder(Protocol):
    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        raise NotImplementedError


class RetrievalModelError(RuntimeError):
    """Raised when a local retrieval model is missing, invalid, or fails."""


def verify_model_receipt(model_root: Path, receipt_path: Path) -> dict[str, object]:
    try:
        receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RetrievalModelError(f"model receipt is unreadable: {receipt_path}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise RetrievalModelError("model receipt schema is invalid")
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise RetrievalModelError("model receipt has no files")
    root = Path(model_root).resolve()
    for entry in files:
        if not isinstance(entry, dict):
            raise RetrievalModelError("model receipt file entry is invalid")
        path = (root / str(entry.get("path", ""))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RetrievalModelError("model receipt path escapes model root") from exc
        if not path.is_file():
            raise RetrievalModelError(f"model file is missing: {path.name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry.get("sha256"):
            raise RetrievalModelError(f"model file hash mismatch: {path.name}")
    return receipt


class FastEmbedEncoder:
    def __init__(self, model: object, *, model_id: str) -> None:
        self._model = model
        self.model_id = model_id
        self._inference_lock = RLock()

    @classmethod
    def from_local(cls, model_root: Path, receipt_path: Path) -> FastEmbedEncoder:
        receipt = verify_model_receipt(model_root, receipt_path)
        try:
            from fastembed import TextEmbedding

            model = TextEmbedding(
                model_name=str(receipt["model_id"]),
                specific_model_path=str(Path(model_root).resolve()),
                local_files_only=True,
                providers=[str(receipt.get("execution_provider", "CPUExecutionProvider"))],
            )
        except Exception as exc:
            raise RetrievalModelError(f"failed to load local retrieval model: {exc}") from exc
        return cls(model, model_id=str(receipt["model_id"]))

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not query.strip():
            raise RetrievalModelError("query must not be empty")
        if not passages:
            return []
        try:
            with self._inference_lock:
                vectors = list(self._model.embed([query, *passages], batch_size=1))
            query_vector = np.asarray(vectors[0], dtype=np.float32)
            return [
                float(np.dot(query_vector, np.asarray(vector, dtype=np.float32)))
                for vector in vectors[1:]
            ]
        except Exception as exc:
            raise RetrievalModelError(f"dense reranking failed: {exc}") from exc


__all__ = [
    "DenseEncoder",
    "FastEmbedEncoder",
    "RetrievalModelError",
    "verify_model_receipt",
]
