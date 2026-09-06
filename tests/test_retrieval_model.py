from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evidence_route.providers.dense import RetrievalModelError, verify_model_receipt


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "model"
    root.mkdir()
    model = root / "model.onnx"
    model.write_bytes(b"model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": 1,
        "model_id": "fixture",
        "model_revision": "a" * 40,
        "files": [{"path": "model.onnx", "sha256": digest}],
    }), encoding="utf-8")
    return root, receipt


def test_model_receipt_accepts_matching_files(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    identity = verify_model_receipt(root, receipt)
    assert identity["model_id"] == "fixture"


def test_model_receipt_rejects_tampered_file(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    (root / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(RetrievalModelError, match="hash mismatch"):
        verify_model_receipt(root, receipt)
