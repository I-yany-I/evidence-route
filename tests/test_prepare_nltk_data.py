import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts.prepare_nltk_data import (
    extract_checked_zip,
    prepare_from_spec,
    safe_member_path,
    tree_sha256,
)


def _zip_bytes(name: str = "punkt_tab/sample.txt", content: bytes = b"token") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


def test_safe_member_path_rejects_traversal() -> None:
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        safe_member_path("../escape.txt")
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        safe_member_path("/absolute.txt")


def test_extract_checked_zip_and_tree_hash_are_deterministic(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    archive.write_bytes(_zip_bytes())
    output = tmp_path / "out"
    extract_checked_zip(archive, output)
    assert (output / "punkt_tab" / "sample.txt").read_bytes() == b"token"
    first = tree_sha256(output)
    second = tree_sha256(output)
    assert first == second
    assert len(first) == 64


def test_prepare_from_spec_verifies_downloaded_bytes(tmp_path: Path) -> None:
    payload = _zip_bytes("wordnet/sample.txt", b"lemma")
    digest = hashlib.sha256(payload).hexdigest()
    spec = {
        "repository": "example/data",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "files": {
            "packages/corpora/wordnet.zip": {"size": len(payload), "sha256": digest}
        },
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    def fetch(_url: str) -> bytes:
        return payload

    receipt = prepare_from_spec(spec_path, tmp_path / "nltk", fetch=fetch)
    assert receipt["commit"] == spec["commit"]
    assert (tmp_path / "nltk" / "corpora" / "wordnet" / "sample.txt").exists()
