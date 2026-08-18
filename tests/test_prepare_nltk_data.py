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
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        safe_member_path("C:/escape.txt")


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


def test_tree_hash_includes_nested_receipt_named_files(tmp_path: Path) -> None:
    root = tmp_path / "out"
    nested = root / "corpora" / "wordnet"
    nested.mkdir(parents=True)
    receipt = nested / "PREPARATION_RECEIPT.json"
    receipt.write_text("one", encoding="utf-8")
    first = tree_sha256(root)
    receipt.write_text("two", encoding="utf-8")
    assert tree_sha256(root) != first


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


def test_extract_failure_preserves_existing_output(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.zip"
    archive.write_bytes(_zip_bytes("../escape.txt"))
    output = tmp_path / "existing"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_text("keep", encoding="utf-8")

    with pytest.raises((ValueError, FileExistsError)):
        extract_checked_zip(archive, output)
    assert keep.read_text(encoding="utf-8") == "keep"


def test_force_failure_keeps_previous_verified_install(tmp_path: Path) -> None:
    payload = _zip_bytes("wordnet/sample.txt", b"lemma")
    spec = {
        "repository": "example/data",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "files": {
            "packages/corpora/wordnet.zip": {
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "nltk"
    prepare_from_spec(spec_path, output, fetch=lambda _url: payload)

    with pytest.raises(ValueError, match="size mismatch"):
        prepare_from_spec(spec_path, output, fetch=lambda _url: b"bad", force=True)
    assert (output / "corpora" / "wordnet" / "sample.txt").read_bytes() == b"lemma"
    assert (output / "PREPARATION_RECEIPT.json").is_file()


def test_existing_receipt_revalidates_extracted_tree(tmp_path: Path) -> None:
    payload = _zip_bytes("wordnet/sample.txt", b"lemma")
    spec = {
        "repository": "example/data",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "files": {
            "packages/corpora/wordnet.zip": {
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "nltk"
    prepare_from_spec(spec_path, output, fetch=lambda _url: payload)
    (output / "corpora" / "wordnet" / "sample.txt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="integrity"):
        prepare_from_spec(spec_path, output, fetch=lambda _url: payload)


def test_source_spec_rejects_mutable_revision(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "repository": "example/data",
                "commit": "main",
                "files": {
                    "packages/corpora/wordnet.zip": {"size": 1, "sha256": "a" * 64}
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="40-character lowercase"):
        prepare_from_spec(spec_path, tmp_path / "nltk", fetch=lambda _url: b"x")
