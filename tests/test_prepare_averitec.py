import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_averitec import (
    _parse_jsonl_member,
    build_parser,
    normalize_member,
    prepare_dataset,
    select_balanced_ids,
    select_stability_ids,
    split_runtime_and_gold,
    stream_selected_member,
    verify_lfs_pointer_bytes,
)
from tests.fixtures.averitec.fakes import FakeRemoteZip

LABELS = [
    "Supported", "Refuted", "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


def test_prepare_cli_uses_registered_default_paths() -> None:
    args = build_parser().parse_args([])
    assert args.source_spec == Path("data/sources/averitec.json")
    assert args.output_root == Path("data/processed/averitec")
    assert args.runtime_manifest_root == Path("data/manifests")
    assert args.scorer_manifest_root == Path("data/scorer_manifests")


def test_balanced_ids_are_hash_stable() -> None:
    rows = json.loads(Path("tests/fixtures/averitec/source/train.json").read_text("utf-8"))
    first = select_balanced_ids(rows, split="train", per_label=1, seed=20260817)
    second = select_balanced_ids(rows, split="train", per_label=1, seed=20260817)
    assert first == second
    assert first == [3, 4, 5, 6]
    assert {rows[index]["label"] for index in first} == set(LABELS)


def test_runtime_manifest_has_no_gold_fields() -> None:
    row = {
        "claim": "A claim", "label": "Refuted", "questions": [{"question": "why"}],
        "justification": "gold explanation",
    }
    runtime, gold = split_runtime_and_gold("dev", 7, row)
    assert set(runtime) == {"claim_id", "original_id", "claim", "split"}
    assert not ({"label", "questions", "justification", "gold", "claim_types"} & runtime.keys())
    assert gold["label"] == "Refuted"
    assert gold["claim"] == "A claim"


def test_member_normalization_discards_type_and_query() -> None:
    source = [{
        "claim_id": "7", "type": "gold", "query": "annotator question",
        "url": "https://example.org/a", "url2text": ["Sentence one.", "Sentence two."],
    }]
    records = list(normalize_member("dev", 7, source))
    text = json.dumps(records, ensure_ascii=False).lower()
    assert "annotator question" not in text
    assert '"type"' not in text
    assert set(records[0]) == {
        "evidence_id", "title", "source_url", "text", "snapshot_sha256",
    }
    assert records[0]["evidence_id"] == "av:dev:7:0:0"


def test_jsonl_parser_preserves_unicode_line_separators_inside_text() -> None:
    payload = (
        json.dumps(
            {"url": "https://example.org/a", "url2text": ["before\u2028after"]},
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    rows = _parse_jsonl_member(payload, member_name="7.json")
    assert rows[0]["url2text"] == ["before\u2028after"]


def test_member_guard_rejects_oversized_or_unlisted_entries() -> None:
    fake_archive = FakeRemoteZip()
    fake_archive.add("output_dev/7.json", file_size=268_435_457)
    with pytest.raises(ValueError, match="member exceeds 268435456 bytes"):
        stream_selected_member(
            fake_archive, "output_dev/7.json",
            allowed={"output_dev/7.json"}, max_uncompressed_bytes=268_435_456,
        )
    assert fake_archive.extract_calls == 0
    assert fake_archive.extractall_calls == 0


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/private",
        "http://192.168.1.10/private",
        "http://localhost/private",
        "http://[::1]/private",
    ],
)
def test_member_normalization_rejects_local_urls(url: str) -> None:
    with pytest.raises(ValueError, match="local/private|unsupported"):
        list(normalize_member("dev", 1, [{"url": url, "url2text": ["text"]}]))


def test_member_guard_rejects_duplicate_and_traversal_entries() -> None:
    duplicate = FakeRemoteZip()
    duplicate.add("output_dev/7.json", payload=b'{"url":"https://example.org","url2text":["x"]}\n')
    duplicate.add("output_dev/7.json", payload=b'{"url":"https://example.org","url2text":["y"]}\n')
    with pytest.raises(ValueError, match="duplicate"):
        stream_selected_member(
            duplicate,
            "output_dev/7.json",
            allowed={"output_dev/7.json"},
            max_uncompressed_bytes=100,
        )

    traversal = FakeRemoteZip()
    traversal.add("../output_dev/7.json", payload=b"x")
    with pytest.raises(ValueError, match="unsafe"):
        stream_selected_member(
            traversal,
            "output_dev/7.json",
            allowed={"output_dev/7.json"},
            max_uncompressed_bytes=100,
        )


def test_member_guard_allows_benign_directory_entries() -> None:
    archive = FakeRemoteZip()
    archive.add("output_dev/")
    archive.add("output_dev/7.json", payload=b"x")
    assert (
        stream_selected_member(
            archive,
            "output_dev/7.json",
            allowed={"output_dev/7.json"},
            max_uncompressed_bytes=100,
        )
        == b"x"
    )


def test_stability_selection_uses_selected_dev_ids_only() -> None:
    rows = json.loads(Path("tests/fixtures/averitec/source/dev.json").read_text("utf-8"))
    selected = select_balanced_ids(rows, split="dev", per_label=1, seed=20260817)
    assert select_stability_ids(rows, selected, per_label=1, seed=20260817) == selected


def test_lfs_pointer_identity_is_strict() -> None:
    digest = "a" * 64
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{digest}\nsize 123\n"
    ).encode("ascii")
    verify_lfs_pointer_bytes(pointer, expected_sha256=digest, expected_size=123)
    with pytest.raises(ValueError, match="size"):
        verify_lfs_pointer_bytes(pointer, expected_sha256=digest, expected_size=124)
    with pytest.raises(ValueError, match="version"):
        verify_lfs_pointer_bytes(
            f"oid sha256:{digest}\nsize 123\n".encode("ascii"),
            expected_sha256=digest,
            expected_size=123,
        )


def test_prepare_dataset_writes_aligned_manifests_with_fake_archives(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    train_rows = json.loads(Path("tests/fixtures/averitec/source/train.json").read_text("utf-8"))
    dev_rows = json.loads(Path("tests/fixtures/averitec/source/dev.json").read_text("utf-8"))
    train_path = source_dir / "train.json"
    dev_path = source_dir / "dev.json"
    train_payload = json.dumps(train_rows, separators=(",", ":")).encode("utf-8")
    dev_payload = json.dumps(dev_rows, separators=(",", ":")).encode("utf-8")
    train_path.write_bytes(train_payload)
    dev_path.write_bytes(dev_payload)
    archive_members: dict[str, dict[str, bytes]] = {}
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        for original_id, _row in enumerate(rows):
            archive_path = (
                "data_store/knowledge_store/train/train_0_999.zip"
                if split == "train"
                else "data_store/knowledge_store/dev_knowledge_store.zip"
            )
            member_path = (
                f"{original_id}.json" if split == "train" else f"output_dev/{original_id}.json"
            )
            member_row = {
                "url": "https://example.org/source",
                "type": "gold",
                "query": "must not leak",
                "url2text": [f"Evidence for {split}-{original_id}"],
            }
            archive_members.setdefault(archive_path, {})[member_path] = (
                (json.dumps(member_row, separators=(",", ":")) + "\n").encode("utf-8")
            )
    spec = {
        "dataset": "AVeriTeC",
        "license": "CC BY-NC 4.0",
        "huggingface_repo": "test/repo",
        "huggingface_revision": "a" * 40,
        "metadata": {
            "data/train.json": {
                "sha256": hashlib.sha256(train_payload).hexdigest(),
                "size": len(train_payload),
            },
            "data/dev.json": {
                "sha256": hashlib.sha256(dev_payload).hexdigest(),
                "size": len(dev_payload),
            },
        },
        "knowledge_store": {
            path: {"lfs_oid_sha256": "b" * 64, "size": 1}
            for path in archive_members
        },
        "local_metadata_files": {"train": "train.json", "dev": "dev.json"},
    }
    source_spec_path = source_dir / "averitec.json"
    source_spec_path.write_bytes(json.dumps(spec, indent=2).encode("utf-8"))

    def archive_factory(path: str, timeout: float) -> FakeRemoteZip:
        del timeout
        archive = FakeRemoteZip()
        for member_path, payload in archive_members[path].items():
            archive.add(member_path, payload=payload)
        return archive

    def fetcher(url: str, timeout: float) -> bytes:
        del timeout
        assert not (tmp_path / "manifests").exists()
        assert "/test/repo/raw/" in url
        assert "/datasets/" not in url
        return (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{'b' * 64}\nsize 1\n"
        ).encode("ascii")

    result = prepare_dataset(
        source_spec_path,
        tmp_path / "processed",
        tmp_path / "manifests",
        tmp_path / "scorer_manifests",
        calibration_per_label=1,
        dev_per_label=1,
        stability_per_label=1,
        fetcher=fetcher,
        archive_factory=archive_factory,
    )
    assert result["calibration_count"] == 4
    assert result["dev_count"] == 4
    assert result["stability_count"] == 4
    runtime_path = tmp_path / "manifests" / "averitec_dev_runtime.json"
    gold_path = tmp_path / "scorer_manifests" / "averitec_dev_gold.json"
    runtime_payload = runtime_path.read_bytes()
    runtime = json.loads(runtime_payload)
    gold = json.loads(gold_path.read_text("utf-8"))
    assert len(runtime["items"]) == len(gold["items"]) == 4
    assert all("label" not in item for item in runtime["items"])
    assert gold["runtime_manifest_sha256"] == hashlib.sha256(runtime_payload).hexdigest()
    assert runtime_path.with_suffix(".json.sha256").read_text("ascii").strip() == hashlib.sha256(
        runtime_payload
    ).hexdigest()
