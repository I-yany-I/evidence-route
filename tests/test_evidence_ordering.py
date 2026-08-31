import hashlib
import json

import pytest

from evidence_route.providers.averitec import AveritecFrozenProvider


def _row(evidence_id: str, source_url: str, text: str = "same evidence") -> str:
    return json.dumps(
        {
            "evidence_id": evidence_id,
            "title": "Source",
            "source_url": source_url,
            "text": text,
            "snapshot_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    )


@pytest.mark.asyncio
async def test_equal_score_candidates_are_stable_before_truncation(tmp_path) -> None:
    first = [
        _row("same-id", "https://example.org/z"),
        _row("same-id", "https://example.org/a"),
    ]
    second = list(reversed(first))
    corpus_a = tmp_path / "a"
    corpus_b = tmp_path / "b"
    corpus_a.mkdir()
    corpus_b.mkdir()
    (corpus_a / "claim.jsonl").write_text("\n".join(first) + "\n", encoding="utf-8")
    (corpus_b / "claim.jsonl").write_text("\n".join(second) + "\n", encoding="utf-8")

    left = await AveritecFrozenProvider(corpus_a).search(
        "claim", "unseen", top_k=1, max_chars=6
    )
    right = await AveritecFrozenProvider(corpus_b).search(
        "claim", "unseen", top_k=1, max_chars=6
    )

    assert [item.evidence_id for item in left] == [item.evidence_id for item in right]
    assert [item.model_dump(mode="json") for item in left] == [
        item.model_dump(mode="json") for item in right
    ]
    assert [str(item.source_url) for item in left] == [
        "https://example.org/a",
    ]


@pytest.mark.asyncio
async def test_equal_score_id_and_url_use_snapshot_hash_as_tie_break(tmp_path) -> None:
    first = [
        _row("same-id", "https://example.org/fact/", "alpha"),
        _row("same-id", "HTTPS://EXAMPLE.ORG:443/fact#quote", "beta"),
    ]
    second = list(reversed(first))
    corpus_a = tmp_path / "a"
    corpus_b = tmp_path / "b"
    corpus_a.mkdir()
    corpus_b.mkdir()
    (corpus_a / "claim.jsonl").write_text("\n".join(first) + "\n", encoding="utf-8")
    (corpus_b / "claim.jsonl").write_text("\n".join(second) + "\n", encoding="utf-8")

    left = await AveritecFrozenProvider(corpus_a).search(
        "claim", "unseen", top_k=1, max_chars=20
    )
    right = await AveritecFrozenProvider(corpus_b).search(
        "claim", "unseen", top_k=1, max_chars=20
    )

    expected_hash = min(
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in ("alpha", "beta")
    )
    assert left[0].snapshot_sha256 == expected_hash
    assert right[0].snapshot_sha256 == expected_hash
    assert left[0].model_dump(mode="json") == right[0].model_dump(mode="json")
