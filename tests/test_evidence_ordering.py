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
        "claim", "unseen", top_k=2, max_chars=6
    )
    right = await AveritecFrozenProvider(corpus_b).search(
        "claim", "unseen", top_k=2, max_chars=6
    )

    assert [item.evidence_id for item in left] == [item.evidence_id for item in right]
    assert [item.model_dump(mode="json") for item in left] == [
        item.model_dump(mode="json") for item in right
    ]
    assert [str(item.source_url) for item in left] == [
        "https://example.org/a",
        "https://example.org/z",
    ]
