import hashlib
import json
from pathlib import Path

import pytest

from evidence_route.evaluation.stability import canonicalize_citation_url
from evidence_route.providers.averitec import AveritecFrozenProvider


@pytest.mark.asyncio
async def test_bm25_returns_stable_ids_and_truncates() -> None:
    provider = AveritecFrozenProvider(Path("tests/fixtures/averitec/corpora"))
    results = await provider.search(
        "dev-0", "Was the Connery letter authentic?", top_k=2, max_chars=30
    )
    assert results[0].evidence_id == "av:dev:0:0:0"
    assert len(results[0].text) <= 30
    assert results[0].snapshot_sha256 != "0" * 64


@pytest.mark.asyncio
async def test_missing_claim_corpus_is_an_error() -> None:
    provider = AveritecFrozenProvider(Path("tests/fixtures/averitec/corpora"))
    with pytest.raises(FileNotFoundError):
        await provider.search("dev-999", "query", top_k=2, max_chars=30)


@pytest.mark.asyncio
async def test_repeated_search_reuses_one_claim_index() -> None:
    provider = AveritecFrozenProvider(
        Path("tests/fixtures/averitec/corpora"), max_cached_claims=1
    )
    await provider.search("dev-0", "letter", top_k=2, max_chars=30)
    await provider.search("dev-0", "imaginary", top_k=2, max_chars=30)
    assert provider.index_build_count == 1


@pytest.mark.asyncio
async def test_search_indexes_source_title_for_queries_missing_from_body(tmp_path: Path) -> None:
    corpus = tmp_path / "claim-title.jsonl"
    rows = [
        {
            "evidence_id": "e-title",
            "title": "Fiat loan report",
            "source_url": "https://example.org/title",
            "text": "The article body omits the key phrase.",
        },
        {
            "evidence_id": "e-other",
            "title": "Other report",
            "source_url": "https://example.org/other",
            "text": "Unrelated text only.",
        },
        {
            "evidence_id": "e-third",
            "title": "Third report",
            "source_url": "https://example.org/third",
            "text": "More unrelated text only.",
        },
    ]
    with corpus.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            row["snapshot_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
            handle.write(json.dumps(row) + "\n")

    results = await AveritecFrozenProvider(tmp_path).search(
        "claim-title", "Fiat loan", top_k=1, max_chars=100
    )

    assert [item.evidence_id for item in results] == ["e-title"]


@pytest.mark.asyncio
async def test_search_can_bound_candidates_per_canonical_source(tmp_path: Path) -> None:
    corpus = tmp_path / "claim-sources.jsonl"
    rows = [
        {
            "evidence_id": "e-one-a",
            "title": "Source one",
            "source_url": "https://example.org/fact/",
            "text": "alpha beta gamma",
        },
        {
            "evidence_id": "e-one-b",
            "title": "Source one mirror",
            "source_url": "HTTPS://EXAMPLE.ORG:443/fact#section",
            "text": "alpha beta delta",
        },
        {
            "evidence_id": "e-two",
            "title": "Source two",
            "source_url": "https://other.example/fact",
            "text": "alpha beta epsilon",
        },
    ]
    with corpus.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            row["snapshot_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
            handle.write(json.dumps(row) + "\n")

    results = await AveritecFrozenProvider(tmp_path).search(
        "claim-sources", "alpha beta", top_k=3, max_chars=100, max_per_source=1
    )

    assert len(results) == 2
    assert {item.evidence_id for item in results} <= {"e-one-a", "e-one-b", "e-two"}
    assert "e-two" in {item.evidence_id for item in results}
    assert len({canonicalize_citation_url(str(item.source_url)) for item in results}) == 2
