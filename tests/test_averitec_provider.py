from pathlib import Path

import pytest

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
