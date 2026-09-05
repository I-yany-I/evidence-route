from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evidence_route.providers.averitec import AveritecFrozenProvider
from evidence_route.providers.averitec_v2 import (
    AveritecHybridProvider,
    RetrievalModelError,
    load_frozen_index,
    source_candidates,
)


def _write_corpus(root: Path) -> None:
    rows = [
        ("a-1", "Source A", "https://a.test/", "claim claim claim common"),
        ("a-2", "Source A", "https://a.test/", "claim claim common"),
        ("a-3", "Source A", "https://a.test/", "common filler"),
        ("b-1", "Source B", "https://b.test/", "decisive phrase"),
    ]
    path = root / "claim-1.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for evidence_id, title, source_url, text in rows:
            handle.write(json.dumps({
                "evidence_id": evidence_id,
                "title": title,
                "source_url": source_url,
                "text": text,
                "snapshot_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }) + "\n")


@pytest.mark.asyncio
async def test_v1_provider_remains_compatible(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    result = await AveritecFrozenProvider(tmp_path).search(
        "claim-1", "claim common", top_k=2, max_chars=80
    )
    assert [item.evidence_id for item in result] == ["b-1", "a-1"]


def test_source_candidates_are_diverse_and_deterministic(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    index = load_frozen_index(tmp_path / "claim-1.jsonl")

    first = source_candidates(
        index,
        "claim decisive phrase",
        source_candidate_k=2,
        passages_per_source=2,
        dense_candidate_k=4,
    )
    second = source_candidates(
        index,
        "claim decisive phrase",
        source_candidate_k=2,
        passages_per_source=2,
        dense_candidate_k=4,
    )

    assert {item.source_key for item in first} == {"https://a.test/", "https://b.test/"}
    assert len([item for item in first if item.source_key == "https://a.test/"]) <= 2
    assert first == second


class FakeEncoder:
    model_id = "fixture-dense"

    def score(self, query: str, passages: list[str]) -> list[float]:
        del query
        return [1.0 if "decisive" in passage else 0.0 for passage in passages]


class FailingEncoder:
    model_id = "fixture-failing"

    def score(self, query: str, passages: list[str]) -> list[float]:
        del query, passages
        raise RuntimeError("boom")


def _hybrid_settings() -> dict[str, object]:
    return {
        "source_candidate_k": 2,
        "passages_per_source": 2,
        "dense_candidate_k": 4,
        "lexical_weight": 0.2,
        "source_weight": 0.1,
        "dense_weight": 0.7,
        "final_per_source": 2,
    }


@pytest.mark.asyncio
async def test_hybrid_reranker_promotes_semantic_passage_and_caps_sources(tmp_path: Path) -> None:
    rows = [
        ("a-1", "Source A", "https://a.test/", "generic claim text"),
        ("a-2", "Source A", "https://a.test/", "another generic claim"),
        ("b-1", "Source B", "https://b.test/", "decisive evidence"),
    ]
    path = tmp_path / "claim-1.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for evidence_id, title, source_url, text in rows:
            handle.write(json.dumps({
                "evidence_id": evidence_id, "title": title, "source_url": source_url,
                "text": text, "snapshot_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }) + "\n")
    result = await AveritecHybridProvider(
        tmp_path, settings=_hybrid_settings(), encoder=FakeEncoder()
    ).search("claim-1", "paraphrased query", top_k=3, max_chars=80)
    assert result[0].evidence_id == "b-1"
    assert len(result) == 3


@pytest.mark.asyncio
async def test_encoder_failure_is_explicit(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    provider = AveritecHybridProvider(
        tmp_path, settings=_hybrid_settings(), encoder=FailingEncoder()
    )
    with pytest.raises(RetrievalModelError, match="dense reranking failed"):
        await provider.search("claim-1", "query", top_k=3, max_chars=80)
