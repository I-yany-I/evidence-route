from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_route.evaluation.retrieval_diagnostics import (
    RetrievalObservation,
    canonicalize_source_identity,
    evaluate_observation,
)


def test_evaluate_observation_counts_canonical_gold_source_hit() -> None:
    observation = RetrievalObservation(
        claim_id="dev-0",
        candidate_evidence_ids=["e-1", "e-2"],
        candidate_source_urls=["https://example.org/article#part"],
        final_evidence_ids=["e-2"],
        final_source_urls=["https://example.org/article"],
        elapsed_ms=12,
    )

    result = evaluate_observation(
        observation,
        gold_source_urls=["HTTPS://EXAMPLE.ORG:443/article"],
    )

    assert result.claim_id == "dev-0"
    assert result.source_hit is True
    assert result.final_hit is True
    assert result.candidate_count == 2


def test_evaluate_observation_ignores_non_url_gold_placeholders() -> None:
    observation = RetrievalObservation(
        claim_id="dev-0",
        candidate_evidence_ids=["e-1"],
        candidate_source_urls=["https://example.org/article"],
        final_evidence_ids=["e-1"],
        final_source_urls=["https://example.org/article"],
        elapsed_ms=1,
    )

    result = evaluate_observation(
        observation,
        gold_source_urls=["Metadata", "https://example.org/article"],
    )

    assert result.source_hit is True
    assert result.gold_source_urls == ["https://example.org/article"]


def test_source_identity_treats_wayback_url_as_original_url() -> None:
    assert canonicalize_source_identity(
        "https://web.archive.org/web/20230420110826/https://example.org/article"
    ) == "https://example.org/article"


@pytest.mark.asyncio
async def test_diagnostic_resume_skips_completed_matching_progress(tmp_path: Path) -> None:
    from evidence_route.evaluation.retrieval_diagnostics import run_diagnostic

    runtime = tmp_path / "runtime.json"
    gold = tmp_path / "gold.json"
    corpus = tmp_path / "corpora"
    corpus.mkdir()
    row = {
        "evidence_id": "e-1",
        "title": "Example",
        "source_url": "https://example.org/article",
        "text": "A short fact.",
    }
    import hashlib

    row["snapshot_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
    corpus_file = corpus / "dev-0.jsonl"
    corpus_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
    corpus_hash = hashlib.sha256(corpus_file.read_bytes()).hexdigest()
    claim = "A short fact."
    claim_hash = hashlib.sha256(claim.encode()).hexdigest()
    runtime_payload = {
        "schema_version": "1",
        "dataset": "AVeriTeC",
        "revision": "a" * 40,
        "source_metadata_sha256": "b" * 64,
        "seed": 1,
        "items": [{
            "claim_id": "dev-0", "original_id": 0, "claim": claim,
            "claim_sha256": claim_hash, "split": "dev",
            "corpus_relpath": "dev-0.jsonl", "corpus_sha256": corpus_hash,
            "corpus_bytes": corpus_file.stat().st_size, "corpus_records": 1,
        }],
    }
    runtime.write_text(json.dumps(runtime_payload), encoding="utf-8")
    runtime.with_suffix(".json.sha256").write_text(
        hashlib.sha256(runtime.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    gold_payload = {
        "schema_version": "1", "dataset": "AVeriTeC", "revision": "a" * 40,
        "source_metadata_sha256": "b" * 64,
        "runtime_manifest_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "seed": 1,
        "items": [{
            "claim_id": "dev-0", "original_id": 0, "claim": claim,
            "label": "Supported", "questions": [{"answers": [{
                "source_url": "https://example.org/article"
            }]}], "justification": "fact",
        }],
    }
    gold.write_text(json.dumps(gold_payload), encoding="utf-8")
    gold.with_suffix(".json.sha256").write_text(
        hashlib.sha256(gold.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    progress = tmp_path / "progress.json"
    output = tmp_path / "diagnostic.json"
    timing = tmp_path / "timing.json"

    class FakeRetriever:
        calls = 0

        async def retrieve(self, claim_id: str, query: str) -> RetrievalObservation:
            self.calls += 1
            return RetrievalObservation(
                claim_id=claim_id,
                candidate_evidence_ids=["e-1"],
                candidate_source_urls=["https://example.org/article"],
                final_evidence_ids=["e-1"],
                final_source_urls=["https://example.org/article"],
                elapsed_ms=1,
            )

    retriever = FakeRetriever()
    first = await run_diagnostic(
        runtime_manifest=runtime,
        gold_manifest=gold,
        corpus_dir=corpus,
        config_path=None,
        progress_path=progress,
        output_path=output,
        timing_path=timing,
        retriever=retriever,
    )
    second = await run_diagnostic(
        runtime_manifest=runtime,
        gold_manifest=gold,
        corpus_dir=corpus,
        config_path=None,
        progress_path=progress,
        output_path=output,
        timing_path=timing,
        retriever=retriever,
    )

    assert first["summary"]["final_hits"] == 1
    assert second["summary"]["final_hits"] == 1
    assert retriever.calls == 1
