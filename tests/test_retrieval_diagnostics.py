from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_route.config import EvidenceSettings
from evidence_route.evaluation.retrieval_diagnostics import (
    HybridCorpusRetriever,
    RetrievalObservation,
    SourceAwareCorpusRetriever,
    canonicalize_source_identity,
    evaluate_observation,
    resolve_diagnostic_mode,
)
from evidence_route.providers.averitec_v2 import AveritecHybridProvider, RetrievalModelError


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


def test_default_diagnostic_mode_follows_retrieval_settings() -> None:
    settings = EvidenceSettings(
        retrieval_mode="source_hybrid_v2",
        source_candidate_k=2,
        passages_per_source=2,
        dense_candidate_k=4,
        final_per_source=1,
        dense_model_id="fixture",
        dense_model_revision="a" * 40,
        dense_model_receipt_sha256="b" * 64,
    )

    assert resolve_diagnostic_mode("default", settings) == "source_hybrid_v2"


def test_diagnostic_mode_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError, match="unknown retrieval diagnostic ablation"):
        resolve_diagnostic_mode("typo", EvidenceSettings())


def test_hybrid_ablation_requires_hybrid_retrieval_settings() -> None:
    with pytest.raises(ValueError, match="requires source_hybrid_v2 config"):
        resolve_diagnostic_mode("hybrid", EvidenceSettings())

    settings = EvidenceSettings(
        retrieval_mode="source_hybrid_v2",
        source_candidate_k=2,
        passages_per_source=2,
        dense_candidate_k=4,
        final_per_source=1,
        dense_model_id="fixture",
        dense_model_revision="a" * 40,
        dense_model_receipt_sha256="b" * 64,
    )
    assert resolve_diagnostic_mode("hybrid", settings) == "source_hybrid_v2"


@pytest.mark.asyncio
async def test_source_only_retriever_returns_diversified_candidates(tmp_path: Path) -> None:
    corpus = tmp_path / "corpora"
    corpus.mkdir()
    rows = [
        {
            "evidence_id": "a-1",
            "title": "A",
            "source_url": "https://a.test/",
            "text": "claim common",
        },
        {
            "evidence_id": "a-2",
            "title": "A",
            "source_url": "https://a.test/",
            "text": "claim common two",
        },
        {
            "evidence_id": "b-1",
            "title": "B",
            "source_url": "https://b.test/",
            "text": "decisive phrase",
        },
    ]
    import hashlib

    with (corpus / "dev-0.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            row["snapshot_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
            handle.write(json.dumps(row) + "\n")

    observation = await SourceAwareCorpusRetriever(
        corpus,
        source_candidate_k=2,
        passages_per_source=1,
        candidate_k=2,
    ).retrieve("dev-0", "claim decisive")

    assert set(observation.candidate_source_urls) == {"https://a.test/", "https://b.test/"}


def test_source_only_defaults_to_one_passage_per_source() -> None:
    from evidence_route.evaluation.retrieval_diagnostics import source_only_settings

    settings = source_only_settings({"single_top_k": 8, "single_chars": 800})

    assert settings["passages_per_source"] == 1
    assert settings["source_candidate_k"] >= settings["candidate_k"]


def test_source_only_uses_wide_source_candidate_pool() -> None:
    from evidence_route.evaluation.retrieval_diagnostics import source_only_settings

    settings = source_only_settings({"single_top_k": 8, "single_chars": 800})

    assert settings["candidate_k"] >= 256
    assert settings["source_candidate_k"] >= settings["candidate_k"]


@pytest.mark.asyncio
async def test_hybrid_diagnostic_uses_configured_candidate_limits(tmp_path: Path) -> None:
    corpus = tmp_path / "corpora"
    corpus.mkdir()
    rows = [
        {
            "evidence_id": f"{source}-{passage}",
            "title": source,
            "source_url": f"https://{source}.test/",
            "text": f"target {passage}",
        }
        for source in ("a", "b", "c")
        for passage in (1, 2)
    ]
    import hashlib

    with (corpus / "dev-0.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            row["snapshot_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
            handle.write(json.dumps(row) + "\n")

    class Encoder:
        model_id = "fixture"

        def score(self, query: str, passages: list[str]) -> list[float]:
            del query
            return [0.0 for _ in passages]

    settings = EvidenceSettings(
        retrieval_mode="source_hybrid_v2",
        single_top_k=2,
        source_candidate_k=2,
        passages_per_source=2,
        dense_candidate_k=3,
        final_per_source=1,
        dense_model_id="fixture",
        dense_model_revision="a" * 40,
        dense_model_receipt_sha256="b" * 64,
    )

    observation = await HybridCorpusRetriever(
        corpus,
        encoder=Encoder(),
        settings=settings,
    ).retrieve("dev-0", "query")

    assert observation.candidate_evidence_ids == ["a-1", "a-2", "b-1"]
    assert len(observation.final_evidence_ids) == 2


@pytest.mark.asyncio
async def test_hybrid_diagnostic_uses_configured_ranking_weights(tmp_path: Path) -> None:
    corpus = tmp_path / "corpora"
    corpus.mkdir()
    rows = [
        {
            "evidence_id": "lexical",
            "title": "Lexical",
            "source_url": "https://lexical.test/",
            "text": "target target target",
        },
        {
            "evidence_id": "dense",
            "title": "Dense",
            "source_url": "https://dense.test/",
            "text": "unrelated",
        },
        {
            "evidence_id": "filler",
            "title": "Filler",
            "source_url": "https://filler.test/",
            "text": "background",
        },
    ]
    import hashlib

    with (corpus / "dev-0.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            row["snapshot_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
            handle.write(json.dumps(row) + "\n")

    class Encoder:
        model_id = "fixture"

        def score(self, query: str, passages: list[str]) -> list[float]:
            del query
            return [1.0 if "unrelated" in passage else 0.0 for passage in passages]

    common = {
        "retrieval_mode": "source_hybrid_v2",
        "single_top_k": 1,
        "source_candidate_k": 3,
        "passages_per_source": 1,
        "dense_candidate_k": 3,
        "final_per_source": 1,
        "dense_model_id": "fixture",
        "dense_model_revision": "a" * 40,
        "dense_model_receipt_sha256": "b" * 64,
    }
    lexical = await HybridCorpusRetriever(
        corpus,
        encoder=Encoder(),
        settings=EvidenceSettings(
            **common,
            lexical_weight=1.0,
            source_weight=0.0,
            dense_weight=0.0,
        ),
    ).retrieve("dev-0", "target")
    dense = await HybridCorpusRetriever(
        corpus,
        encoder=Encoder(),
        settings=EvidenceSettings(
            **common,
            lexical_weight=0.0,
            source_weight=0.0,
            dense_weight=1.0,
        ),
    ).retrieve("dev-0", "target")

    assert lexical.final_evidence_ids == ["lexical"]
    assert dense.final_evidence_ids == ["dense"]


@pytest.mark.asyncio
async def test_hybrid_diagnostic_matches_production_provider_ranking(tmp_path: Path) -> None:
    corpus = tmp_path / "corpora"
    corpus.mkdir()
    rows = [
        {
            "evidence_id": evidence_id,
            "title": source,
            "source_url": f"https://{source}.test/",
            "text": text,
        }
        for evidence_id, source, text in (
            ("a-1", "a", "target target"),
            ("a-2", "a", "target"),
            ("b-1", "b", "semantic match"),
            ("c-1", "c", "background"),
        )
    ]
    import hashlib

    with (corpus / "dev-0.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            row["snapshot_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
            handle.write(json.dumps(row) + "\n")

    class Encoder:
        model_id = "fixture"

        def score(self, query: str, passages: list[str]) -> list[float]:
            del query
            return [1.0 if "semantic match" in passage else 0.0 for passage in passages]

    settings = EvidenceSettings(
        retrieval_mode="source_hybrid_v2",
        single_top_k=3,
        source_candidate_k=3,
        passages_per_source=2,
        dense_candidate_k=4,
        final_per_source=2,
        dense_model_id="fixture",
        dense_model_revision="a" * 40,
        dense_model_receipt_sha256="b" * 64,
        lexical_weight=0.2,
        source_weight=0.1,
        dense_weight=0.7,
    )
    diagnostic = await HybridCorpusRetriever(
        corpus,
        encoder=Encoder(),
        settings=settings,
    ).retrieve("dev-0", "target")
    production = await AveritecHybridProvider(
        corpus,
        settings=settings.model_dump(mode="python"),
        encoder=Encoder(),
    ).search(
        "dev-0",
        "target",
        top_k=settings.single_top_k,
        max_chars=settings.single_chars,
    )

    assert diagnostic.final_evidence_ids == [item.evidence_id for item in production]


@pytest.mark.asyncio
async def test_hybrid_diagnostic_wraps_score_count_mismatch(tmp_path: Path) -> None:
    corpus = tmp_path / "corpora"
    corpus.mkdir()
    text = "target"
    import hashlib

    (corpus / "dev-0.jsonl").write_text(
        json.dumps({
            "evidence_id": "a-1",
            "title": "a",
            "source_url": "https://a.test/",
            "text": text,
            "snapshot_sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
        + "\n",
        encoding="utf-8",
    )

    class ShortEncoder:
        model_id = "fixture"

        def score(self, query: str, passages: list[str]) -> list[float]:
            del query, passages
            return []

    retriever = HybridCorpusRetriever(
        corpus,
        encoder=ShortEncoder(),
        settings=EvidenceSettings(
            retrieval_mode="source_hybrid_v2",
            single_top_k=1,
            source_candidate_k=1,
            passages_per_source=1,
            dense_candidate_k=1,
            final_per_source=1,
            dense_model_id="fixture",
            dense_model_revision="a" * 40,
            dense_model_receipt_sha256="b" * 64,
        ),
    )

    with pytest.raises(RetrievalModelError, match="score count mismatch"):
        await retriever.retrieve("dev-0", "target")


@pytest.mark.asyncio
async def test_hybrid_paths_reject_blank_query_consistently(tmp_path: Path) -> None:
    corpus = tmp_path / "corpora"
    corpus.mkdir()
    text = "target"
    import hashlib

    (corpus / "dev-0.jsonl").write_text(
        json.dumps({
            "evidence_id": "a-1",
            "title": "a",
            "source_url": "https://a.test/",
            "text": text,
            "snapshot_sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
        + "\n",
        encoding="utf-8",
    )

    class Encoder:
        model_id = "fixture"

        def score(self, query: str, passages: list[str]) -> list[float]:
            del query
            return [0.0 for _ in passages]

    settings = EvidenceSettings(
        retrieval_mode="source_hybrid_v2",
        single_top_k=1,
        source_candidate_k=1,
        passages_per_source=1,
        dense_candidate_k=1,
        final_per_source=1,
        dense_model_id="fixture",
        dense_model_revision="a" * 40,
        dense_model_receipt_sha256="b" * 64,
    )
    provider = AveritecHybridProvider(
        corpus,
        settings=settings.model_dump(mode="python"),
        encoder=Encoder(),
    )
    diagnostic = HybridCorpusRetriever(corpus, encoder=Encoder(), settings=settings)

    with pytest.raises(ValueError, match="^query must not be empty$"):
        await provider.search("dev-0", "   ", top_k=1, max_chars=80)
    with pytest.raises(ValueError, match="^query must not be empty$"):
        await diagnostic.retrieve("dev-0", "   ")


@pytest.mark.asyncio
async def test_hybrid_diagnostic_uses_configured_final_per_source(tmp_path: Path) -> None:
    corpus = tmp_path / "corpora"
    corpus.mkdir()
    rows = [
        {
            "evidence_id": evidence_id,
            "title": source,
            "source_url": f"https://{source}.test/",
            "text": text,
        }
        for evidence_id, source, text in (
            ("a-1", "a", "target target target"),
            ("a-2", "a", "target target"),
            ("b-1", "b", "target"),
        )
    ]
    import hashlib

    with (corpus / "dev-0.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            row["snapshot_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
            handle.write(json.dumps(row) + "\n")

    class Encoder:
        model_id = "fixture"

        def score(self, query: str, passages: list[str]) -> list[float]:
            del query
            return [0.0 for _ in passages]

    observation = await HybridCorpusRetriever(
        corpus,
        encoder=Encoder(),
        settings=EvidenceSettings(
            retrieval_mode="source_hybrid_v2",
            single_top_k=3,
            source_candidate_k=2,
            passages_per_source=2,
            dense_candidate_k=3,
            final_per_source=2,
            dense_model_id="fixture",
            dense_model_revision="a" * 40,
            dense_model_receipt_sha256="b" * 64,
            lexical_weight=1.0,
            source_weight=0.0,
            dense_weight=0.0,
        ),
    ).retrieve("dev-0", "target")

    assert len(observation.final_evidence_ids) == 3
    assert observation.final_source_urls.count("https://a.test/") == 2


@pytest.mark.asyncio
async def test_run_diagnostic_passes_complete_hybrid_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evidence_route.evaluation.retrieval_diagnostics as diagnostics

    runtime = tmp_path / "runtime.json"
    gold = tmp_path / "gold.json"
    corpus = tmp_path / "corpora"
    config = tmp_path / "config.yaml"
    corpus.mkdir()
    runtime.write_text("{}", encoding="utf-8")
    gold.write_text("{}", encoding="utf-8")
    expected = EvidenceSettings(
        retrieval_mode="source_hybrid_v2",
        single_top_k=3,
        source_candidate_k=2,
        passages_per_source=2,
        dense_candidate_k=3,
        final_per_source=2,
        dense_model_id="fixture",
        dense_model_revision="a" * 40,
        dense_model_receipt_sha256="b" * 64,
        lexical_weight=0.6,
        source_weight=0.3,
        dense_weight=0.1,
    )
    config.write_text(
        json.dumps({"evidence": expected.model_dump(mode="json")}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    encoder = object()

    monkeypatch.setattr(diagnostics, "load_runtime_manifest", lambda *args, **kwargs: object())
    monkeypatch.setattr(diagnostics, "load_gold_manifest", lambda *args, **kwargs: object())
    monkeypatch.setattr(diagnostics, "align_runtime_and_gold", lambda *args: [])
    monkeypatch.setattr(diagnostics, "manifest_digest", lambda path: "c" * 64)
    monkeypatch.setattr(
        diagnostics,
        "build_evidence_provider",
        lambda corpus_dir, settings: type("Provider", (), {"encoder": encoder})(),
    )

    class CapturingHybridRetriever:
        def __init__(
            self,
            corpus_dir: Path,
            *,
            encoder: object,
            settings: EvidenceSettings,
        ) -> None:
            captured.update(corpus_dir=corpus_dir, encoder=encoder, settings=settings)

    monkeypatch.setattr(diagnostics, "HybridCorpusRetriever", CapturingHybridRetriever)

    await diagnostics.run_diagnostic(
        runtime_manifest=runtime,
        gold_manifest=gold,
        corpus_dir=corpus,
        config_path=config,
        progress_path=tmp_path / "progress.json",
        output_path=tmp_path / "diagnostic.json",
        timing_path=tmp_path / "timing.json",
    )

    assert captured == {"corpus_dir": corpus, "encoder": encoder, "settings": expected}


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
    assert "elapsed_ms" not in first["records"][0]
    timing_payload = json.loads(timing.read_text(encoding="utf-8"))
    assert timing_payload["records"] == [{"claim_id": "dev-0", "elapsed_ms": 1}]

    with pytest.raises(ValueError, match="unknown retrieval diagnostic ablation"):
        await run_diagnostic(
            runtime_manifest=runtime,
            gold_manifest=gold,
            corpus_dir=corpus,
            config_path=None,
            progress_path=tmp_path / "typo-progress.json",
            output_path=tmp_path / "typo-output.json",
            timing_path=tmp_path / "typo-timing.json",
            retriever=retriever,
            ablation="typo",
        )
