from __future__ import annotations

import json
from pathlib import Path

import pytest

import evidence_route.evaluation.retrieval_diagnostics as retrieval_diagnostics
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


def _gate_records(
    *,
    claim_count: int = 32,
    source_hits: int = 22,
    final_hits: int = 14,
    sentinel_final_hit: bool = True,
) -> list[dict[str, object]]:
    claim_ids = ["train-2468", *[f"train-{index}" for index in range(claim_count - 1)]]
    records = [
        {
            "claim_id": claim_id,
            "candidate_count": 8,
            "source_hit": index < source_hits,
            "final_hit": index < final_hits,
        }
        for index, claim_id in enumerate(claim_ids)
    ]
    if records and not sentinel_final_hit:
        records[0]["final_hit"] = False
        if final_hits < len(records):
            records[final_hits]["final_hit"] = True
    return records


def _gate_summary(records: list[dict[str, object]]) -> dict[str, int]:
    return {
        "claim_count": len(records),
        "source_hits": sum(bool(item["source_hit"]) for item in records),
        "final_hits": sum(bool(item["final_hit"]) for item in records),
        "candidate_count": sum(int(item["candidate_count"]) for item in records),
    }


def test_retrieval_gates_pass_at_declared_boundaries() -> None:
    records = _gate_records()
    result = retrieval_diagnostics.evaluate_retrieval_gates(_gate_summary(records), records)

    assert list(result) == [
        "claim_count",
        "unique_claim_count",
        "sentinel_present",
        "candidate_source_hits",
        "final_top8_hits",
        "sentinel_final_hit",
        "failed_gates",
        "passed",
    ]
    assert result == {
        "claim_count": {"actual": 32, "required": 32, "passed": True},
        "unique_claim_count": {"actual": 32, "required": 32, "passed": True},
        "sentinel_present": {"actual": True, "required": True, "passed": True},
        "candidate_source_hits": {"actual": 22, "required": 22, "passed": True},
        "final_top8_hits": {"actual": 14, "required": 14, "passed": True},
        "sentinel_final_hit": {"actual": True, "required": True, "passed": True},
        "failed_gates": [],
        "passed": True,
    }


@pytest.mark.parametrize("claim_count", [31, 33])
def test_retrieval_gates_require_exactly_32_unique_claims(claim_count: int) -> None:
    records = _gate_records(claim_count=claim_count)
    result = retrieval_diagnostics.evaluate_retrieval_gates(_gate_summary(records), records)

    assert result["claim_count"] == {
        "actual": claim_count,
        "required": 32,
        "passed": False,
    }
    assert result["unique_claim_count"] == {
        "actual": claim_count,
        "required": 32,
        "passed": False,
    }
    assert result["passed"] is False


def test_retrieval_gates_reject_duplicate_claims() -> None:
    records = _gate_records()
    records[-1]["claim_id"] = records[-2]["claim_id"]

    result = retrieval_diagnostics.evaluate_retrieval_gates(_gate_summary(records), records)

    assert result["claim_count"]["passed"] is True
    assert result["unique_claim_count"] == {
        "actual": 31,
        "required": 32,
        "passed": False,
    }
    assert result["failed_gates"] == ["unique_claim_count"]


def test_retrieval_gates_require_train_2468() -> None:
    records = _gate_records()
    records[0]["claim_id"] = "train-9999"

    result = retrieval_diagnostics.evaluate_retrieval_gates(_gate_summary(records), records)

    assert result["sentinel_present"] == {
        "actual": False,
        "required": True,
        "passed": False,
    }
    assert result["failed_gates"] == ["sentinel_present", "sentinel_final_hit"]


def test_retrieval_gates_reject_21_candidate_source_hits() -> None:
    records = _gate_records(source_hits=21)
    result = retrieval_diagnostics.evaluate_retrieval_gates(_gate_summary(records), records)

    assert result["candidate_source_hits"] == {
        "actual": 21,
        "required": 22,
        "passed": False,
    }
    assert result["failed_gates"] == ["candidate_source_hits"]


def test_retrieval_gates_reject_13_final_top8_hits() -> None:
    records = _gate_records(final_hits=13)
    result = retrieval_diagnostics.evaluate_retrieval_gates(_gate_summary(records), records)

    assert result["final_top8_hits"] == {
        "actual": 13,
        "required": 14,
        "passed": False,
    }
    assert result["failed_gates"] == ["final_top8_hits"]


def test_retrieval_gates_require_train_2468_final_hit() -> None:
    records = _gate_records(sentinel_final_hit=False)
    result = retrieval_diagnostics.evaluate_retrieval_gates(_gate_summary(records), records)

    assert result["final_top8_hits"]["passed"] is True
    assert result["sentinel_final_hit"] == {
        "actual": False,
        "required": True,
        "passed": False,
    }
    assert result["failed_gates"] == ["sentinel_final_hit"]


@pytest.mark.parametrize(
    "field",
    ["claim_count", "source_hits", "final_hits", "candidate_count"],
)
def test_retrieval_gates_reject_summary_record_mismatch(field: str) -> None:
    records = _gate_records()
    summary = _gate_summary(records)
    summary[field] += 1

    with pytest.raises(ValueError) as exc_info:
        retrieval_diagnostics.evaluate_retrieval_gates(summary, records)

    assert str(exc_info.value) == f"retrieval diagnostic summary mismatch: {field}"


@pytest.mark.parametrize("invalid_claim_id", [None, "", "   ", "claim-1", "train-x"])
def test_retrieval_gates_reject_missing_blank_or_invalid_claim_id(
    invalid_claim_id: str | None,
) -> None:
    records = _gate_records()
    if invalid_claim_id is None:
        records[-1].pop("claim_id")
    else:
        records[-1]["claim_id"] = invalid_claim_id

    with pytest.raises(ValueError) as exc_info:
        retrieval_diagnostics.evaluate_retrieval_gates(_gate_summary(records), records)

    assert str(exc_info.value) == "retrieval diagnostic record 31 invalid claim_id"


@pytest.mark.parametrize("claim_id", ["dev-9000", "test-9000"])
def test_retrieval_gates_accept_supported_non_train_claim_id(claim_id: str) -> None:
    records = _gate_records()
    records[-1]["claim_id"] = claim_id

    result = retrieval_diagnostics.evaluate_retrieval_gates(_gate_summary(records), records)

    assert result["unique_claim_count"]["passed"] is True


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("source_hit", "false"),
        ("source_hit", 0),
        ("source_hit", 1),
        ("final_hit", "false"),
        ("final_hit", 0),
        ("final_hit", 1),
    ],
)
def test_retrieval_gates_require_exact_boolean_hit_fields(
    field: str, invalid_value: object
) -> None:
    records = _gate_records()
    records[-1][field] = invalid_value

    with pytest.raises(ValueError) as exc_info:
        retrieval_diagnostics.evaluate_retrieval_gates(_gate_summary(records), records)

    assert str(exc_info.value) == f"retrieval diagnostic record 31 invalid {field}"


@pytest.mark.parametrize("invalid_value", [True, "8", 8.0, -1])
def test_retrieval_gates_require_nonnegative_exact_integer_candidate_count(
    invalid_value: object,
) -> None:
    records = _gate_records()
    records[-1]["candidate_count"] = invalid_value

    with pytest.raises(ValueError) as exc_info:
        retrieval_diagnostics.evaluate_retrieval_gates(_gate_summary(records), records)

    assert str(exc_info.value) == (
        "retrieval diagnostic record 31 invalid candidate_count"
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


@pytest.mark.parametrize(("gold_index", "expected"), [(7, True), (8, False)])
def test_evaluate_observation_limits_final_hit_to_top_eight(
    gold_index: int, expected: bool
) -> None:
    source_urls = [f"https://example.org/article-{index}" for index in range(9)]
    evidence_ids = [f"e-{index}" for index in range(9)]
    observation = RetrievalObservation(
        claim_id="dev-0",
        candidate_evidence_ids=evidence_ids,
        candidate_source_urls=source_urls,
        final_evidence_ids=evidence_ids,
        final_source_urls=source_urls,
        elapsed_ms=1,
    )

    result = evaluate_observation(
        observation,
        gold_source_urls=[source_urls[gold_index]],
    )

    assert result.source_hit is True
    assert result.final_hit is expected
    assert result.final_evidence_ids == evidence_ids
    assert result.final_source_urls == source_urls


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

    assert observation.candidate_evidence_ids == ["a-1", "b-1", "a-2"]
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
    import hashlib

    claim_ids = ["train-2468", *[f"train-{index}" for index in range(31)]]
    runtime_items: list[dict[str, object]] = []
    gold_items: list[dict[str, object]] = []
    for claim_id in claim_ids:
        original_id = int(claim_id.split("-", 1)[1])
        claim = f"A short fact for {claim_id}."
        row = {
            "evidence_id": f"{claim_id}-e-1",
            "title": "Example",
            "source_url": "https://example.org/article",
            "text": claim,
        }
        row["snapshot_sha256"] = hashlib.sha256(claim.encode()).hexdigest()
        corpus_file = corpus / f"{claim_id}.jsonl"
        corpus_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
        runtime_items.append(
            {
                "claim_id": claim_id,
                "original_id": original_id,
                "claim": claim,
                "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
                "split": "train",
                "corpus_relpath": f"{claim_id}.jsonl",
                "corpus_sha256": hashlib.sha256(corpus_file.read_bytes()).hexdigest(),
                "corpus_bytes": corpus_file.stat().st_size,
                "corpus_records": 1,
            }
        )
        gold_items.append(
            {
                "claim_id": claim_id,
                "original_id": original_id,
                "claim": claim,
                "label": "Supported",
                "questions": [{"answers": [{"source_url": "https://example.org/article"}]}],
                "justification": "fact",
            }
        )
    runtime_payload = {
        "schema_version": "1",
        "dataset": "AVeriTeC",
        "revision": "a" * 40,
        "source_metadata_sha256": "b" * 64,
        "seed": 1,
        "items": runtime_items,
    }
    runtime.write_text(json.dumps(runtime_payload), encoding="utf-8")
    runtime.with_suffix(".json.sha256").write_text(
        hashlib.sha256(runtime.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    gold_payload = {
        "schema_version": "1",
        "dataset": "AVeriTeC",
        "revision": "a" * 40,
        "source_metadata_sha256": "b" * 64,
        "runtime_manifest_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "seed": 1,
        "items": gold_items,
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

    assert first["summary"]["final_hits"] == 32
    assert second["summary"]["final_hits"] == 32
    assert first["gates"] == retrieval_diagnostics.evaluate_retrieval_gates(
        first["summary"], first["records"]
    )
    assert first["gates"]["passed"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["gates"] == first["gates"]
    assert retriever.calls == 32
    assert "elapsed_ms" not in first["records"][0]
    timing_payload = json.loads(timing.read_text(encoding="utf-8"))
    assert timing_payload["records"] == [
        {"claim_id": claim_id, "elapsed_ms": 1} for claim_id in claim_ids
    ]

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
