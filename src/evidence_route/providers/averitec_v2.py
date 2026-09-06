"""Deterministic source-aware candidate generation over frozen AVeriTeC data."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evidence_route.contracts import Evidence
from evidence_route.evaluation.stability import canonicalize_citation_url
from evidence_route.providers.averitec import (
    AveritecFrozenProvider,
    _Index,
    _Record,
    tokenize,
)
from evidence_route.providers.dense import DenseEncoder, RetrievalModelError

FrozenEvidenceRecord = _Record
FrozenClaimIndex = _Index


@dataclass(frozen=True)
class RetrievalCandidate:
    record: FrozenEvidenceRecord
    source_key: str
    lexical_score: float
    source_rank: int


@dataclass(frozen=True)
class HybridRetrievalTrace:
    candidates: list[RetrievalCandidate]
    dense_scores: list[float]
    ranked: list[tuple[RetrievalCandidate, float]]


def load_frozen_index(corpus_path: Path) -> FrozenClaimIndex:
    """Load one claim corpus using the exact v1 validation and BM25 index."""

    provider = AveritecFrozenProvider(Path(corpus_path).parent)
    claim_id = Path(corpus_path).stem
    return provider._get_index(claim_id)


def source_candidates(
    index: FrozenClaimIndex,
    query: str,
    *,
    source_candidate_k: int,
    passages_per_source: int,
    dense_candidate_k: int,
) -> list[RetrievalCandidate]:
    if source_candidate_k < 1 or passages_per_source < 1 or dense_candidate_k < 1:
        raise ValueError("candidate limits must be positive")
    scores = index.bm25.get_scores(tokenize(query))
    by_source: dict[str, list[tuple[FrozenEvidenceRecord, float]]] = defaultdict(list)
    for record, score in zip(index.records, scores, strict=True):
        by_source[canonicalize_citation_url(record.source_url)].append((record, float(score)))

    def sentence_key(item: tuple[FrozenEvidenceRecord, float]) -> tuple[float, str, str, str]:
        record, score = item
        return (
            -score,
            record.evidence_id,
            canonicalize_citation_url(record.source_url),
            record.snapshot_sha256,
        )

    def source_key(
        item: tuple[str, list[tuple[FrozenEvidenceRecord, float]]],
    ) -> tuple[float, str, str, str]:
        canonical_url, rows = item
        best_record, best_score = min(rows, key=sentence_key)
        return (
            -best_score,
            best_record.evidence_id,
            canonical_url,
            best_record.snapshot_sha256,
        )

    ranked_sources = sorted(by_source.items(), key=source_key)[:source_candidate_k]
    candidates: list[RetrievalCandidate] = []
    for source_rank, (canonical_url, rows) in enumerate(ranked_sources):
        for record, score in sorted(rows, key=sentence_key)[:passages_per_source]:
            candidates.append(RetrievalCandidate(record, canonical_url, score, source_rank))
    return candidates[:dense_candidate_k]


def _min_max(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _take_with_source_cap(
    ranked: Sequence[tuple[RetrievalCandidate, float]], *, top_k: int, per_source: int
) -> list[tuple[RetrievalCandidate, float]]:
    result: list[tuple[RetrievalCandidate, float]] = []
    counts: dict[str, int] = {}
    for candidate, score in ranked:
        count = counts.get(candidate.source_key, 0)
        if count >= per_source:
            continue
        result.append((candidate, score))
        counts[candidate.source_key] = count + 1
        if len(result) >= top_k:
            break
    return result


def fuse_candidates(
    candidates: Sequence[RetrievalCandidate],
    dense_scores: Sequence[float],
    *,
    lexical_weight: float,
    source_weight: float,
    dense_weight: float,
    top_k: int,
    final_per_source: int,
) -> list[tuple[RetrievalCandidate, float]]:
    if not candidates or len(candidates) != len(dense_scores):
        raise ValueError("dense score count must match candidates")
    if top_k < 1 or final_per_source < 1:
        raise ValueError("ranking limits must be positive")
    weights = (lexical_weight, source_weight, dense_weight)
    if any(weight < 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("ranking weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0):
        raise ValueError("ranking weights must sum to one")
    if any(not math.isfinite(float(value)) for value in dense_scores):
        raise ValueError("dense scores must be finite")
    lexical = _min_max([item.lexical_score for item in candidates])
    source = _min_max([-float(item.source_rank) for item in candidates])
    ranked = sorted(
        zip(candidates, lexical, source, dense_scores, strict=True),
        key=lambda item: (
            -(
                lexical_weight * item[1]
                + source_weight * item[2]
                + dense_weight * float(item[3])
            ),
            item[0].source_rank,
            item[0].record.evidence_id,
            item[0].source_key,
            item[0].record.snapshot_sha256,
        ),
    )
    scored = [
        (
            item[0],
            float(
                item[1] * lexical_weight
                + item[2] * source_weight
                + item[3] * dense_weight
            ),
        )
        for item in ranked
    ]
    return _take_with_source_cap(
        scored,
        top_k=top_k,
        per_source=final_per_source,
    )


def trace_hybrid_retrieval(
    index: FrozenClaimIndex,
    query: str,
    *,
    settings: Mapping[str, object],
    encoder: DenseEncoder,
    top_k: int,
    max_per_source: int | None = None,
) -> HybridRetrievalTrace:
    if not query.strip():
        raise ValueError("query must not be empty")
    candidates = source_candidates(
        index,
        query,
        source_candidate_k=int(settings["source_candidate_k"]),
        passages_per_source=int(settings["passages_per_source"]),
        dense_candidate_k=int(settings["dense_candidate_k"]),
    )
    passages = [f"{item.record.title}\n\n{item.record.text}" for item in candidates]
    try:
        dense_scores = encoder.score(query, passages)
    except Exception as exc:
        raise RetrievalModelError(f"dense reranking failed: {exc}") from exc
    if len(dense_scores) != len(candidates):
        raise RetrievalModelError("dense reranking failed: score count mismatch")
    try:
        ranked = fuse_candidates(
            candidates,
            dense_scores,
            lexical_weight=float(settings["lexical_weight"]),
            source_weight=float(settings["source_weight"]),
            dense_weight=float(settings["dense_weight"]),
            top_k=top_k,
            final_per_source=int(settings.get("final_per_source", max_per_source or top_k)),
        )
    except ValueError as exc:
        raise RetrievalModelError(f"dense reranking failed: {exc}") from exc
    return HybridRetrievalTrace(
        candidates=candidates,
        dense_scores=dense_scores,
        ranked=ranked,
    )


class AveritecHybridProvider:
    retrieval_config_name = "evidence-route-source-hybrid-v2"

    def __init__(
        self,
        corpus_dir: Path,
        *,
        settings: Mapping[str, object],
        encoder: DenseEncoder,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.settings = settings
        self.encoder = encoder

    async def search(
        self,
        claim_id: str,
        query: str,
        *,
        top_k: int,
        max_chars: int,
        max_per_source: int | None = None,
    ) -> list[Evidence]:
        if top_k < 1 or max_chars < 1:
            raise ValueError("top_k and max_chars must be positive")
        trace = trace_hybrid_retrieval(
            load_frozen_index(self.corpus_dir / f"{claim_id}.jsonl"),
            query,
            settings=self.settings,
            encoder=self.encoder,
            top_k=top_k,
            max_per_source=max_per_source,
        )
        return [
            Evidence(
                evidence_id=item.record.evidence_id,
                title=item.record.title,
                source_url=item.record.source_url,
                text=item.record.text[:max_chars],
                provider="averitec_frozen",
                snapshot_sha256=item.record.snapshot_sha256,
                ranking_score=score,
                char_start=0,
                char_end=min(len(item.record.text), max_chars),
            )
            for item, score in trace.ranked
        ]


__all__ = [
    "FrozenClaimIndex",
    "FrozenEvidenceRecord",
    "AveritecHybridProvider",
    "HybridRetrievalTrace",
    "RetrievalModelError",
    "RetrievalCandidate",
    "load_frozen_index",
    "fuse_candidates",
    "source_candidates",
    "trace_hybrid_retrieval",
]
