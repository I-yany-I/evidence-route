"""Offline, resumable retrieval diagnostics for frozen AVeriTeC corpora."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlsplit

import yaml

from evidence_route.artifacts import atomic_write_json
from evidence_route.config import EvidenceSettings
from evidence_route.evaluation.runtime_manifest import load_runtime_manifest, manifest_digest
from evidence_route.evaluation.scorer_manifest import align_runtime_and_gold, load_gold_manifest
from evidence_route.evaluation.stability import canonicalize_citation_url
from evidence_route.providers.averitec import AveritecFrozenProvider
from evidence_route.providers.averitec_v2 import (
    load_frozen_index,
    source_candidates,
    trace_hybrid_retrieval,
)
from evidence_route.providers.dense import DenseEncoder
from evidence_route.retrieval import build_evidence_provider

_CLAIM_ID_RE = re.compile(r"^(?:train|dev|test)-\d+$")


@dataclass(frozen=True)
class RetrievalObservation:
    claim_id: str
    candidate_evidence_ids: list[str]
    candidate_source_urls: list[str]
    final_evidence_ids: list[str]
    final_source_urls: list[str]
    elapsed_ms: int
    candidate_lexical_scores: list[float] = field(default_factory=list)
    candidate_dense_scores: list[float] = field(default_factory=list)
    candidate_source_ranks: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalDiagnosticItem:
    claim_id: str
    candidate_count: int
    candidate_evidence_ids: list[str]
    final_evidence_ids: list[str]
    source_hit: bool
    final_hit: bool
    gold_source_urls: list[str]
    candidate_source_urls: list[str]
    final_source_urls: list[str]
    elapsed_ms: int
    candidate_lexical_scores: list[float] = field(default_factory=list)
    candidate_dense_scores: list[float] = field(default_factory=list)
    candidate_source_ranks: list[int] = field(default_factory=list)


class DiagnosticRetriever(Protocol):
    async def retrieve(self, claim_id: str, query: str) -> RetrievalObservation:
        raise NotImplementedError


class FrozenCorpusRetriever:
    def __init__(
        self,
        provider: AveritecFrozenProvider,
        *,
        top_k: int,
        max_chars: int,
        max_per_source: int | None,
    ) -> None:
        self.provider = provider
        self.top_k = top_k
        self.max_chars = max_chars
        self.max_per_source = max_per_source

    async def retrieve(self, claim_id: str, query: str) -> RetrievalObservation:
        started = time.perf_counter()
        evidence = await self.provider.search(
            claim_id,
            query,
            top_k=self.top_k,
            max_chars=self.max_chars,
            max_per_source=self.max_per_source,
        )
        elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        return RetrievalObservation(
            claim_id=claim_id,
            candidate_evidence_ids=[item.evidence_id for item in evidence],
            candidate_source_urls=[str(item.source_url) for item in evidence],
            final_evidence_ids=[item.evidence_id for item in evidence],
            final_source_urls=[str(item.source_url) for item in evidence],
            elapsed_ms=elapsed_ms,
        )


class SourceAwareCorpusRetriever:
    def __init__(
        self,
        corpus_dir: Path,
        *,
        source_candidate_k: int,
        passages_per_source: int,
        candidate_k: int,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.source_candidate_k = source_candidate_k
        self.passages_per_source = passages_per_source
        self.candidate_k = candidate_k

    async def retrieve(self, claim_id: str, query: str) -> RetrievalObservation:
        started = time.perf_counter()
        candidates = source_candidates(
            load_frozen_index(self.corpus_dir / f"{claim_id}.jsonl"),
            query,
            source_candidate_k=self.source_candidate_k,
            passages_per_source=self.passages_per_source,
            dense_candidate_k=self.candidate_k,
        )
        elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        return RetrievalObservation(
            claim_id=claim_id,
            candidate_evidence_ids=[item.record.evidence_id for item in candidates],
            candidate_source_urls=[item.record.source_url for item in candidates],
            final_evidence_ids=[item.record.evidence_id for item in candidates],
            final_source_urls=[item.record.source_url for item in candidates],
            elapsed_ms=elapsed_ms,
        )


class HybridCorpusRetriever:
    def __init__(
        self,
        corpus_dir: Path,
        *,
        encoder: DenseEncoder,
        settings: EvidenceSettings,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.encoder = encoder
        self.settings = settings

    async def retrieve(self, claim_id: str, query: str) -> RetrievalObservation:
        started = time.perf_counter()
        trace = trace_hybrid_retrieval(
            load_frozen_index(self.corpus_dir / f"{claim_id}.jsonl"),
            query,
            settings=self.settings.model_dump(mode="python"),
            encoder=self.encoder,
            top_k=self.settings.single_top_k,
        )
        elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        return RetrievalObservation(
            claim_id=claim_id,
            candidate_evidence_ids=[item.record.evidence_id for item in trace.candidates],
            candidate_source_urls=[item.record.source_url for item in trace.candidates],
            final_evidence_ids=[item.record.evidence_id for item, _ in trace.ranked],
            final_source_urls=[item.record.source_url for item, _ in trace.ranked],
            elapsed_ms=elapsed_ms,
            candidate_lexical_scores=[item.lexical_score for item in trace.candidates],
            candidate_dense_scores=[float(score) for score in trace.dense_scores],
            candidate_source_ranks=[item.source_rank for item in trace.candidates],
        )


def canonicalize_source_identity(value: str) -> str:
    decoded = unquote(value).strip()
    parts = urlsplit(decoded)
    if parts.hostname and parts.hostname.lower() == "web.archive.org":
        path = parts.path
        marker = "/web/"
        if marker in path:
            captured = path.split(marker, 1)[1]
            pieces = captured.split("/", 1)
            if len(pieces) == 2 and "://" in pieces[1]:
                decoded = pieces[1]
    return canonicalize_citation_url(decoded)


def _canonical_sources(values: list[str]) -> list[str]:
    canonical: set[str] = set()
    for value in values:
        try:
            canonical.add(canonicalize_source_identity(value))
        except ValueError:
            continue
    return sorted(canonical)


def _canonical_source_sequence(values: list[str]) -> list[str]:
    canonical: list[str] = []
    for value in values:
        try:
            canonical.append(canonicalize_source_identity(value))
        except ValueError:
            canonical.append("")
    return canonical


def evaluate_observation(
    observation: RetrievalObservation, *, gold_source_urls: list[str]
) -> RetrievalDiagnosticItem:
    gold = _canonical_sources(gold_source_urls)
    candidates = _canonical_source_sequence(observation.candidate_source_urls)
    final = _canonical_source_sequence(observation.final_source_urls)
    return RetrievalDiagnosticItem(
        claim_id=observation.claim_id,
        candidate_count=len(observation.candidate_evidence_ids),
        candidate_evidence_ids=list(observation.candidate_evidence_ids),
        final_evidence_ids=list(observation.final_evidence_ids),
        source_hit=bool(set(gold) & set(candidates)),
        final_hit=bool(set(gold) & set(final[:8])),
        gold_source_urls=gold,
        candidate_source_urls=candidates,
        final_source_urls=final,
        elapsed_ms=observation.elapsed_ms,
        candidate_lexical_scores=list(observation.candidate_lexical_scores),
        candidate_dense_scores=list(observation.candidate_dense_scores),
        candidate_source_ranks=list(observation.candidate_source_ranks),
    )


def evaluate_retrieval_gates(
    summary: dict[str, object], records: list[dict[str, object]]
) -> dict[str, object]:
    for index, item in enumerate(records):
        claim_id = item.get("claim_id")
        if type(claim_id) is not str or _CLAIM_ID_RE.fullmatch(claim_id) is None:
            raise ValueError(f"retrieval diagnostic record {index} invalid claim_id")
        for hit_field in ("source_hit", "final_hit"):
            if type(item.get(hit_field)) is not bool:
                raise ValueError(
                    f"retrieval diagnostic record {index} invalid {hit_field}"
                )
        candidate_count = item.get("candidate_count")
        if type(candidate_count) is not int or candidate_count < 0:
            raise ValueError(
                f"retrieval diagnostic record {index} invalid candidate_count"
            )

    claim_ids = [str(item["claim_id"]) for item in records]
    claim_count = len(records)
    unique_claim_count = len(set(claim_ids))
    candidate_source_hits = sum(item["source_hit"] is True for item in records)
    final_top8_hits = sum(item["final_hit"] is True for item in records)
    candidate_count = sum(int(item["candidate_count"]) for item in records)
    calculated_summary = {
        "claim_count": claim_count,
        "source_hits": candidate_source_hits,
        "final_hits": final_top8_hits,
        "candidate_count": candidate_count,
    }
    for summary_field, actual in calculated_summary.items():
        declared = summary.get(summary_field)
        if type(declared) is not int or declared != actual:
            raise ValueError(f"retrieval diagnostic summary mismatch: {summary_field}")

    sentinel_present = "train-2468" in claim_ids
    sentinel_final_hit = any(
        item["claim_id"] == "train-2468" and item["final_hit"] is True for item in records
    )
    checks: dict[str, dict[str, object]] = {
        "claim_count": {
            "actual": claim_count,
            "required": 32,
            "passed": claim_count == 32,
        },
        "unique_claim_count": {
            "actual": unique_claim_count,
            "required": 32,
            "passed": unique_claim_count == 32,
        },
        "sentinel_present": {
            "actual": sentinel_present,
            "required": True,
            "passed": sentinel_present,
        },
        "candidate_source_hits": {
            "actual": candidate_source_hits,
            "required": 22,
            "passed": candidate_source_hits >= 22,
        },
        "final_top8_hits": {
            "actual": final_top8_hits,
            "required": 14,
            "passed": final_top8_hits >= 14,
        },
        "sentinel_final_hit": {
            "actual": sentinel_final_hit,
            "required": True,
            "passed": sentinel_final_hit,
        },
    }
    failed_gates = [name for name, check in checks.items() if not check["passed"]]
    return {**checks, "failed_gates": failed_gates, "passed": not failed_gates}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_identity(config_path: Path | None) -> tuple[str, dict[str, object]]:
    if config_path is None:
        return hashlib.sha256(b"{}").hexdigest(), {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("retrieval diagnostic config must be a mapping")
    return _sha256(config_path), raw


def _gold_sources(gold_item: object) -> list[str]:
    values: list[str] = []
    for question in getattr(gold_item, "questions", []):
        if not isinstance(question, dict):
            continue
        answers = question.get("answers", [])
        if not isinstance(answers, list):
            continue
        for answer in answers:
            if isinstance(answer, dict):
                for key in ("source_url", "cached_source_url"):
                    value = answer.get(key)
                    if isinstance(value, str) and value:
                        values.append(value)
    return values


def source_only_settings(evidence_config: dict[str, object]) -> dict[str, int]:
    candidate_k = max(256, int(evidence_config.get("dense_candidate_k", 256)))
    return {
        "candidate_k": candidate_k,
        "source_candidate_k": max(
            candidate_k,
            int(evidence_config.get("source_candidate_k", candidate_k)),
        ),
        "passages_per_source": int(evidence_config.get("passages_per_source", 1)),
    }


def resolve_diagnostic_mode(ablation: str, settings: EvidenceSettings) -> str:
    if ablation == "default":
        return settings.retrieval_mode
    if ablation == "source-only":
        return "source_bm25_diagnostic"
    if ablation == "hybrid":
        if settings.retrieval_mode != "source_hybrid_v2":
            raise ValueError("hybrid diagnostic ablation requires source_hybrid_v2 config")
        return "source_hybrid_v2"
    raise ValueError(f"unknown retrieval diagnostic ablation: {ablation}")


def _common_root(*paths: Path) -> Path:
    resolved = [str(path.resolve()) for path in paths]
    return Path(__import__("os").path.commonpath(resolved))


async def run_diagnostic(
    *,
    runtime_manifest: Path,
    gold_manifest: Path,
    corpus_dir: Path,
    config_path: Path | None,
    progress_path: Path,
    output_path: Path,
    timing_path: Path,
    retriever: DiagnosticRetriever | None = None,
    ablation: str = "default",
) -> dict[str, object]:
    root = _common_root(runtime_manifest, gold_manifest, corpus_dir)
    runtime = load_runtime_manifest(runtime_manifest, allowed_root=root, corpus_root=corpus_dir)
    gold = load_gold_manifest(gold_manifest, allowed_root=root)
    aligned = align_runtime_and_gold(runtime, gold)
    config_hash, config = _config_identity(config_path)
    evidence_config = dict(config.get("evidence") or {})
    evidence_settings = EvidenceSettings.model_validate(evidence_config)
    diagnostic_mode = resolve_diagnostic_mode(ablation, evidence_settings)
    receipt = corpus_dir.parent / "preparation_receipt.json"
    identity = {
        "runtime_manifest_sha256": manifest_digest(runtime_manifest),
        "gold_manifest_sha256": _sha256(gold_manifest),
        "config_sha256": config_hash,
        "corpus_receipt_sha256": _sha256(receipt) if receipt.is_file() else None,
        "ablation": ablation,
        "retrieval_mode": diagnostic_mode,
    }
    progress_path = Path(progress_path)
    completed: dict[str, dict[str, object]] = {}
    if progress_path.is_file():
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("retrieval diagnostic progress identity differs")
        completed = dict(payload.get("completed") or {})

    if retriever is None:
        if diagnostic_mode == "source_bm25_diagnostic":
            source_settings = source_only_settings(evidence_config)
            retriever = SourceAwareCorpusRetriever(
                corpus_dir,
                source_candidate_k=source_settings["source_candidate_k"],
                passages_per_source=source_settings["passages_per_source"],
                candidate_k=source_settings["candidate_k"],
            )
        elif diagnostic_mode == "source_hybrid_v2":
            provider = build_evidence_provider(corpus_dir, evidence_settings)
            retriever = HybridCorpusRetriever(
                corpus_dir,
                encoder=provider.encoder,
                settings=evidence_settings,
            )
        else:
            retriever = FrozenCorpusRetriever(
                AveritecFrozenProvider(corpus_dir),
                top_k=int(evidence_config.get("single_top_k", 8)),
                max_chars=int(evidence_config.get("single_chars", 800)),
                max_per_source=(
                    int(evidence_config["max_per_source"])
                    if evidence_config.get("max_per_source") is not None
                    else None
                ),
            )

    for runtime_item, gold_item in aligned:
        if runtime_item.claim_id in completed:
            saved = completed[runtime_item.claim_id]
            observation = RetrievalObservation(
                claim_id=runtime_item.claim_id,
                candidate_evidence_ids=list(saved.get("candidate_evidence_ids", [])),
                candidate_source_urls=list(saved.get("candidate_source_urls", [])),
                final_evidence_ids=list(saved.get("final_evidence_ids", [])),
                final_source_urls=list(saved.get("final_source_urls", [])),
                elapsed_ms=int(saved.get("elapsed_ms", 0)),
                candidate_lexical_scores=list(saved.get("candidate_lexical_scores", [])),
                candidate_dense_scores=list(saved.get("candidate_dense_scores", [])),
                candidate_source_ranks=list(saved.get("candidate_source_ranks", [])),
            )
            completed[runtime_item.claim_id] = asdict(
                evaluate_observation(observation, gold_source_urls=_gold_sources(gold_item))
            )
            continue
        observation = await retriever.retrieve(runtime_item.claim_id, runtime_item.claim)
        item = evaluate_observation(observation, gold_source_urls=_gold_sources(gold_item))
        completed[item.claim_id] = asdict(item)
        atomic_write_json(
            progress_path,
            {"version": 1, "identity": identity, "completed": completed},
        )

    timed_records = [completed[item.claim_id] for item, _ in aligned]
    records = [
        {key: value for key, value in item.items() if key != "elapsed_ms"}
        for item in timed_records
    ]
    summary = {
        "claim_count": len(records),
        "source_hits": sum(bool(item["source_hit"]) for item in records),
        "final_hits": sum(bool(item["final_hit"]) for item in records),
        "candidate_count": sum(int(item["candidate_count"]) for item in records),
    }
    result = {
        "version": 1,
        "identity": identity,
        "ablation": ablation,
        "summary": summary,
        "records": records,
        "gates": evaluate_retrieval_gates(summary, records),
    }
    atomic_write_json(output_path, result)
    atomic_write_json(
        timing_path,
        {
            "identity": identity,
            "records": [
                {"claim_id": item["claim_id"], "elapsed_ms": item["elapsed_ms"]}
                for item in timed_records
            ],
        },
    )
    return result


__all__ = [
    "DiagnosticRetriever",
    "FrozenCorpusRetriever",
    "HybridCorpusRetriever",
    "SourceAwareCorpusRetriever",
    "RetrievalDiagnosticItem",
    "RetrievalObservation",
    "canonicalize_source_identity",
    "evaluate_observation",
    "evaluate_retrieval_gates",
    "resolve_diagnostic_mode",
    "run_diagnostic",
    "source_only_settings",
]
