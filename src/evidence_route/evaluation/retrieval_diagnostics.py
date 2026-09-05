"""Offline, resumable retrieval diagnostics for frozen AVeriTeC corpora."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlsplit

import yaml

from evidence_route.artifacts import atomic_write_json
from evidence_route.evaluation.runtime_manifest import load_runtime_manifest, manifest_digest
from evidence_route.evaluation.scorer_manifest import align_runtime_and_gold, load_gold_manifest
from evidence_route.evaluation.stability import canonicalize_citation_url
from evidence_route.providers.averitec import AveritecFrozenProvider


@dataclass(frozen=True)
class RetrievalObservation:
    claim_id: str
    candidate_evidence_ids: list[str]
    candidate_source_urls: list[str]
    final_evidence_ids: list[str]
    final_source_urls: list[str]
    elapsed_ms: int


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


def evaluate_observation(
    observation: RetrievalObservation, *, gold_source_urls: list[str]
) -> RetrievalDiagnosticItem:
    gold = _canonical_sources(gold_source_urls)
    candidates = _canonical_sources(observation.candidate_source_urls)
    final = _canonical_sources(observation.final_source_urls)
    return RetrievalDiagnosticItem(
        claim_id=observation.claim_id,
        candidate_count=len(observation.candidate_evidence_ids),
        candidate_evidence_ids=list(observation.candidate_evidence_ids),
        final_evidence_ids=list(observation.final_evidence_ids),
        source_hit=bool(set(gold) & set(candidates)),
        final_hit=bool(set(gold) & set(final)),
        gold_source_urls=gold,
        candidate_source_urls=candidates,
        final_source_urls=final,
        elapsed_ms=observation.elapsed_ms,
    )


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
    receipt = corpus_dir.parent / "preparation_receipt.json"
    identity = {
        "runtime_manifest_sha256": manifest_digest(runtime_manifest),
        "gold_manifest_sha256": _sha256(gold_manifest),
        "config_sha256": config_hash,
        "corpus_receipt_sha256": _sha256(receipt) if receipt.is_file() else None,
        "ablation": ablation,
    }
    progress_path = Path(progress_path)
    completed: dict[str, dict[str, object]] = {}
    if progress_path.is_file():
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("retrieval diagnostic progress identity differs")
        completed = dict(payload.get("completed") or {})

    if retriever is None:
        evidence_config = dict(config.get("evidence") or {})
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

    records = [completed[item.claim_id] for item, _ in aligned]
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
    }
    atomic_write_json(output_path, result)
    atomic_write_json(
        timing_path,
        {
            "identity": identity,
            "records": [
                {"claim_id": item["claim_id"], "elapsed_ms": item["elapsed_ms"]}
                for item in records
            ],
        },
    )
    return result


__all__ = [
    "DiagnosticRetriever",
    "FrozenCorpusRetriever",
    "RetrievalDiagnosticItem",
    "RetrievalObservation",
    "canonicalize_source_identity",
    "evaluate_observation",
    "run_diagnostic",
]
