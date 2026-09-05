from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from evidence_route.contracts import Evidence
from evidence_route.evaluation.stability import canonicalize_citation_url

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?|[\u4e00-\u9fff]", re.I)
_RETRIEVAL_CONFIG = "evidence-route-regex-bm25-v1"
_REQUIRED_KEYS = {"evidence_id", "title", "source_url", "text", "snapshot_sha256"}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class _Record:
    evidence_id: str
    title: str
    source_url: str
    text: str
    snapshot_sha256: str


@dataclass(frozen=True)
class _Index:
    records: tuple[_Record, ...]
    bm25: BM25Okapi


class AveritecFrozenProvider:
    """Offline BM25 retrieval over one claim-specific frozen corpus at a time.

    The tokenizer is deliberately project-local and is not the official AVeriTeC
    baseline NLTK BM25 tokenizer.
    """

    retrieval_config_name = _RETRIEVAL_CONFIG

    def __init__(self, corpus_dir: Path, *, max_cached_claims: int = 1) -> None:
        if max_cached_claims < 1:
            raise ValueError("max_cached_claims must be positive")
        self.corpus_dir = Path(corpus_dir)
        self.max_cached_claims = max_cached_claims
        self._cache: OrderedDict[str, _Index] = OrderedDict()
        self.index_build_count = 0

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
        if max_per_source is not None and max_per_source < 1:
            raise ValueError("max_per_source must be positive when provided")
        index = self._get_index(claim_id)
        scores = index.bm25.get_scores(tokenize(query))
        ranked_candidates = sorted(
            zip(index.records, scores, strict=True),
            key=lambda item: (
                -float(item[1]),
                item[0].evidence_id,
                canonicalize_citation_url(item[0].source_url),
                item[0].snapshot_sha256,
            ),
        )
        ranked: list[tuple[_Record, float]] = []
        source_counts: dict[str, int] = {}
        for record, score in ranked_candidates:
            source_url = canonicalize_citation_url(record.source_url)
            count = source_counts.get(source_url, 0)
            if max_per_source is not None and count >= max_per_source:
                continue
            ranked.append((record, score))
            source_counts[source_url] = count + 1
            if len(ranked) == top_k:
                break
        return [
            Evidence(
                evidence_id=record.evidence_id,
                title=record.title,
                source_url=record.source_url,
                text=record.text[:max_chars],
                provider="averitec_frozen",
                snapshot_sha256=record.snapshot_sha256,
                ranking_score=float(score),
                char_start=0,
                char_end=min(len(record.text), max_chars),
            )
            for record, score in ranked
        ]

    def _get_index(self, claim_id: str) -> _Index:
        cached = self._cache.get(claim_id)
        if cached is not None:
            self._cache.move_to_end(claim_id)
            return cached
        path = self.corpus_dir / f"{claim_id}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        records = tuple(self._read_records(path))
        if not records:
            raise ValueError(f"claim corpus is empty: {path}")
        index = _Index(
            records=records,
            bm25=BM25Okapi(
                [tokenize(f"{record.title} {record.text}") for record in records]
            ),
        )
        self._cache[claim_id] = index
        self._cache.move_to_end(claim_id)
        self.index_build_count += 1
        while len(self._cache) > self.max_cached_claims:
            self._cache.popitem(last=False)
        return index

    @staticmethod
    def _read_records(path: Path) -> list[_Record]:
        records: list[_Record] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
                if not isinstance(row, dict) or set(row) != _REQUIRED_KEYS:
                    raise ValueError(
                        f"corpus row at {path}:{line_number} must contain only {_REQUIRED_KEYS}"
                    )
                if not all(isinstance(row[key], str) and row[key] for key in _REQUIRED_KEYS):
                    raise ValueError(f"corpus row at {path}:{line_number} has invalid values")
                digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                if row["snapshot_sha256"] != digest:
                    raise ValueError(f"snapshot hash mismatch at {path}:{line_number}")
                records.append(
                    _Record(
                        evidence_id=row["evidence_id"],
                        title=row["title"],
                        source_url=row["source_url"],
                        text=row["text"],
                        snapshot_sha256=digest,
                    )
                )
        return records
