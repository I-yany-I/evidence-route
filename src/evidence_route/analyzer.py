from __future__ import annotations

import re

from evidence_route.contracts import ClaimFeatures, ClaimUnit, Evidence

_BOUNDARY_RE = re.compile(
    r"(?:[;；。]|\bbut\b|\band\b|\bwhile\b|\bwhereas\b|但是|但|而且|并且)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_TIME_RE = re.compile(r"\b(?:19|20)\d{2}\b|\d{4}年|今年|去年|本月|today|yesterday", re.I)
_COMPARISON_RE = re.compile(r"more than|less than|higher|lower|超过|低于|高于|相比", re.I)
_CAUSAL_RE = re.compile(r"because|caused|therefore|因为|导致|因此", re.I)
_CONTRAST_RE = re.compile(r"\bbut\b|whereas|however|但是|然而|并非", re.I)
_NEGATION_RE = re.compile(r"\bnot\b|\bnever\b|没有|并非|从未", re.I)


def analyze_claim(claim: str, probe_evidence: list[Evidence]) -> ClaimFeatures:
    """Extract stable lexical features used by routing, without an LLM call."""
    parts = [part.strip(" ,，") for part in _BOUNDARY_RE.split(claim) if part.strip(" ,，")]
    units = [
        ClaimUnit(unit_id=f"u{index}", text=text)
        for index, text in enumerate(parts or [claim])
    ]
    scores = [item.ranking_score for item in probe_evidence]
    all_numbers = [set(_NUMBER_RE.findall(item.text)) for item in probe_evidence]
    conflicting_numbers = len({number for group in all_numbers for number in group}) > 1
    negated = any(_NEGATION_RE.search(item.text) for item in probe_evidence)
    return ClaimFeatures(
        claim_units=units,
        atomic_clause_count=len(units),
        entity_count=len(
            re.findall(r"\b[A-Z][A-Za-z0-9-]+\b|[\u4e00-\u9fff]{2,8}(?:公司|大学|政府|协会)", claim)
        ),
        numeric_count=len(_NUMBER_RE.findall(claim)),
        time_scope_count=len(_TIME_RE.findall(claim)),
        has_comparison=bool(_COMPARISON_RE.search(claim)),
        has_causal=bool(_CAUSAL_RE.search(claim)),
        has_contrast=bool(_CONTRAST_RE.search(claim)),
        probe_source_count=len({str(item.source_url) for item in probe_evidence}),
        probe_score_spread=(max(scores) - min(scores)) if scores else 0.0,
        probe_conflict_hint=conflicting_numbers and negated,
    )
