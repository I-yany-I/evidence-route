# EvidenceRoute Retrieval v2 Design

**Goal:** Raise frozen-corpus evidence recall before spending on another provider campaign, while
preserving scorer isolation, deterministic ordering, bounded memory, and auditable experiment identity.

## Context

The completed `evidence-route-quality-recovery-v1-20260902` campaign reached 100% completion and
15/20 stability but only 0.342 full-manifest macro-F1. The provider-free calibration diagnostic in
`docs/RETRIEVAL_DIAGNOSTIC_20260903.md` found that the current claim-query sentence BM25 retrieves a
gold source in its top eight for only 5 of 27 eligible claims (18.52%). More prompt tuning or transport
retries cannot recover evidence that never reaches the verifier.

## Approaches Considered

### Increase sentence BM25 top-k

This is the smallest change, but it sends more redundant and weak passages into already bounded model
contexts. It does not address source duplication or semantic mismatch and would raise token cost. It is
retained only as a diagnostic comparison.

### Source-aware lexical retrieval only

Grouping candidates by source and selecting the best passage per source should improve source coverage
without new dependencies. It is deterministic and cheap, but remains lexical and is unlikely to resolve
paraphrased or indirect evidence consistently. This becomes the first-stage ablation, not the final v2.

### Source-aware candidates plus local dense reranking

This is the selected design. Lexical retrieval narrows each claim corpus to a bounded, source-diverse
candidate set; a pinned local ONNX embedding model reranks only those passages. It adds model asset and
dependency management, but avoids provider calls and makes semantic ranking feasible on CPU. Remote
embedding APIs are rejected because they add cost, network variance, model-identity ambiguity, and a new
untracked billing surface.

## Runtime Architecture

### Versioned modes

`EvidenceSettings` gains an explicit retrieval mode. `sentence_bm25_v1` preserves current behavior
byte-for-byte. `source_hybrid_v2` is opt-in and changes the config hash, campaign fingerprint, and
retrieval identity. Existing Gate A and quality-recovery activities remain immutable and cannot resume
under v2.

### Stage 1: source-aware candidate generation

For the incoming query, score all sentence records with the existing project tokenizer and BM25. Group
records by canonical source URL and rank sources by a deterministic aggregate of their best sentence
scores. Keep at most 64 sources, then retain at most four highest-scoring passages per source, capped at
256 passages total. This prevents one repetitive source from consuming the candidate budget while
retaining passage-level provenance and original evidence IDs.

The implementation must not concatenate all source text into model prompts. Claim indexes remain
bounded by the existing one-claim LRU policy, and ties use score, evidence ID, canonical URL, and snapshot
hash in a fixed order.

### Stage 2: local dense reranking

A pinned English retrieval model is loaded from a prepared local asset directory and used through a CPU
ONNX runtime. The implementation uses FastEmbed with `BAAI/bge-small-en-v1.5`; the dependency lock and
model preparation receipt fix the exact package version, model revision, and required file hashes before
evaluation. Runtime evaluation must use local files only and must not download models implicitly.

Each candidate is encoded as title plus passage text. Query-to-passage cosine similarity is combined
with normalized lexical and source-rank signals using weights selected only on the train calibration
cohort. Final selection is deterministic, capped at two passages per source, and returns the existing
`Evidence` contract with no fabricated text or rewritten source identity.

The encoder is injected behind a narrow protocol so unit tests use a deterministic fake. If v2 is
selected and the model asset or receipt is missing, mismatched, or unreadable, retrieval fails closed;
it must not silently fall back to v1 during a benchmark campaign.

## Configuration And Freeze Identity

The v2 configuration records retrieval mode, source candidate count, passages per source, dense
candidate count, per-source final cap, model ID, model revision, receipt SHA-256, and scoring weights.
All numeric limits are positive and internally consistent. The campaign freeze binds the model receipt
alongside config, corpus, prompt, pricing, evaluator, and dependency hashes. No secret or absolute local
path enters a committed config or report.

The local model preparation command downloads into ignored `data/external/`, verifies the expected
revision and every required file hash, then writes a receipt. Evaluation accepts only a verified receipt
and an explicit local model root.

## Scorer Isolation And Diagnostics

Runtime provider modules continue to depend only on claim ID, query, and frozen corpus rows. Gold URLs,
labels, questions, and justifications remain in scorer-only evaluation code and are never imported by the
graph or provider.

A provider-free `diagnose-retrieval` command compares v1, source-only, and hybrid v2 on the 32 train
calibration claims. Its JSON report includes input hashes, retrieval/model identities, per-claim source
eligibility, source-candidate hit, final top-k hit, latency, and aggregate recall. The report is resumable
at claim boundaries because the corpus is large, and regenerated output must be byte-deterministic apart
from a separately stored timing section.

## Quality Gates

No paid provider call is allowed until all of these gates pass:

- Source-candidate recall at 64 is at least 22/27 (81.48%) on the frozen calibration diagnostic.
- Hybrid final top-8 gold-source recall is at least 14/27 (51.85%) and at least twice the 5/27 v1 rate.
- `train-2468` returns at least one passage from its frozen gold source in top eight.
- Two repeated diagnostics produce identical ranked evidence IDs for every claim.
- Existing v1 provider tests prove byte-compatible ranking and all non-network tests pass.
- Model receipt, dependency lock, configuration validation, budget preview, and report freeze audits pass.

Calibration gold is allowed only to select declared scoring weights and evaluate these gates. The final
weights and configuration are frozen before any dev result is inspected.

## Evaluation Sequence

After offline gates pass, create a new v2 activity and campaign identity. Run a paid 10-item smoke batch,
audit completion, usage, model identity, citations, and billing, then continue in batches of 10 only if no
new correctness or accounting defect appears. The full campaign remains 280 work items and is compared
against always-single, always-multi, Gate A, and quality-recovery using the complete manifest denominator.

The v2 result replaces the resume baseline only if it passes publication auditing and improves the
predeclared quality/cost trade-off. Completion and stability gains alone do not justify replacing a higher
macro-F1 baseline.

## Testing

Tests cover source grouping, duplicate URLs, per-source caps, lexical/dense score fusion, stable ties,
empty queries, malformed corpora, encoder failures, receipt tampering, missing local assets, v1
compatibility, cache eviction, scorer import isolation, resumable diagnostics, and freeze mismatch
rejection. Integration tests use small local corpora and a fake encoder; network/model-download tests are
explicitly marked and excluded from the default suite.

## Non-Goals

- No web search, live corpus refresh, MCP, UI, or Chinese benchmark in this phase.
- No gold-derived query expansion or runtime access to scorer manifests.
- No automatic provider campaign immediately after implementation.
- No mutation or reuse of prior activity artifacts under the new retrieval implementation.
