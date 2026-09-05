# EvidenceRoute Quality Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve EvidenceRoute's frozen-corpus quality and stability while preserving strict evidence, status, artifact, and billing contracts.

**Architecture:** Keep runtime retrieval gold-blind and provider-neutral. Add deterministic source-aware retrieval/citation projection behind explicit configuration, then harden route/validation state transitions and provider accounting. Each task is gated by a failing regression test, a minimal implementation, and provider-free diagnostics.

**Tech Stack:** Python 3.11, Pydantic 2, LangGraph, rank-bm25, pytest, Ruff, SQLite artifacts.

---

### Task 1: Freeze the failure baseline

**Files:**
- Create: `tests/test_quality_recovery_baseline.py`
- Modify: `src/evidence_route/evaluation/stability_diagnostics.py`
- Test fixtures: `tests/fixtures/evaluation/`
- Docs: `docs/RETRIEVAL_DIAGNOSTIC_20260903.md`

- [ ] **Step 1: Write the failing test** asserting a diagnostic keeps `partial`, `failed`, and missing repeats in its denominator and emits category counts for evidence, route, validation, and provider drift.
- [ ] **Step 2: Run `pytest tests/test_quality_recovery_baseline.py -q` and verify the assertion fails because the diagnostic omits at least one required record/category.
- [ ] **Step 3: Implement the smallest deterministic diagnostic projection, preserving existing JSON field names and sorting by claim ID/repeat.
- [ ] **Step 4: Run the focused test, then `pytest tests/test_stability_diagnostics.py tests/test_quality_recovery_baseline.py -q`.
- [ ] **Step 5: Run the provider-free baseline command from the diagnostic document and save its hash and output under a new experiment identity.

### Task 2: Add source-aware retrieval mode

**Files:**
- Modify: `src/evidence_route/config.py`
- Modify: `src/evidence_route/providers/averitec.py`
- Modify: `src/evidence_route/retrieval.py`
- Create/modify: `tests/test_retrieval_v2.py`, `tests/test_averitec_provider.py`

- [ ] **Step 1: Write failing tests for explicit retrieval mode selection, bounded candidate count, per-source cap, stable score/ID/URL tie-breaks, and preservation of original evidence IDs/text.
- [ ] **Step 2: Run `pytest tests/test_retrieval_v2.py -q` and verify failure is caused by the missing mode/ranking behavior.
- [ ] **Step 3: Add validated retrieval settings with `sentence_bm25_v1` as the default and implement source-aware candidate ranking behind the opt-in mode.
- [ ] **Step 4: Run focused retrieval tests and all provider tests; confirm old default behavior remains unchanged.
- [ ] **Step 5: Add an offline calibration recall test that reads gold only in the diagnostic layer and fails closed when a selected local model receipt is absent or mismatched.

### Task 3: Make citation projection deterministic and evidence-bound

**Files:**
- Modify: `src/evidence_route/verification.py`
- Modify: `src/evidence_route/validation.py`
- Modify: `src/evidence_route/contracts.py`
- Create/modify: `tests/test_citation_normalization.py`, `tests/test_verification.py`, `tests/test_validation.py`

- [ ] **Step 1: Write failing tests for duplicate URL removal, stable citation ordering, complete claim-unit coverage, and rejection of citations whose evidence ID was not supplied to the verifier.
- [ ] **Step 2: Run `pytest tests/test_citation_normalization.py tests/test_verification.py tests/test_validation.py -q` and observe the expected citation/provenance failures.
- [ ] **Step 3: Implement a pure citation projection function using canonical URL, evidence ID, claim-unit IDs, and frozen candidate order as tie-breakers.
- [ ] **Step 4: Make validation reject fabricated or out-of-set citations while retaining existing fail-closed status/error semantics.
- [ ] **Step 5: Run the full offline suite and the retrieval diagnostic; compare candidate recall and citation drift against the frozen baseline.

### Task 4: Stabilize route and recovery state transitions

**Files:**
- Modify: `src/evidence_route/routing.py`
- Modify: `src/evidence_route/graph.py`
- Modify: `src/evidence_route/verification.py`
- Modify: `src/evidence_route/artifacts.py`
- Create/modify: `tests/test_routing.py`, `tests/test_graph.py`, `tests/test_hardening.py`

- [ ] **Step 1: Write failing repeat tests for deterministic ambiguous routing, one bounded multi-to-single recovery, retained initial route/original error, and no recovery after a recovery attempt.
- [ ] **Step 2: Run the focused tests and verify failures identify nondeterministic route or recovery metadata.
- [ ] **Step 3: Implement deterministic threshold evaluation and explicit recovery state fields without changing the public verdict enum.
- [ ] **Step 4: Persist recovery events and ensure infrastructure failures never become `Not Enough Evidence`.
- [ ] **Step 5: Run all stability diagnostics and require at least 17/20 consistent claims before moving to provider hardening.

### Task 5: Harden provider contracts and cost accounting

**Files:**
- Modify: `src/evidence_route/llm.py`
- Modify: `src/evidence_route/execution.py`
- Modify: `src/evidence_route/budget.py`
- Modify: `src/evidence_route/evaluation/runner.py`
- Create/modify: `tests/test_llm.py`, `tests/test_provider_smoke.py`, `tests/test_budget.py`, `tests/test_execution.py`

- [ ] **Step 1: Write failing tests for missing usage, malformed structured output, transient retry limits, response model drift, and billing uncertainty stop behavior.
- [ ] **Step 2: Run the focused provider/budget tests and verify each fails for the intended contract reason.
- [ ] **Step 3: Implement bounded idempotent retry handling and strict terminal states for incomplete usage, billing uncertainty, and model drift.
- [ ] **Step 4: Persist requested alias, raw response model ID, endpoint/price hashes, usage source, and retry attempts in existing artifacts/ledger fields.
- [ ] **Step 5: Run capability smoke, budget preview, and the complete offline test suite without network credentials.

### Task 6: Run gated evaluation and publish recovery report

**Files:**
- Modify: `src/evidence_route/cli.py`
- Modify: `src/evidence_route/evaluation/reporting.py`
- Create: `docs/EVIDENCE_QUALITY_RECOVERY_20260905.md`
- Tests: `tests/test_reporting.py`, `tests/test_cli.py`

- [ ] **Step 1: Write failing tests requiring new experiment identity, immutable parent artifact hashes, full-manifest denominator, and explicit gate results in the report.
- [ ] **Step 2: Run focused reporting/CLI tests and verify the new gate fields are absent or rejected.
- [ ] **Step 3: Implement report and CLI wiring for offline calibration/stability gates and isolated experiment directories.
- [ ] **Step 4: Run `pytest -q` and `ruff check src tests`.
- [ ] **Step 5: Run the offline calibration gate, stability diagnostic, and budget preview; only if all gates pass, prepare a new paid campaign command and record final metrics without modifying historical reports.

## Final verification

- [ ] `pytest -q`
- [ ] `ruff check src tests`
- [ ] Offline calibration recall gate passes.
- [ ] Strict stability is at least 17/20.
- [ ] Full-manifest macro-F1 is at least 0.392.
- [ ] Historical Gate A and prior experiment directories remain unchanged.
