# EvidenceRoute Stability v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make citation output deterministic and add an auditable single-verifier recovery for terminal multi failures so the next pilot can test stability and completion against the 17/20 and 90% gates.

**Architecture:** Keep retrieval and model verdict generation intact. Add one shared evidence-aware citation projection at the verifier/validator boundary, and add an explicit graph recovery node whose result keeps the original multi route while recording recovery metadata. Extend the immutable call-bound calculation for the new worst case before any paid execution.

**Tech Stack:** Python 3.11, LangGraph, Pydantic v2, SQLite run accounting, pytest, ruff.

---

### Task 1: Lock citation projection behavior

**Files:**
- Modify: `tests/test_citation_normalization.py`
- Modify: `tests/test_verification.py`
- Modify: `src/evidence_route/verification.py`

- [ ] Add tests that two model citation lists with the same claim-unit coverage but different order and duplicate source URLs produce the same projected citation IDs, while unknown citations are excluded.
- [ ] Add a test that a citation set missing a claim unit is not filled with a fabricated citation and remains eligible for `INSUFFICIENT_COVERAGE`.
- [ ] Run the focused tests and confirm the new expectations fail before changing production code.
- [ ] Implement a shared projection helper that accepts the model citations, frozen evidence order, and maximum count; normalizes URLs, deduplicates canonical source URLs, chooses one deterministic citation per claim unit, and keeps only known evidence IDs.
- [ ] Use the helper for single results and judge results; preserve citation text and source URL from the selected model citation.
- [ ] Run the focused tests and the existing verification/validation suites.

### Task 2: Add explicit multi-to-single recovery

**Files:**
- Modify: `src/evidence_route/contracts.py`
- Modify: `src/evidence_route/graph.py`
- Modify: `src/evidence_route/validation.py`
- Modify: `tests/test_graph.py`
- Modify: `tests/test_validation.py`

- [ ] Add tests for a multi validation failure followed by a successful single recovery and for a failed recovery; assert `initial_route`, `fallback_used`, status, errors, and no second recovery.
- [ ] Run those tests and confirm they fail because the recovery contract and graph state do not exist.
- [ ] Add a typed `fallback_used` result field and graph state marker with a default of false.
- [ ] Add a recovery node/edge after terminal multi validation failure. It invokes the existing single verifier once, keeps `initial_route=multi`, validates with recovery disabled, and preserves failure when the recovery is invalid.
- [ ] Ensure worker and recovery call IDs remain in the run summary and that no failure is silently promoted.
- [ ] Run graph, validation, execution, and artifact tests.

### Task 3: Extend deterministic accounting

**Files:**
- Modify: `src/evidence_route/evaluation/activity.py`
- Modify: `src/evidence_route/evaluation/runner.py`
- Modify: `src/evidence_route/evaluation/stability.py`
- Modify: `tests/test_campaign_runner.py`
- Modify: `tests/test_stability_diagnostics.py`

- [ ] Add tests showing the call profile and cost estimate include one recovery single call for each eligible multi item.
- [ ] Run the tests and confirm the expected bound is currently absent.
- [ ] Extend call-bound contracts and estimates with the explicit recovery allowance; make the budget preview use the same node limits as execution.
- [ ] Include `fallback_used` in repeat snapshots and route comparison fields without treating latency/token differences as instability.
- [ ] Run accounting, stability, and lifecycle tests.

### Task 4: Full offline verification and readiness gate

**Files:**
- Modify: `README.md`
- Modify: `docs/RESUME_PROJECT.md`

- [ ] Run `python -m pytest -m "not network and not live and not paid" -q`.
- [ ] Run `python -m ruff check src tests scripts`, `python -m compileall -q src tests scripts`, `python -m pip check`, and `git diff --check`.
- [ ] Run the CLI budget preview with the pilot configuration and verify the cap before constructing any paid transport.
- [ ] Update the project docs to state the v4 behavior and that a pilot is diagnostic until both gates pass.
- [ ] Start a 5-10 item paid pilot only when the offline checks and budget preview pass; otherwise continue offline debugging.

