# EvidenceRoute Stability V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in hardened execution mode that removes demonstrated routing/decomposition variance and applies deterministic verdict adjudication for a new isolated stability experiment.

**Architecture:** Keep the historical path as the default. Add small mode flags to the existing routing, decomposition, judging, and validation components; wire them from the experiment config only. Reuse the existing campaign runner, SQLite ledger, checkpoint recovery, artifact fingerprints, and parent identity checks.

**Tech Stack:** Python 3.11+, Pydantic, LangGraph, pytest, existing BM25 frozen provider and SQLite ledger.

---

### Task 1: Add Opt-In Hardened Configuration

**Files:**
- Modify: `src/evidence_route/config.py`
- Modify: `configs/default.yaml`
- Create: `configs/stability-v2.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing config test**

Add a test that loads a v2 config with environment credentials and asserts the hardened flags are
enabled, while the default config remains historical.

- [ ] **Step 2: Run the test and verify it fails**

Run `python -m pytest tests/test_config.py -q` and confirm the new fields are rejected or absent.

- [ ] **Step 3: Implement the minimal configuration fields**

Add `deterministic_ambiguous: bool = False`, `deterministic_decomposition: bool = False`,
`hardened_judge: bool = False`, and `adjudication: bool = False` to `RoutingSettings` or a small
`HardeningSettings` model. Keep defaults false and set all four true only in `configs/stability-v2.yaml`.

- [ ] **Step 4: Run focused config tests**

Run `python -m pytest tests/test_config.py -q`; expect all tests to pass.

- [ ] **Step 5: Commit**

Run `git add src/evidence_route/config.py configs/default.yaml configs/stability-v2.yaml tests/test_config.py` and commit with `feat: add opt-in stability v2 config`.

### Task 2: Make Ambiguous Routing and Decomposition Deterministic

**Files:**
- Modify: `src/evidence_route/routing.py`
- Modify: `src/evidence_route/verification.py`
- Modify: `src/evidence_route/evaluation/production.py`
- Modify: `src/evidence_route/cli.py`
- Test: `tests/test_routing.py`, `tests/test_verification.py`, `tests/test_production_orchestration.py`

- [ ] **Step 1: Write failing behavior tests**

Test that an ambiguous adaptive feature set never calls the router LLM when hardened mode is on and
returns `multi` with `source == "rule"`. Test that deterministic decomposition returns one task per
claim unit in order and never invokes its LLM.

- [ ] **Step 2: Run focused tests and verify the new tests fail**

Run `python -m pytest tests/test_routing.py tests/test_verification.py -q` and confirm the constructors
do not accept hardened mode or still invoke the LLM.

- [ ] **Step 3: Implement the minimal mode-aware behavior**

Pass the mode from `AppConfig` into `HybridRouter` and `ClaimDecomposer`. Ambiguous hardened routing
returns the conservative multi decision. Hardened decomposition creates `t0`, `t1`, and `t2` from
the first three claim units with exact unit text queries. Keep the legacy calls unchanged by default.

- [ ] **Step 4: Wire the flags through both production constructors**

Use the same settings when `GraphCampaignExecutor` and the single-run CLI build components. No
historical config should activate the hardened path.

- [ ] **Step 5: Run focused tests**

Run `python -m pytest tests/test_routing.py tests/test_verification.py tests/test_production_orchestration.py -q`.

- [ ] **Step 6: Commit**

Commit the routing and decomposition changes as `fix: stabilize ambiguous adaptive execution`.

### Task 3: Harden Judge Output and Deterministic Adjudication

**Files:**
- Modify: `src/evidence_route/prompts.py`
- Modify: `src/evidence_route/verification.py`
- Modify: `src/evidence_route/validation.py`
- Test: `tests/test_prompts.py`, `tests/test_verification.py`, `tests/test_validation.py`

- [ ] **Step 1: Write failing adjudication tests**

Cover unanimous candidates, `Supported` versus `Refuted`, and any disagreement involving `Not Enough
Evidence`. Assert that failed/incomplete candidates are excluded from adjudication and that citations
are selected deterministically by sorted evidence IDs.

- [ ] **Step 2: Run tests and verify the expected failures**

Run `python -m pytest tests/test_verification.py tests/test_validation.py -q` and confirm the helper
does not exist.

- [ ] **Step 3: Implement the pure adjudication helper**

Add a small function with a typed return value. It must never manufacture citations, promote partial
results, or convert a failed result into a valid one. Use the existing `VerificationResult` contract
and canonical URL normalization.

- [ ] **Step 4: Use the hardened judge prompt only when enabled**

Add a constructor flag to `VerdictJudge`, select `hardened_judge_messages` only in v2, and preserve
the existing prompt hash function for Gate A.

- [ ] **Step 5: Run focused tests and commit**

Run `python -m pytest tests/test_prompts.py tests/test_verification.py tests/test_validation.py -q` and commit with `fix: constrain stability v2 verdict boundary`.

### Task 4: Wire and Verify an Isolated V2 Experiment

**Files:**
- Modify: `src/evidence_route/evaluation/production.py`
- Modify: `src/evidence_route/evaluation/production_evaluation.py`
- Modify: `src/evidence_route/evaluation/experiment.py`
- Modify: `tests/test_experiment.py`, `tests/test_production_services.py`
- Modify: `README.md`, `docs/RESUME_PROJECT.md`

- [ ] **Step 1: Write preflight tests**

Assert v2 construction records a different config/prompt identity, creates a separate experiment
directory, and refuses to start when the parent report, manifest, pricing, or prompt hash changes.

- [ ] **Step 2: Run focused tests and verify failures**

Run `python -m pytest tests/test_experiment.py tests/test_production_services.py -q`.

- [ ] **Step 3: Wire v2 settings into the executor and experiment identity**

Ensure the selected v2 config and hardened prompt hash are persisted before transport construction.
Keep repeat-0 reuse disabled whenever route, decomposition, evidence construction, or judge prompt
behavior changes; schedule all three repeats for a clean comparison.

- [ ] **Step 4: Add the bounded execution command to documentation**

Document `--max-items 1`, `--resume`, the new activity ID, and the separate v2 artifact/report paths.
State that a failed v2 run is diagnostic only and does not replace Gate A.

- [ ] **Step 5: Run all offline verification**

Run `python -m pytest -m "not network and not live and not paid" -q`, `python -m ruff check src tests scripts`, `python -m pip check`, `python -m compileall -q src tests scripts`, and `git diff --check`.

- [ ] **Step 6: Commit**

Commit only implementation/tests/docs for v2 as `feat: add isolated stability v2 execution`.

### Task 5: Paid Readiness Gate, Then Optional Evaluation

**Files:**
- Create: `reports/evidence-route-stability-v2-20260901/` only after offline verification
- Modify: `docs/RESUME_PROJECT.md` with measured result

- [ ] **Step 1: Verify the parent baseline and cost preview**

Recompute all parent and repeat-0 hashes, check the ledger has no unresolved calls, and obtain the
exact v2 worst-case reservation before any paid call.

- [ ] **Step 2: Run a bounded one-item live smoke test**

Use a new activity identity and `--max-items 1`. Verify the persisted artifact, usage, cost, route,
and hardened flags before proceeding.

- [ ] **Step 3: Resume in bounded batches**

Run one or a few items per invocation with `--resume`. After each batch verify campaign state,
ledger completion, and no billing uncertainty. Keep the computer awake externally if needed; resume
must remain valid after process interruption.

- [ ] **Step 4: Run the full v2 stability cohort only if smoke diagnostics are sound**

Use the same 20 claims and three repeats in a separate directory. Do not overwrite Gate A artifacts.

- [ ] **Step 5: Publish only the measured outcome**

Generate before/after diagnostics. Promote v2 only if it reaches 17/20 and all accounting gates;
otherwise retain the original baseline and record v2 as engineering failure analysis.

