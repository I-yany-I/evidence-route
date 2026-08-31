# Batched Evaluation Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe, resumable batch limits to calibration and Gate A evaluation while preserving frozen evaluation and accounting invariants.

**Architecture:** Add a typed `user_paused` status/reason pair to the existing activity model. The campaign runner and calibration collector stop only after a persisted complete item/case, while their existing resume paths clear the pause marker after freeze and ledger validation. Expose per-invocation `--max-cases` and `--max-items` controls with defaults of 4 and 10.

**Tech Stack:** Python 3.11, Pydantic, Typer, asyncio, pytest, existing SQLite run store and JSON journals.

---

### Task 1: Add resumable pause contracts

**Files:**
- Modify: `src/evidence_route/evaluation/activity.py`
- Modify: `src/evidence_route/evaluation/lifecycle.py`
- Test: `tests/test_activity.py`
- Test: `tests/test_activity_models.py`
- Test: `tests/test_lifecycle.py`

- [ ] **Step 1: Write failing tests**

Add assertions that `USER_PAUSED` maps to `PAUSED`, that a paused campaign derives `PAUSED`, and
that `resume_activity_phase` clears a `USER_PAUSED` marker.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/test_activity.py tests/test_activity_models.py tests/test_lifecycle.py -q`

Expected: failures because the new enum values and resumable transition are absent.

- [ ] **Step 3: Implement the smallest contract change**

Add `CampaignStatus.PAUSED`, `CampaignStopReason.USER_PAUSED`, map the reason to the status, include
the reason in precedence below safety stops, and allow `resume_activity_phase` to clear either
`PROCESS_INTERRUPTION` or `USER_PAUSED`. Keep `billing_uncertain` false for user pause.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_activity.py tests/test_activity_models.py tests/test_lifecycle.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evidence_route/evaluation/activity.py src/evidence_route/evaluation/lifecycle.py tests/test_activity.py tests/test_activity_models.py tests/test_lifecycle.py
git commit -m "feat: model resumable evaluation pauses"
```

### Task 2: Pause and resume the campaign runner by item count

**Files:**
- Modify: `src/evidence_route/evaluation/runner.py`
- Modify: `src/evidence_route/evaluation/production_evaluation.py`
- Test: `tests/test_campaign_runner.py`
- Test: `tests/test_production_services.py`

- [ ] **Step 1: Write failing tests**

Add an async runner test that executes a fixture plan with `max_items=2`, expects `PAUSED` and two
new executor calls, then resumes with `max_items=2` and expects only the remaining items to run.
Add service coverage that a `USER_PAUSED` campaign is accepted by `mode="resume"` and the activity
phase is resumed before execution.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/test_campaign_runner.py tests/test_production_services.py -q`

Expected: failures because runner/service signatures do not accept a batch limit or pause state.

- [ ] **Step 3: Implement bounded runner execution**

Validate `max_items` as a positive integer when supplied. Pass it through `run` and `resume` to
`_execute`. Count only items executed in the current invocation. After persisting a successful
artifact and state, if the count reaches the limit and pending work remains, set
`state.stop_reason = CampaignStopReason.USER_PAUSED`, `state.status = CampaignStatus.PAUSED`, write
the state, and return. Extend resume normalization to accept `USER_PAUSED`, clear it before
execution, and preserve all existing billing-uncertain checks.

- [ ] **Step 4: Thread the service parameter**

Read `max_items` from service kwargs, pass it into `runner.run`/`runner.resume`, and accept
`USER_PAUSED` in the resumable stop-reason set. Update activity phase recovery to use the existing
resume transition helper.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_campaign_runner.py tests/test_production_services.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/evidence_route/evaluation/runner.py src/evidence_route/evaluation/production_evaluation.py tests/test_campaign_runner.py tests/test_production_services.py
git commit -m "feat: pause production campaigns between items"
```

### Task 3: Pause and resume calibration by case count

**Files:**
- Modify: `src/evidence_route/evaluation/production_calibration.py`
- Modify: `src/evidence_route/cli.py`
- Test: `tests/test_calibration_orchestration.py`
- Test: `tests/test_production_services.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Add collector coverage for `max_cases=1`: the first case is complete, the activity is paused, and
the next invocation with `resume=True` completes the remaining cases without rerunning the first.
Add CLI assertions for defaults `max_cases=4` and `max_items=10` and forwarding of explicit values.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/test_calibration_orchestration.py tests/test_production_services.py tests/test_cli.py -q`

Expected: failures because the collector and CLI have no batch controls.

- [ ] **Step 3: Implement calibration batch boundaries**

Validate `max_cases` as a positive integer. Count completed cases in the current invocation. After
persisting a case and activity link, if the limit is reached and pending cases remain, transition
the calibration activity to `PAUSED` with `USER_PAUSED`, persist it, and return a payload containing
`status`, `completed_cases`, and `paused: true`. On resume, accept and clear `USER_PAUSED` before
building the transport. Do not call `_finalize` until all 32 case artifacts exist.

- [ ] **Step 4: Add CLI options and forwarding**

Add `--max-cases` to `calibrate` with default `4` and `--max-items` to `evaluate` with default
`10`; include each value in the corresponding service kwargs. Keep preview/replay behavior
unchanged and reject non-positive values through the service validation.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_calibration_orchestration.py tests/test_production_services.py tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/evidence_route/evaluation/production_calibration.py src/evidence_route/cli.py tests/test_calibration_orchestration.py tests/test_production_services.py tests/test_cli.py
git commit -m "feat: add bounded calibration collection"
```

### Task 4: Preserve reporting and run the full verification suite

**Files:**
- Modify: `README.md`
- Modify: `docs/RESUME_PROJECT.md`
- Test: `tests/test_reporting.py`
- Test: `tests/test_repository_contract.py`

- [ ] **Step 1: Write failing publication tests**

Add a paused activity fixture and assert report generation rejects it with the existing incomplete
activity error.

- [ ] **Step 2: Implement documentation and publication guard coverage**

Document the two commands and explain that a paused activity is resumable but not publishable.
Keep all existing claims about real provider quality, cost, latency, and model identity unchanged.

- [ ] **Step 3: Run focused publication tests**

Run: `python -m pytest tests/test_reporting.py tests/test_repository_contract.py -q`

Expected: PASS.

- [ ] **Step 4: Run the full suite and lint**

Run: `python -m pytest -q` and `python -m ruff check src tests`

Expected: all tests and lint checks pass.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/RESUME_PROJECT.md tests/test_reporting.py tests/test_repository_contract.py
git commit -m "docs: explain resumable evaluation batches"
```
