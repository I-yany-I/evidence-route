# EvidenceRoute Quality Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce systematic false `Not Enough Evidence` outputs and prevent valid insufficient-evidence judgments from being discarded as partial or failed results.

**Architecture:** Keep runtime and scorer manifests isolated. Improve the hardened verification contract so attribution claims are treated as statement-verification tasks while substantive claims still require support for the underlying fact. Make coverage validation verdict-aware: affirmative verdicts require complete claim-unit coverage, while an insufficient-evidence verdict may identify only the evidence that was actually available.

**Tech Stack:** Python, Pydantic, pytest, existing EvidenceRoute verification graph.

---

### Task 1: Hardened verdict semantics

**Files:**
- Modify: `src/evidence_route/prompts.py`
- Test: `tests/test_prompts.py`

- [ ] Add a failing assertion that the hardened single/worker prompts distinguish attribution claims from substantive claims and require explicit conflict comparison.
- [ ] Run the focused prompt test and confirm it fails against the current prompt.
- [ ] Update the hardened prompt text and version/hash inputs without changing legacy prompts.
- [ ] Run the focused prompt and verification tests.

### Task 2: Verdict-aware evidence coverage

**Files:**
- Modify: `src/evidence_route/validation.py`
- Test: `tests/test_validation.py`

- [ ] Add a failing test showing a completed `Not Enough Evidence` result with incomplete claim-unit citations is accepted when its citations are valid.
- [ ] Add a failing test showing a completed `Supported` result with incomplete claim-unit citations still fails or escalates.
- [ ] Implement the smallest verdict-aware coverage rule and preserve unknown evidence, duplicate citation, and low-confidence checks.
- [ ] Run the focused validation and graph tests.

### Task 3: Repository verification

**Files:**
- No production changes expected.

- [ ] Run the full pytest suite with the repository Python environment.
- [ ] Run `ruff check src tests scripts`, `compileall`, `pip check`, and `git diff --check`.
- [ ] Review the diff and record that a paid re-evaluation is still required to measure actual benchmark improvement.
