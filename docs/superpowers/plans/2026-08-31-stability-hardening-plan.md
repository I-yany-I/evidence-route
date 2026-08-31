# EvidenceRoute Stability Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provider-free claim-level stability diagnostics and deterministic hardening so a new isolated EvidenceRoute experiment can explain and, where justified, improve the Gate A baseline of 13/20 stable claims toward at least 17/20.

**Architecture:** Keep `evidence-route-gate-a-20260830-clean1` immutable. Add a small evaluation diagnostics module that converts the existing repeat-0/1/2 `RunArtifact` records into canonical snapshots and ordered drift categories, then expose it through the existing report workflow. Apply only evidence ordering, citation URL normalization, structured-output normalization, and validation-boundary fixes demonstrated by those records; run any changed configuration in a new experiment directory with parent/hash checks and the existing resumable SQLite activity runner.

**Tech Stack:** Python 3.11, Pydantic v2, `urllib.parse`, Typer, existing JSON artifact journals, existing SQLite run store, pytest, ruff.

---

## File Map

- Create: `src/evidence_route/evaluation/stability.py` for canonical repeat snapshots, deterministic classification, and summary models. It must not import a provider, transport, executor, or API configuration.
- Modify: `src/evidence_route/evaluation/reporting.py` to pass validated repeat artifacts and run-store accounting into the diagnostic builder and to write diagnostic JSON/Markdown beside the normal report.
- Modify: `src/evidence_route/cli.py` to expose an opt-in `--stability-diagnostics` report flag and an isolated experiment command that accepts a parent activity and explicit batch size.
- Modify: `src/evidence_route/providers/averitec.py` to impose a total ordering on equal-score evidence candidates before truncation and serialization.
- Modify: `src/evidence_route/verification.py` and `src/evidence_route/validation.py` to canonicalize citation URLs and normalize final structured verdict fields at the validation boundary.
- Modify: `src/evidence_route/graph.py` to pass the execution-owned evidence ID set into final validation instead of trusting a model result field.
- Modify: `src/evidence_route/prompts.py` to make insufficient/conflicting evidence outcomes explicit in the judge contract.
- Create: `src/evidence_route/evaluation/experiment.py` for persisted parent-baseline identity, input-hash checks, and experiment metadata; it delegates calls to the existing resumable production evaluation service.
- Modify: `src/evidence_route/evaluation/production_evaluation.py` only where needed to validate experiment metadata before constructing the transport and to persist the new configuration/prompt hashes.
- Test: `tests/test_stability_diagnostics.py`, `tests/test_experiment.py`, `tests/test_evidence_ordering.py`, `tests/test_citation_normalization.py`, `tests/test_validation.py`, `tests/test_prompts.py`, `tests/test_graph.py`, `tests/test_reporting.py`, `tests/test_cli.py`.
- Modify: `README.md` and `docs/RESUME_PROJECT.md` to document the diagnostic command, experiment identity, batch resume behavior, and the rule that a failed 85% target is reported honestly.

## Shared Interfaces

The implementation uses these exact public interfaces so later tasks remain consistent:

```python
# src/evidence_route/evaluation/stability.py
class StabilityCategory(StrEnum):
    INCOMPLETE_OR_FAILED = "incomplete_or_failed"
    EVIDENCE_OR_CITATION_DRIFT = "evidence_or_citation_drift"
    ROUTE_DRIFT = "route_drift"
    VALIDATION_OR_STATUS_DRIFT = "validation_or_status_drift"
    VERDICT_DRIFT = "verdict_drift"
    PROVIDER_VARIANCE = "provider_variance"

class RepeatSnapshot(StrictModel):
    repeat: int
    run_id: str | None
    artifact_sha256: str | None
    valid: bool
    status: str | None
    verdict: str | None
    route: str | None
    route_source: str | None
    escalated: bool | None
    worker_count: int | None
    error_codes: list[str]
    citation_urls: list[str]
    citations_valid: bool | None
    evidence_ids: list[str]
    evidence_coverage: list[str]
    selected_evidence_order: list[str]
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    fresh_latency_ms: int | None
    exact_cost_micro_cny: int | None
    transport_attempts: int | None
    cache_hits: int | None

class StabilityClaimDiagnostic(StrictModel):
    claim_id: str
    repeats: list[RepeatSnapshot]
    differing_fields: list[str]
    categories: list[StabilityCategory]
    primary_category: StabilityCategory | None

class StabilityDiagnosticSummary(StrictModel):
    activity_id: str
    campaign_id: str
    claim_count: int
    consistent_claim_count: int
    category_counts: dict[str, int]
    records: list[StabilityClaimDiagnostic]

def build_stability_diagnostics(
    *,
    activity_id: str,
    campaign_id: str,
    repeat_runs: Mapping[str, Mapping[int, RunArtifact | None]],
    run_store: SQLiteRunStore | None,
) -> StabilityDiagnosticSummary:
    raise NotImplementedError("implemented in Task 2")

def canonicalize_citation_url(value: str) -> str:
    raise NotImplementedError("implemented in Task 1")

def citation_urls(result: VerificationResult) -> list[str]:
    raise NotImplementedError("implemented in Task 1")

def citation_is_valid(result: VerificationResult) -> bool | None:
    raise NotImplementedError("implemented in Task 1")

def normalize_result(result: VerificationResult | None) -> dict[str, object]:
    raise NotImplementedError("implemented in Task 1")
```

`build_stability_diagnostics` sorts claims by `claim_id`, always emits repeats `0`, `1`, and `2`, and represents missing or malformed repeats with `valid=False`. It obtains `transport_attempts` from `run_store.summarize_run(run_id)` when a store is supplied; no provider or network object is constructed. A claim with no instability category has `categories=[]` and `primary_category=None`.

### Task 1: Add Canonical Stability Diagnostic Contracts

**Files:**
- Create: `src/evidence_route/evaluation/stability.py`
- Create: `tests/test_stability_diagnostics.py`

- [ ] **Step 1: Write failing model and normalization tests**

Add tests for URL normalization and structured result normalization:

```python
def test_canonicalize_citation_url_removes_fragment_default_port_and_trailing_slash() -> None:
    assert canonicalize_citation_url(
        "HTTPS://Example.COM:443/fact/?b=2&a=1#quote"
    ) == "https://example.com/fact?a=1&b=2"


def test_normalize_result_uses_enum_values_and_sorted_error_codes(result_factory) -> None:
    result = result_factory(errors=[" Z", "A", "A "])
    normalized = normalize_result(result)
    assert normalized["status"] == "completed"
    assert normalized["verdict"] == "Supported"
    assert normalized["errors"] == ["A", "Z"]
```

Add a fixture builder that creates valid `RunArtifact` instances with one citation, two evidence IDs, complete usage, and a deterministic latency block. It must mutate only the requested field so every diagnostic test exercises the real Pydantic schema.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/test_stability_diagnostics.py -q`

Expected: collection fails because `evidence_route.evaluation.stability` and its exported functions do not exist.

- [ ] **Step 3: Implement canonical helpers and strict models**

Implement `canonicalize_citation_url` with `urlsplit`/`urlunsplit`: lowercase scheme and hostname, remove default ports `80`/`443`, drop fragments, normalize an empty path to `/`, remove only trailing path slashes beyond `/`, and sort query pairs with `parse_qsl`/`urlencode`. Reject a non-absolute URL with `ValueError`.

Implement `citation_is_valid` using the existing report rule: completed/partial results with no citations are valid; otherwise every citation must reference an available evidence ID, use a unique evidence ID, have non-empty claim-unit IDs/question/answer/quote, and have a valid absolute URL. `citation_urls` returns sorted unique canonical URLs. `normalize_result` returns only comparison fields and normalizes status, verdict, route, route source (`"llm"`, `"rule"`, `"fallback"`, or `None`), escalation, sorted unique errors, sorted unique available evidence IDs, canonical citation URLs, and citation validity.

Define the Pydantic models from the Shared Interfaces block with `extra="forbid"`, `repeat` constrained to `0..2`, non-negative numeric fields where present, and `categories`/`differing_fields` sorted by the deterministic order defined in Task 2.

- [ ] **Step 4: Run the focused tests and lint**

Run: `python -m pytest tests/test_stability_diagnostics.py -q` and `python -m ruff check src/evidence_route/evaluation/stability.py tests/test_stability_diagnostics.py`.

Expected: PASS and no lint errors.

- [ ] **Step 5: Commit**

```bash
git add src/evidence_route/evaluation/stability.py tests/test_stability_diagnostics.py
git commit -m "feat: add canonical stability diagnostic contracts"
```

### Task 2: Implement Claim-Level Drift Classification

**Files:**
- Modify: `src/evidence_route/evaluation/stability.py`
- Modify: `tests/test_stability_diagnostics.py`

- [ ] **Step 1: Write classification tests**

Cover each primary rule with three repeats derived from the fixture in Task 1:

```python
    def test_missing_repeat_is_visible_and_primary_incomplete() -> None:
        summary = build_stability_diagnostics(
            activity_id="activity",
            campaign_id="campaign",
            repeat_runs={"claim": {0: artifact, 1: None, 2: artifact}},
            run_store=None,
        )
        record = summary.records[0]
        assert record.primary_category is StabilityCategory.INCOMPLETE_OR_FAILED
        assert record.repeats[1].valid is False

    def test_evidence_or_citation_change_is_classified_before_verdict() -> None:
        summary = build_stability_diagnostics(
            activity_id="activity",
            campaign_id="campaign",
            repeat_runs={"claim": {0: artifact, 1: artifact_with_new_evidence, 2: artifact}},
            run_store=None,
        )
        assert summary.records[0].primary_category is StabilityCategory.EVIDENCE_OR_CITATION_DRIFT

    def test_route_source_and_escalation_change_is_route_drift() -> None:
        summary = build_stability_diagnostics(
            activity_id="activity",
            campaign_id="campaign",
            repeat_runs={"claim": {0: artifact, 1: artifact_with_route_change, 2: artifact}},
            run_store=None,
        )
        assert summary.records[0].primary_category is StabilityCategory.ROUTE_DRIFT

    def test_same_verdict_with_status_or_validation_change_is_status_drift() -> None:
        summary = build_stability_diagnostics(
            activity_id="activity",
            campaign_id="campaign",
            repeat_runs={"claim": {0: artifact, 1: artifact_with_status_change, 2: artifact}},
            run_store=None,
        )
        assert summary.records[0].primary_category is StabilityCategory.VALIDATION_OR_STATUS_DRIFT

    def test_verdict_change_after_equal_deterministic_fields_is_verdict_drift() -> None:
        summary = build_stability_diagnostics(
            activity_id="activity",
            campaign_id="campaign",
            repeat_runs={"claim": {0: artifact, 1: artifact_with_verdict_change, 2: artifact}},
            run_store=None,
        )
        assert summary.records[0].primary_category is StabilityCategory.VERDICT_DRIFT

    def test_unexplained_verdict_change_is_provider_variance() -> None:
        summary = build_stability_diagnostics(
            activity_id="activity",
            campaign_id="campaign",
            repeat_runs={"claim": {0: artifact, 1: artifact_with_provider_change, 2: artifact}},
            run_store=None,
        )
        assert summary.records[0].primary_category is StabilityCategory.PROVIDER_VARIANCE

    def test_token_cost_latency_only_changes_have_no_instability_category() -> None:
        summary = build_stability_diagnostics(
            activity_id="activity",
            campaign_id="campaign",
            repeat_runs={"claim": {0: artifact, 1: artifact_with_quantitative_change, 2: artifact}},
            run_store=None,
        )
        assert summary.records[0].categories == []
```

The tests must assert both `categories` and `primary_category`, not only the aggregate rate. Include a malformed/missing artifact case and assert that it remains in `records` with all three repeat slots.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_stability_diagnostics.py -q`.

Expected: failures because `build_stability_diagnostics` and ordered classification are not implemented.

- [ ] **Step 3: Implement ordered comparison**

Build a `RepeatSnapshot` from each artifact. Use `artifact.result.available_evidence_ids` as `evidence_coverage` and preserve that list as `selected_evidence_order`; use the sorted citation evidence IDs as `evidence_ids`. A missing artifact produces a snapshot with `valid=False`, all nullable fields set to `None`, and no fabricated usage/cost values.

Compare only these deterministic groups:

```python
EVIDENCE_FIELDS = {"evidence_ids", "evidence_coverage", "selected_evidence_order", "citation_urls", "citations_valid"}
ROUTE_FIELDS = {"route", "route_source", "escalated", "worker_count"}
STATUS_FIELDS = {"status", "error_codes"}
QUANTITATIVE_FIELDS = {"input_tokens", "output_tokens", "total_tokens", "fresh_latency_ms", "exact_cost_micro_cny", "transport_attempts", "cache_hits"}
```

Apply the exact ordered categories from the design: missing/invalid/failed result first; evidence or citation drift next; route drift next; status/validation drift next; verdict drift next; provider variance only when the verdict/output differs and no deterministic group explains it. Do not add a category for `QUANTITATIVE_FIELDS`. A claim is stable only when all three valid final results have the same normalized verdict. Count each claim once under its primary category and count additional categories separately. Sort records by `claim_id`, categories by the fixed enum order, and `category_counts` by key when serialized.

- [ ] **Step 4: Add run-store accounting coverage**

Use a fake `summarize_run` object in the test to return `transport_attempts`, `cache_hit_count`, and exact usage for each run. Assert these values are copied into the correct repeat snapshot and that `run_store=None` still produces a valid provider-free diagnostic with nullable transport attempts.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest tests/test_stability_diagnostics.py -q`.

Expected: PASS.

```bash
git add src/evidence_route/evaluation/stability.py tests/test_stability_diagnostics.py
git commit -m "feat: classify claim-level stability drift"
```

### Task 3: Integrate Provider-Free Diagnostics into Reporting

**Files:**
- Modify: `src/evidence_route/evaluation/reporting.py`
- Modify: `src/evidence_route/cli.py`
- Modify: `tests/test_reporting.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write report integration tests**

Add a report fixture containing three stability repeats and assert that `build_report_bundle` remains unchanged by default. Add an opt-in test that calls the report service with `stability_diagnostics=True`, writes `stability_diagnostics.json` and `stability_diagnostics.md`, and verifies the JSON digest is deterministic across two writes. Monkeypatch `StructuredLLM`, transport construction, and official evaluators to fail if called by the diagnostic path; the diagnostic must still pass.

Add CLI coverage for `--stability-diagnostics` forwarding and for the default being `False`.

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_reporting.py tests/test_cli.py -q`.

Expected: failures because the report service, bundle writer, and CLI option do not accept the diagnostic flag or output files.

- [ ] **Step 3: Wire the existing validated artifacts into the builder**

After `_load_models` and `_load_artifacts` have verified the activity, plan, state, artifact fingerprints, and state/artifact identity, group `plan.stability_repeat_zero_links` with repeat-0 dev artifacts and repeat-1/2 stability artifacts. Pass the resulting mapping to `build_stability_diagnostics` together with the activity and campaign IDs and the already-open `SQLiteRunStore`. Do not re-read raw JSON outside the existing `_inside` path checks.

Extend `ReportBundle` with `stability_diagnostics: StabilityDiagnosticSummary | None`. When present, `write()` emits canonical `stability_diagnostics.json` and a Markdown table containing claim ID, primary category, categories, and the three verdict/status values. Publication still requires the existing strict gate; a diagnostic may be generated for an incomplete activity but must label missing repeats and must never make that activity publishable.

- [ ] **Step 4: Add the report flag and exact output behavior**

Add `stability_diagnostics: bool = False` to `CliServices.report` and the Typer `report` command as `--stability-diagnostics`. When enabled, write the two files in the requested `--output-dir`; when disabled, omit or remove stale diagnostic files so a report directory cannot claim a newly generated diagnostic. Return `stability_diagnostics` in the JSON service payload as the file path or `None`.

- [ ] **Step 5: Run focused reporting tests**

Run: `python -m pytest tests/test_reporting.py tests/test_cli.py -q`.

Expected: PASS, with no network markers required.

- [ ] **Step 6: Commit**

```bash
git add src/evidence_route/evaluation/reporting.py src/evidence_route/cli.py tests/test_reporting.py tests/test_cli.py
git commit -m "feat: expose provider-free stability diagnostics"
```

### Task 4: Harden Deterministic Evidence and Verdict Boundaries

**Files:**
- Modify: `src/evidence_route/providers/averitec.py`
- Modify: `src/evidence_route/verification.py`
- Modify: `src/evidence_route/validation.py`
- Modify: `src/evidence_route/prompts.py` to define a separately versioned hardened judge contract without changing the frozen Gate A prompt hash.
- Modify: `src/evidence_route/graph.py`
- Modify: `tests/test_evidence_ordering.py`
- Modify: `tests/test_citation_normalization.py`
- Modify: `tests/test_validation.py`
- Test: `tests/test_prompts.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: Write regression tests from diagnostic causes**

Add an AVeriTeC provider test with equal ranking scores and different evidence IDs/URLs; call search twice with the input records in opposite order, use `top_k=1`, and assert identical evidence IDs, order, and serialized evidence payload. Add a final tie test with equal score/evidence ID/canonical URL and different snapshot hashes. Add citation tests for equivalent URL spellings and assert one canonical URL set and identical validity. Add validation tests for completed, partial, insufficient, and conflicting verdicts, asserting that normalized status transitions are explicit and never silently changed by missing coverage. Add a graph test proving final validation receives evidence IDs owned by execution state/provider rather than trusting a result-supplied available ID list.

- [ ] **Step 2: Run focused tests and verify the new tests fail**

Run: `python -m pytest tests/test_evidence_ordering.py tests/test_citation_normalization.py tests/test_validation.py tests/test_prompts.py tests/test_graph.py -q`.

Expected: at least the equal-score ordering and URL canonicalization tests fail before the implementation changes.

- [ ] **Step 3: Make evidence ordering total and stable**

In `AveriteCFrozenProvider.search`, sort candidates by the tuple `(-ranking_score, evidence_id, canonicalize_citation_url(source_url), snapshot_sha256)` before applying `top_k` and character limits. Preserve the existing ranking semantics and evidence schema. If diagnostic records show a different deterministic tie, use only fields already present in the frozen evidence record and add the corresponding regression assertion.

- [ ] **Step 4: Canonicalize citations at construction and comparison**

Use `canonicalize_citation_url` when creating `Citation` values in `verification.py` and when checking citation validity in `validation.py`/reporting. Compare canonical URL strings, but preserve the original citation text and quote. Deduplicate citations by evidence ID after canonicalization and emit them in evidence ID order where the contract permits. Keep the existing `prompt_hash()` and default validator behavior unchanged so historical Gate A replay remains compatible; the hardened prompt and opt-in normalization apply only to the new experiment.

- [ ] **Step 5: Normalize structured verdict output at the validation boundary**

Add one pure helper in `validation.py` that strips surrounding whitespace from rationale/errors, converts enum-like verdict strings through `Verdict`, sorts unique available evidence IDs, and sets `partial` only when the explicit coverage rule is met. The rule is: a completed result requires a verdict and complete usage; a partial result may carry a verdict but must carry an explicit validation error or incomplete coverage; a failed result carries no verdict. Insufficient and conflicting verdicts remain distinct enum values and are never collapsed. Expose normalization as an opt-in validator setting that defaults to the historical Gate A behavior; Task 5 enables it only for the new isolated experiment before persistence.

- [ ] **Step 6: Run focused tests and commit**

Run: `python -m pytest tests/test_evidence_ordering.py tests/test_citation_normalization.py tests/test_validation.py tests/test_stability_diagnostics.py -q`.

Expected: PASS.

```bash
git add src/evidence_route/providers/averitec.py src/evidence_route/verification.py src/evidence_route/validation.py tests/test_evidence_ordering.py tests/test_citation_normalization.py tests/test_validation.py
git commit -m "fix: harden deterministic evidence and verdict normalization"
```

### Task 5: Add Isolated Experiment Identity and Parent Checks

**Files:**
- Create: `src/evidence_route/evaluation/experiment.py`
- Modify: `src/evidence_route/evaluation/production_evaluation.py`
- Modify: `src/evidence_route/cli.py`
- Create: `tests/test_experiment.py`

- [ ] **Step 1: Write parent identity tests**

Test that a new experiment metadata file stores `parent_activity_id`, `parent_report_sha256`, `parent_manifest_sha256`, `parent_pricing_sha256`, `parent_config_sha256`, `parent_prompt_sha256`, `stability_manifest_sha256`, `repeat_schedule`, and provider identity/verification. Test that a changed parent report, manifest, pricing, config, or prompt raises `FreezeMismatch` before the transport factory is called. Test that the experiment directory is distinct from `reports/evidence-route-gate-a-20260830-clean1` and that repeat-0 artifacts are reused only when their artifact hashes match the recorded parent links.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_experiment.py -q`.

Expected: collection or assertion failures because the experiment metadata and validation functions do not exist.

- [ ] **Step 3: Implement immutable experiment metadata**

Define `StabilityExperimentIdentity(StrictModel)` with the fields listed above plus `experiment_id`, `activity_id`, `campaign_id`, `config_sha256`, `prompt_sha256`, and `created_at`. Implement:

```python
def create_experiment_identity(
    *,
    experiment_id: str,
    activity_id: str,
    campaign_id: str,
    parent_activity_id: str,
    parent_report: Path,
    parent_manifest: Path,
    pricing: Path,
    config: Path,
    prompt: Path,
    stability_manifest: Path,
    repeat_schedule: Mapping[str, list[int]],
    requested_alias: str,
    response_model_id: str,
    identity_verified: bool,
    output_dir: Path,
) -> StabilityExperimentIdentity:
    raise NotImplementedError("implemented in Task 5")

def validate_parent_baseline(
    identity: StabilityExperimentIdentity,
    *,
    parent_report: Path,
    parent_manifest: Path,
    pricing: Path,
    config: Path,
    prompt: Path,
) -> None:
    raise NotImplementedError("implemented in Task 5")

def verify_repeat_zero_reuse(
    identity: StabilityExperimentIdentity,
    baseline_artifacts: Mapping[str, Path],
) -> None:
    raise NotImplementedError("implemented in Task 5")
```

Use SHA-256 of bytes for files, stable JSON for the repeat schedule, and the existing `artifact_fingerprint` for artifact content. Reject an experiment when any recorded hash differs, when parent activity/campaign IDs differ, when the stability manifest is not the same 20-claim manifest, or when a target path resolves inside the published Gate A report directory. Persist metadata atomically before any paid call.

- [ ] **Step 4: Integrate checks before transport construction**

Add an explicit experiment mode to the production evaluation service/CLI that requires `--parent-activity`, `--parent-report`, and `--experiment-dir`. Validate parent identity, frozen config/prompt/manifest/pricing, and repeat-0 artifact links before invoking `_transport` or constructing a graph executor. Reuse the existing `max_items` and `--resume` behavior; persist repeats 1 and 2 into the new activity only. A prompt/routing/evidence implementation hash change disables repeat-0 reuse and schedules all three repeats.

- [ ] **Step 5: Test bounded resume and ledger invariants**

Extend experiment tests with `max_items=1`, interruption after a persisted item, and resume. Assert completed artifacts are not executed twice, the run-store has no duplicate logical call, incomplete usage and `billing_uncertain` stop the experiment, and a batch pause is `user_paused` rather than a billing or process stop.

- [ ] **Step 6: Run focused tests and commit**

Run: `python -m pytest tests/test_experiment.py tests/test_campaign_runner.py tests/test_production_services.py -q`.

Expected: PASS.

```bash
git add src/evidence_route/evaluation/experiment.py src/evidence_route/evaluation/production_evaluation.py src/evidence_route/cli.py tests/test_experiment.py
git commit -m "feat: isolate stability experiments with parent checks"
```

### Task 6: Produce Before/After Diagnostics and Documentation

**Files:**
- Modify: `src/evidence_route/evaluation/reporting.py`
- Modify: `tests/test_reporting.py`
- Modify: `README.md`
- Modify: `docs/RESUME_PROJECT.md`

- [ ] **Step 1: Write before/after report tests**

Create two local diagnostic summaries from the same 20-claim fixture, one representing the Gate A baseline and one representing the hardening experiment. Assert the report stores `baseline_activity_id`, `experiment_activity_id`, both diagnostic hashes, category counts before and after, `improved_claims`, `regressed_claims`, stable consistency counts, completion rate, Wilson interval, model ID, and `identity_verified=false`. Assert that a result below 17/20 remains publishable only as an engineering failure analysis and never overwrites the original Gate A report.

- [ ] **Step 2: Implement comparison output**

Add `build_stability_comparison(baseline, experiment)` in `stability.py`. Match records by claim ID, compute category count deltas, improved/regressed/unchanged claim IDs, and exact consistency/completion values. Serialize with sorted keys and render a concise table with one row per category. Refuse to compare different claim sets, repeat schedules, or parent identities.

- [ ] **Step 3: Add final verification guards**

Before an experiment report is marked successful, verify: consistency `>= 17/20`, completion `>= 90%`, every call has complete usage, no `billing_uncertain`, no unclosed reservation, all persisted hashes recompute exactly, and the parent Gate A directory has not changed. If any condition fails, write the diagnostics and failure analysis with the measured values and leave the Gate A artifacts untouched.

- [ ] **Step 4: Update project documentation**

Document the local command shape:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m evidence_route.cli report --activity-dir <experiment-activity> --output-dir <experiment-report> --stability-diagnostics
python -m evidence_route.cli evaluate --resume --activity-id <same-activity-id> --activity-dir <same-activity-dir> --config configs/default.yaml --pricing configs/pricing.local.yaml --manifest data/manifests/averitec_dev_runtime.json --stability-manifest data/manifests/averitec_stability_runtime.json --max-items 10
```

State that the published `evidence-route-gate-a-20260830-clean1` activity remains the baseline, diagnostics are provider-free, batches pause only after a complete persisted item, and resume reuses closed artifacts without repeating paid calls. Preserve the existing honest limitations: 20-claim stability sample, relay-reported unverified model identity, and balanced AVeriTeC subset scope.

- [ ] **Step 5: Run focused report tests and commit**

Run: `python -m pytest tests/test_reporting.py tests/test_stability_diagnostics.py tests/test_experiment.py -q`.

Expected: PASS.

```bash
git add src/evidence_route/evaluation/stability.py src/evidence_route/evaluation/reporting.py tests/test_reporting.py README.md docs/RESUME_PROJECT.md
git commit -m "docs: record stability before-after analysis"
```

### Task 7: Full Offline Verification and Experiment Readiness Gate

**Files:**
- Modify: `tests/test_repository_contract.py` only if the new files need contract allowlisting.
- Modify: `docs/RESUME_PROJECT.md` with the final measured readiness status.

- [ ] **Step 1: Run all focused regression suites**

Run:

```powershell
python -m pytest tests/test_stability_diagnostics.py tests/test_evidence_ordering.py tests/test_citation_normalization.py tests/test_experiment.py tests/test_reporting.py tests/test_validation.py tests/test_campaign_runner.py tests/test_production_services.py tests/test_cli.py -q
```

Expected: PASS with no paid or live markers selected.

- [ ] **Step 2: Run the full offline suite and static checks**

Run:

```powershell
python -m pytest -m "not network and not live and not paid" -q
python -m ruff check src tests scripts
python -m pip check
git diff --check
```

Expected: all commands exit `0`. Any existing dirty user files remain untouched and are not included in commits for this plan.

- [ ] **Step 3: Verify artifact and parent immutability**

Recompute the baseline report, Gate A activity, campaign, and artifact hashes recorded before implementation. Assert the hashes match the saved baseline values and that no file under `reports/evidence-route-gate-a-20260830-clean1` or the formal activity directory changed.

- [ ] **Step 4: Commit only the implementation and tests**

```bash
git add src/evidence_route/evaluation/stability.py src/evidence_route/evaluation/reporting.py src/evidence_route/evaluation/experiment.py src/evidence_route/providers/averitec.py src/evidence_route/verification.py src/evidence_route/validation.py src/evidence_route/evaluation/production_evaluation.py src/evidence_route/cli.py tests README.md docs/RESUME_PROJECT.md
git commit -m "feat: harden and diagnose stability experiments"
```

- [ ] **Step 5: Stop before paid execution and request execution mode**

After offline verification, report the exact test/lint results and the expected experiment cost from the existing pricing preview. Do not launch the provider until the user chooses one of the execution modes below and explicitly confirms the new experiment identity.

## Self-Review Checklist

### Spec Traceability

| Design requirement | Plan coverage |
|---|---|
| Claim-level diagnostics with repeat 0/1/2 hashes, statuses, routes, citations, evidence, usage, cost, latency, attempts, and cache hits | Tasks 1-3, `RepeatSnapshot`, and report integration tests |
| Ordered categories including incomplete, evidence/citation, route, status/validation, verdict, and provider variance | Task 2 classification tests and fixed comparison groups |
| Deterministic evidence ordering, citation canonicalization, structured output normalization, explicit partial transitions, and judge constraint coverage | Task 4 regression tests and boundary changes |
| Isolated experiment identity, parent report/config/manifest/pricing/prompt hashes, same manifest/repeats, and repeat-0 reuse rules | Task 5 metadata and pre-transport checks |
| Acceptance threshold 17/20, completion at least 90%, no uncertain billing, reproducible hashes, before/after counts, and honest failure analysis | Tasks 6-7 verification guards and report tests |
| Unit, fixture, provider-free report, experiment-isolation, lifecycle, accounting, and offline regression testing | Tasks 1-7 test files and commands |
| Risks from a 20-claim sample, relay model variance, wrong identity, and interruption recovery | Tasks 2, 5, 6, and 7 preserve per-claim records, provider variance, parent checks, and resumable accounting |

The traceability table covers every section of `2026-08-31-stability-hardening-design.md`; no requirement is deferred to an unspecified task.

- Spec coverage: Tasks 1-2 cover claim-level records, all repeat fields, deterministic category precedence, and quantitative-field exclusion; Task 3 covers provider-free report integration; Task 4 covers all four deterministic hardening points and focused regressions; Task 5 covers isolated activity, parent/hash checks, repeat-0 reuse, and resumable ledger behavior; Task 6 covers before/after summaries, Wilson bounds, acceptance thresholds, and failure analysis; Task 7 covers offline verification and Gate A immutability.
- Placeholder scan: every step contains concrete paths, algorithms, assertions, expected command results, and defined Task references; no step delegates a required behavior to an unspecified future action.
- Type consistency: `StabilityCategory`, `RepeatSnapshot`, `StabilityClaimDiagnostic`, `StabilityDiagnosticSummary`, and `build_stability_diagnostics` are used with the same names and fields throughout; the existing `max_items`/`--resume` interface is reused consistently.
- Safety: no task edits the published Gate A artifact directory, no diagnostic path creates a transport, and no acceptance result is rounded or selectively scoped.

Plan complete and saved to `docs/superpowers/plans/2026-08-31-stability-hardening-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task and review between tasks for fast iteration.
2. **Inline Execution** - execute the tasks in this session using executing-plans with checkpoints.

Which approach?
