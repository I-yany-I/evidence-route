# EvidenceRoute Stability Hardening Design

## Goal

Improve EvidenceRoute's repeated-run consistency from the Gate A baseline of 13/20 (65.0%)
toward the project engineering target of at least 17/20 (85%), while preserving the published
Gate A activity as an immutable baseline.

## Context

The completed Gate A activity is `evidence-route-gate-a-20260830-clean1`. It evaluates 20
adaptive stability claims three times: the dev adaptive run is repeat 0 and two additional
stability runs are repeats 1 and 2. The current report exposes only an aggregate consistency
rate, so the next iteration needs claim-level evidence before changing prompts or routing.

The provider is an OpenAI-compatible relay and reports an unverified model ID. Provider output
variance is therefore a possible cause, but it must be separated from deterministic differences
in evidence ordering, citations, routing, validation, and status handling.

## Non-Goals

- Do not rewrite the published Gate A artifacts or change their hashes.
- Do not change the Gate A cohort, gold manifest, published configuration, or historical metrics.
- Do not claim that a stability improvement is statistically significant from 20 claims.
- Do not add MCP, Web UI, or new benchmark cohorts in this iteration.
- Do not solve instability by returning cached predictions for fresh paid runs.

## Design

### 1. Claim-Level Stability Diagnostics

Add a deterministic diagnostic builder in the evaluation reporting layer. It reads the existing
activity, campaign artifacts, and run-store without invoking the provider. For every stability
claim with repeat 0/1/2 artifacts, it emits a normalized comparison record containing:

- claim ID and the three artifact hashes;
- status, verdict, route, route source, escalation flag, and error code for each repeat;
- canonical citation URL sets and citation validity results;
- evidence item identities, evidence coverage, and selected evidence ordering;
- input/output/total tokens, fresh latency, exact cost, transport attempts, and cache hits;
- a list of fields that differ between repeats;
- one or more deterministic diagnostic categories.

Records with missing or malformed repeats remain visible and are classified as incomplete rather
than silently excluded. The aggregate output includes counts by category and a stable sort by
claim ID.

### 2. Classification Rules

Classification is deterministic and ordered so one claim can have a primary category and
additional contributing categories:

1. `incomplete_or_failed`: a repeat is missing, failed, or has no valid final result;
2. `evidence_or_citation_drift`: evidence identities, ordering, coverage, or canonical citation
   sets differ while the claim path is otherwise comparable;
3. `route_drift`: route, route source, escalation, or worker count differs;
4. `validation_or_status_drift`: verdict is unchanged but status, validation, or error metadata
   differs;
5. `verdict_drift`: the final verdict differs after the preceding deterministic fields are
   compared;
6. `provider_variance`: no deterministic difference explains the changed output, leaving model
   response variance as the residual category.

The report must distinguish a verdict difference from a harmless token or latency difference.
Token, cost, and latency changes are quantitative fields and are not instability categories by
themselves.

### 3. Determinism Hardening

Fix only deterministic causes demonstrated by the diagnostic records:

- sort evidence candidates with a total key before truncation and prompt construction;
- canonicalize citation URLs before validation and comparison;
- normalize structured verdict fields before comparison;
- make coverage and partial-status transitions explicit at the validation boundary;
- constrain judge output to resolve insufficient and conflicting evidence consistently.

Each fix must have a focused regression test. Prompt or routing changes must be isolated in a
new experiment configuration and must not alter the frozen Gate A configuration.

### 4. Isolated Experiment

Create a new stability experiment identity derived from the hardening change. It must persist:

- the parent Gate A activity ID and baseline report hash;
- the new configuration and prompt hashes;
- the same 20-claim stability manifest and same repeat schedule;
- the same pricing and ledger invariants;
- before/after diagnostic summaries;
- exact provider-reported model identity and its verification status.

The experiment may reuse immutable baseline artifacts for repeat 0. Only repeats 1 and 2 are
fresh calls when the implementation and frozen inputs are unchanged. If prompt, routing, or
evidence construction changes, all three repeats are rerun for a clean comparison. No artifact
from the experiment may be copied into the published Gate A directory.

### 5. Acceptance Criteria

The hardening experiment is successful only when all of the following are recorded:

- stability consistency is at least 17/20 (85%);
- completion rate is at least 90% and all missing/failed repeats are reported;
- no call has `billing_uncertain`, incomplete usage, or unclosed reservation;
- all artifact, ledger, configuration, manifest, and prompt hashes are reproducible;
- the before/after report identifies which categories changed and how many claims improved;
- focused and full offline tests pass, and lint/dependency checks pass.

If the 85% target is not reached, publish the diagnostic as an engineering failure analysis and
keep the original Gate A result unchanged. The result must not be rounded or selectively scoped
to hide failed or partial repeats.

## Interfaces

The diagnostic interface should be callable both from Python tests and the existing `report`
workflow. Prefer a report option that reads artifacts locally and writes a JSON diagnostic plus
a concise Markdown table. It must not require API credentials or construct a network transport.

The experiment runner should use the existing resumable activity and SQLite ledger interfaces,
with a bounded per-invocation batch size. It must reject an experiment if its parent activity,
manifest, pricing, or configuration identity does not match the persisted baseline.

## Testing

- Unit tests for canonical comparison and each classification rule.
- Fixture tests proving missing, partial, failed, and verdict-drift repeats remain visible.
- Regression tests proving evidence ordering and citation normalization are stable.
- Report tests proving diagnostic generation is deterministic and provider-free.
- Experiment tests proving baseline artifacts are reused only when hashes match and that a
  mismatched parent is rejected before any paid call.
- Existing Gate A lifecycle, accounting, publication, and offline tests must remain green.

## Risks and Mitigations

- A 20-claim sample can overstate improvement: retain per-claim records and report Wilson bounds.
- Relay model variance may dominate deterministic fixes: keep `provider_variance` explicit and do
  not attribute it to routing without evidence.
- Re-running the wrong identity can contaminate comparisons: require parent and input hashes at
  startup and write a new activity directory.
- A laptop interruption can leave a paid request unresolved: reuse existing checkpoint recovery
  and billing-uncertain safeguards, and pause only after a complete item is persisted.
