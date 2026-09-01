# EvidenceRoute Stability v4 Design

**Goal:** Raise repeat stability and completion by removing citation-choice noise and recovering terminal multi-path validation failures, while preserving truthful accounting and the immutable Gate A baseline.

## Scope and invariants

- The published Gate A artifacts and metrics remain immutable.
- Fresh runs still use the provider output for the verdict; no cross-repeat cache or gold label is introduced.
- A citation is never fabricated or rebound to an evidence item whose text the model did not cite.
- A failed recovery remains failed or partial and remains in the full-manifest denominator.
- Every extra recovery call is represented in call bounds, budget reservation, the SQLite ledger, and the artifact.
- The 85% target means at least 17 of 20 claims have three valid repeats with the same verdict; completion must be at least 90%.

## Design

### Deterministic citation projection

At the verifier boundary, normalize model citations, discard structurally invalid or unknown citations, deduplicate equivalent source URLs, and select a stable citation per claim unit using the frozen provider order followed by evidence ID and URL. The projection only selects among citations actually returned by the model. If a claim unit has no valid citation, validation reports incomplete coverage instead of inventing a quote. The selection policy is shared by single and judge outputs and is tested with the observed v3 drift patterns.

### Multi recovery

The graph adds one explicit recovery state after a multi result cannot pass validation because of failed workers, incomplete coverage, or low confidence. Recovery calls the existing single verifier once, marks `fallback_used=true`, preserves `initial_route=multi`, and validates the recovery result without granting another escalation. If recovery cannot produce a valid result, the final artifact remains failed/partial with both the original and recovery errors.

### Accounting and reporting

The Gate A call profile and cost preview include one possible single recovery for each multi campaign item. Artifacts expose the recovery marker and retain all call IDs. Stability diagnostics compare recovery markers as route metadata and continue to classify incomplete repeats honestly.

## Verification

Offline tests cover citation projection, duplicate-source handling, per-unit coverage, worker failure, single recovery success, recovery failure, and the increased call bound. Existing artifact, lifecycle, accounting, and manifest tests must remain green. No paid run is started until the offline suite and budget preview pass.

