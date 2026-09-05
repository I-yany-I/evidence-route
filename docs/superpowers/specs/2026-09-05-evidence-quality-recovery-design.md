# EvidenceRoute Quality Recovery Design

**Goal:** Raise EvidenceRoute quality against the frozen AVeriTeC evaluation without weakening auditability, failure accounting, or cost bounds.

## Success criteria

The work proceeds through four gated stages:

1. Establish a reproducible failure baseline and regression corpus.
2. Improve evidence candidate recall and deterministic citation binding. The first gate is calibration gold-source recall above the current 5/27 (18.5%).
3. Stabilize routing, validation, and status transitions. The stability gate is at least 17/20 consistent claims (85%).
4. Harden provider compatibility, model drift handling, retries, and budget accounting. The final quality gate is full-manifest macro-F1 at least the Gate A baseline of 0.392, with all failures retained in the denominator.

No paid campaign may be started merely because a code change passes unit tests. Each stage must pass its offline diagnostic and preserve the existing artifact, manifest, and billing contracts.

## Scope and constraints

The frozen balanced manifests, gold labels, evaluator boundary, and historical reports remain unchanged. New experiments use a new identity and isolated artifact directory. Gold evidence is never available to retrieval or runtime verification code. Provider-reported model IDs remain observations, not authenticated identity.

The implementation follows TDD: every defect starts with a minimal failing test, the test failure is observed, then the smallest implementation is added and the full relevant suite is rerun. SDD tasks are ordered and independently reviewable.

## Architecture and data flow

The pipeline remains split into deterministic boundaries:

`claim -> clause/unit analysis -> candidate retrieval -> deterministic ordering/deduplication -> single or multi verification -> citation projection -> schema validation -> artifact/ledger/report`.

Retrieval owns candidate generation and source provenance. Verification may select only candidates supplied to it. Citation projection binds final citations to selected evidence and claim units, preserving source URL, evidence ID, quote, and order. Routing decides execution mode but cannot alter evidence identity. Validation rejects malformed or unsupported outputs and distinguishes infrastructure failure from genuine lack of evidence. Artifacts and the ledger retain every attempt, status, failure reason, usage record, and model observation.

## Stage 1: baseline and regression corpus

Add provider-free diagnostic fixtures for the currently failing claims and categories: evidence/citation drift, route drift, validation/status drift, incomplete/failed, and provider variance. The diagnostic must report per-claim differences, candidate recall, selected evidence, citations, status, and failure reason. It must be deterministic and must not invoke a network transport.

The baseline command and its input hashes are recorded in a new experiment directory. Existing published Gate A artifacts are read-only parents and cannot be overwritten.

## Stage 2: retrieval and citation recovery

Add a source-aware candidate generator behind an explicit retrieval mode. Preserve the current sentence BM25 mode as the default and make any new mode opt-in. Candidate ranking uses claim units, source identity, normalized URL, and deterministic tie-breakers. Candidate count and per-source caps are bounded. Deduplication retains the first item in frozen retrieval order while preserving original evidence IDs and text.

Add an offline calibration gate that compares retrieved candidate IDs with gold-source IDs without exposing gold data to runtime code. Fail closed when the selected retrieval asset or receipt is missing or mismatched. Citation projection must be deterministic across repeated runs, cover all required claim units when possible, remove duplicate URLs, and reject citations whose evidence was not supplied to the verifier.

## Stage 3: routing and validation stability

For each stability claim, compare repeats in deterministic-field order: evidence/citation, route, validation/status, then verdict. Fix nondeterministic ordering, ambiguous route thresholds, and state transitions exposed by the diagnostics. Recovery remains bounded to one authorized multi-to-single attempt, retains the initial route and original error, and records the recovery event in artifacts and the ledger.

Any incomplete, failed, cancelled, or not-run result remains visible and is penalized by full-manifest metrics. No fallback may turn an infrastructure failure into `Not Enough Evidence`.

## Stage 4: provider and cost hardening

Add contract tests for OpenAI-compatible responses with missing usage, malformed structured output, response model drift, transient transport errors, and billing uncertainty. Retries require an explicit bounded budget and idempotent call identity. Missing usage or uncertain billing stops strict evaluation. Model IDs, endpoint hash, price source, and usage source are persisted for every call.

## Testing and release gates

Unit tests cover each pure function and contract boundary. Integration tests replay frozen fixtures through the graph and artifact store. Diagnostic tests assert that missing, partial, failed, and drifted repeats remain in reports. Before any paid run, execute the complete offline suite, retrieval calibration gate, budget preview, capability smoke test, and identity/hash checks. A paid campaign is accepted only when all four stage gates and publication audits pass.

## Out of scope

This design does not add a live MCP provider, Streamlit UI, new benchmark labels, or unverified model claims. Those remain separate work until the Gate A recovery gates pass.
