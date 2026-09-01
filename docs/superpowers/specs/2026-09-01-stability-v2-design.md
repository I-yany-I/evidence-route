# EvidenceRoute Stability V2 Design

## Goal

Improve repeated adaptive-run consistency without changing the published Gate A baseline or
reusing fresh-run predictions as a substitute for evaluation.

## Scope

The new behavior is opt-in and applies only to an isolated stability-v2 experiment. Historical
Gate A configuration, prompt hash, artifacts, and reports remain unchanged.

## Design

1. **Deterministic routing.** Add an opt-in routing flag. Clear rule cases keep their existing
   behavior; ambiguous adaptive cases use conservative `multi` without an LLM router call. The
   decision remains auditable through a typed reason code.
2. **Deterministic decomposition.** In hardened mode, split the already computed lexical claim
   units into at most three tasks in their existing unit order. No extra LLM decomposer call is
   made. Each task query is the exact unit text.
3. **Stable evidence boundary.** Preserve total BM25 ordering and canonical citation URLs. At
   the judge boundary, deduplicate citations by evidence ID and emit them in evidence-ID order.
   Validation compares only evidence owned by execution state.
4. **Conservative verdict contract.** Hardened judge instructions require the full claim to be
   supported before returning `Supported`; partial support is `Not Enough Evidence`, and
   irreconcilable opposing evidence is `Conflicting Evidence/Cherrypicking`. The hardened judge
   prompt has its own version/hash and never changes the Gate A prompt hash.
5. **Bounded adjudication.** Add an opt-in pure adjudication helper for completed, valid
   candidate results. It chooses a verdict by deterministic precedence: a unanimous verdict is
   retained; otherwise, conflicting `Supported` and `Refuted` candidates become `Conflicting`;
   mixed verdicts involving `Not Enough Evidence` become `Not Enough Evidence`. It selects the
   lowest evidence-ID-sorted valid citation set and records an adjudication error when candidates
   disagree. It is not used to hide failed or incomplete calls.

## Data Flow

The experiment loads its hardened config, constructs the same frozen provider and ledger, and
passes the mode into router, decomposer, judge, and validator components. The campaign runner,
checkpointing, billing safeguards, and artifact fingerprints remain shared with Gate A. A new
experiment identity records the hardened config/prompt hashes and uses a separate activity,
campaign, report, checkpoint, and run-store directory.

## Acceptance

Offline tests must pass before any paid call. A paid v2 experiment is successful only if it reaches
at least 17/20 strict consistency, at least 90% completion, complete usage for every call, no
uncertain billing, and intact parent/artifact hashes. Otherwise it is published only as failure
analysis and the original Gate A report remains authoritative.

