# EvidenceRoute Quality Recovery

This document defines the publication contract for the quality-recovery campaign. Historical Gate A
reports are immutable parents. A recovery result is publishable only from a new experiment identity
whose activity, checkpoint database, run store, and report all live below one isolated experiment
directory.

## Required gates

The generated `summary.json` contains `offline_gates` with the measured value, required value,
pass/fail result, and SHA-256 evidence for each offline check:

- retrieval calibration: 32 unique claims including `train-2468`, at least 22 candidate-source hits,
  at least 14 final top-8 hits, and a final hit for `train-2468`;
- strict stability: exactly 20 claims with at least 17 consistent three-run results;
- budget preview: conservative startup cost does not exceed the campaign cap and
  `paid_execution_started` is false;
- final quality: adaptive full-manifest macro-F1 is at least the Gate A baseline of `0.392`.

Missing or malformed evidence, a changed parent report/configuration, or changed repeat-0 parent
artifact hashes blocks publication. Partial, failed, cancelled, and missing dev results remain in the
80-claim full-manifest denominator.

## Evaluation sequence

1. Run the provider-free retrieval calibration and retain its `diagnostic.json`.
2. Run `evaluate` without `--accept-paid-campaign`; retain the emitted JSON as the budget preview.
3. Start or resume the paid campaign with a new `--activity-id` and `--campaign-id`, all four parent
   identity arguments, and all mutable paths below `--experiment-dir`.
4. Generate the report with `--experiment-dir`, `--parent-report`, `--parent-config`,
   `--retrieval-gold-manifest`, `--retrieval-diagnostic`, and `--budget-preview`. Use
   `--parent-activity-dir` when the parent is outside the standard artifact location.
5. Publish only when `summary.json` reports `publishable: true` and every offline gate passes.

The CLI rejects paid evaluation with the historical `gate-a` / `gate-a-dev` defaults and requires
the recovery activity directory, run store, report output, and mutable README to remain inside the
experiment directory. `experiment.json` freezes the parent report,
parent configuration, manifests, pricing, prompt, current configuration, model observation, repeat
schedule, and repeat-0 artifact hashes for resume and report-time verification.
