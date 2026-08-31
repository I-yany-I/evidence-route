# Batched Evaluation Execution Design

## Goal

Allow the frozen EvidenceRoute calibration and Gate A campaign to run in bounded, resumable
batches without changing the evaluation cohort, prompt, model alias, routing policy, pricing, or
publication rules.

## Invariants

- A batch boundary is reached only after the current calibration case or campaign work item has
  produced and persisted a complete artifact with closed provider accounting.
- A batch pause is recorded as `user_paused`, distinct from process interruption, budget stop,
  usage missing, model drift, and billing uncertainty.
- A paused activity remains incomplete and cannot pass reporting/publication gates.
- Resume accepts only the persisted frozen plan and existing ledger; completed artifacts are
  reused and no paid call is repeated.
- An in-flight request is never cancelled by the batch mechanism.

## Interface

- Calibration adds `--max-cases`, defaulting to `4` for collection and applying only to the
  current invocation.
- Campaign evaluation adds `--max-items`, defaulting to `10` for paid execution and applying
  only to the current invocation.
- `--resume` continues the same activity/campaign and accepts the same batch limit.
- A limit must be a positive integer. If the remaining work is smaller than the limit, the
  invocation finishes the phase normally.

## State model

`CampaignStatus.PAUSED` and `CampaignStopReason.USER_PAUSED` are added as a resumable pair. The
campaign runner pauses between work items. Calibration uses the same activity-level pause reason
between complete cases; its case state remains `pending` for future cases. Resume clears only
`user_paused` after the existing ledger and freeze checks pass.

## Verification

Tests cover: positive limit validation, campaign pause after exactly N completed items, campaign
resume without re-executing completed items, calibration pause/resume at case boundaries, CLI
defaults/forwarding, and preservation of publication rejection for paused activities. Existing
billing-uncertain and process-interruption tests remain unchanged.
