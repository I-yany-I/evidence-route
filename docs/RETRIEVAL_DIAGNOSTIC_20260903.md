# EvidenceRoute Calibration Retrieval Diagnostic

## Scope

This is a provider-free diagnostic of the sentence-level BM25 candidate generator used by
`AveritecFrozenProvider`. It does not measure verdict quality and makes no model or paid API calls.

- Run date: 2026-09-03
- Runtime manifest SHA-256: `7480547e1494f2baaff79747dae92c958e8f6c80a85fb96d6290f007f1d3f61a`
- Gold scorer manifest SHA-256: `160772c08ab7c521caa48ea34778edb94b53fc81b0e201d2bded0f1abc276161`
- Retrieval configuration: `evidence-route-regex-bm25-v1`
- Query: frozen runtime claim text
- Candidate limit: `single_top_k=8`
- Evidence truncation: `single_chars=800` (does not affect BM25 ranking)

Gold data was loaded only in this offline diagnostic, outside the runtime retrieval path. A claim is
eligible when at least one gold answer source occurs among that claim's frozen corpus `source_url`
values. URL identity uses `canonicalize_citation_url`, HTML entity decoding, and equivalence between a
Wayback URL and its embedded original URL. The last rule makes `train-1975` eligible: its gold answer
uses a Wayback URL while the corpus stores the same ProPublica page as the original URL.

## Result

- Calibration claims: 32
- Claims with a gold source present in the frozen corpus: 27
- Eligible claims whose gold source appears in BM25 top-8: 5
- Gold-source recall: `5/27 = 18.52%`
- Hits: `train-237`, `train-519`, `train-538`, `train-889`, `train-1471`
- Excluded because no gold source identity occurs in the corpus: `train-28`, `train-81`,
  `train-512`, `train-1691`, `train-2855`

| claim_id | gold source in corpus | gold source in top-8 |
| --- | --- | --- |
| train-28 | no | excluded |
| train-73 | yes | no |
| train-81 | no | excluded |
| train-164 | yes | no |
| train-183 | yes | no |
| train-237 | yes | yes |
| train-352 | yes | no |
| train-467 | yes | no |
| train-512 | no | excluded |
| train-519 | yes | yes |
| train-538 | yes | yes |
| train-553 | yes | no |
| train-827 | yes | no |
| train-857 | yes | no |
| train-889 | yes | yes |
| train-1094 | yes | no |
| train-1106 | yes | no |
| train-1201 | yes | no |
| train-1471 | yes | yes |
| train-1691 | no | excluded |
| train-1694 | yes | no |
| train-1975 | yes (Wayback/original equivalent) | no |
| train-2161 | yes | no |
| train-2341 | yes | no |
| train-2425 | yes | no |
| train-2464 | yes | no |
| train-2468 | yes | no |
| train-2562 | yes | no |
| train-2579 | yes | no |
| train-2719 | yes | no |
| train-2855 | no | excluded |
| train-3051 | yes | no |

## Failure Example

For `train-2468` ("sodium bicarbonate (baking soda) can cure cancer"), the gold source is present in
the corpus. Its direct refutation is record `av:train:2468:0:7`:

> Available scientific evidence also does not support the idea that sodium bicarbonate works as a
> treatment for any form of cancer or that it cures yeast or fungal infections.

The production claim-query top-8 instead returned:

`av:train:2468:225:155`, `av:train:2468:37:0`, `av:train:2468:292:0`,
`av:train:2468:138:27`, `av:train:2468:435:1`, `av:train:2468:454:6`,
`av:train:2468:246:27`, and `av:train:2468:309:309`.

None belongs to the gold source. The model therefore never receives the strongest frozen refutation.
This supports treating candidate recall as the next bottleneck, ahead of another paid full campaign.

## Decision

Do not tune prompts against these gold fields or expose them to the runtime graph. Use this scorer-only
diagnostic to evaluate a source-aware candidate generator and reranker offline. A new paid campaign
must use a new frozen configuration and activity identity after the retrieval change passes a declared
calibration recall gate.

## Quality Recovery Stability Baseline (2026-09-06)

This provider-free projection freezes the failure categories used by the quality recovery work. It
uses a new experiment identity and does not modify the retrieval diagnostic or any published Gate A
artifact.

- Experiment ID: `quality-recovery-baseline-20260906`
- Activity ID: `evidence-quality-recovery-baseline-20260906`
- Campaign ID: `evidence-quality-recovery-provider-free-baseline-20260906`
- Input: `tests/fixtures/evaluation/quality_recovery_baseline_input.json`
- Input SHA-256: `15371eeb88cb822810413ac07c8bc07df92e87c771a04b5a1078c4fe2a0d6d6d`
- Output: `tests/fixtures/evaluation/quality-recovery-baseline-20260906/diagnostic.json`
- Output SHA-256: `db163c1acb48332f11354344a5a029a6eaa54a87505baafc5f0af948e5f3660b`
- Identity: `tests/fixtures/evaluation/quality-recovery-baseline-20260906/experiment.json`

Run from the repository root in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python -m evidence_route.evaluation.stability_diagnostics `
  --input tests/fixtures/evaluation/quality_recovery_baseline_input.json `
  --output tests/fixtures/evaluation/quality-recovery-baseline-20260906/diagnostic.json
```

The output contains five claims and all fifteen scheduled repeat slots. It retains one partial result,
one failed result, and one synthesized missing repeat in that denominator. The category counts are one
each for `evidence_or_citation_drift`, `route_drift`, `validation_or_status_drift`,
`provider_variance`, and `incomplete_or_failed`; `verdict_drift` is zero. Records are sorted by claim
ID and repeats by repeat number.
