# Affected-Part Targeted A/B Plan (Do Not Execute)

## Status and scope

This document defines the next offline-prepared experiment only. It does not authorize execution, contains no API result, and must not be used until an explicit later authorization is given. Production Vision Prompt and production Vision Schema remain unchanged.

The targeted matrix is exactly six logical requests:

| Case | Reference | Reference + Candidate |
|---|---:|---:|
| missingpart-A01 | 1 | 1 |
| missingpart-B01 | 1 | 1 |
| wrongpart-B01 | 1 | 1 |
| **Total** | **3** | **3** |

## Required preconditions

Before any future execution:

1. Use a new isolated experiment directory, experiment UUID, ledger, and exclusive lock. Do not reuse the exhausted incident ledger at `analysis/vision_prompt_ab/results/request_ledger.json`.
2. Set `MAX_PHYSICAL_REQUESTS=6`; reserve each physical request in the persistent ledger before transport.
3. Retain raw response text and parsed output even on schema validation failure.
4. Apply deterministic Candidate membership enforcement to every Reference + Candidate response.
5. Skip completed package IDs on resume. Failed or reserved packages require an explicit retry flag and available physical budget.
6. Permit stale-lock recovery only after PID validation and an explicit operator choice.
7. Confirm the six package manifests and candidate breadth audit offline before authorization. Do not narrow candidates to Ground Truth.

## Evaluation rules

Recovered Reference records from the previous run remain analysis-only and excluded from primary metrics. Ground Truth is used only for evaluation, never inserted into an inference package.

Success criteria, in priority order:

1. Reference schema-valid rate = 100%.
2. Candidate violation rate = 0%.
3. False-confident identity rate decreases.
4. At-least-one identity recall does not decrease.
5. Verifier wrong-identity escape rate = 0%.

A zero Candidate Violation Rate is necessary but insufficient: an in-set incorrect identity such as A01 `EYE_BALL` still counts as semantically wrong. Candidate breadth and identity accuracy must therefore be reported alongside membership compliance.

## Stop conditions

Stop without retry when the physical counter reaches six, a second process owns the lock, package state is ambiguous, or a validation failure cannot preserve the raw response. Do not proceed to Phase 2B based on this plan alone.
