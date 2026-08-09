# Affected-Part Targeted Vision A/B Results

## Outcome

The fixed targeted run completed six logical and six physical requests with zero retries and no request-audit incident. This document and all evaluation artifacts were produced offline after execution; the evaluation made zero API requests.

`TARGETED_AB_DECISION = NO_CLEAR_IMPROVEMENT`

`PROMPT_STRATEGY_RECOMMENDATION = NONE`

`NEXT_EXPERIMENT = LOCALIZATION_GUIDED_ROI`

`PHASE_2B_RECOMMENDATION = BLOCK`

## Request audit

| Item | Result |
|---|---:|
| Logical requests | 6 |
| Physical requests | 6 |
| Automatic retries | 0 |
| Explicit retry requests | 0 |
| HTTP responses received | 6 |
| Raw responses saved | 6 |
| Parsed responses saved | 6 |
| Schema-valid responses | 3 |
| Completed ledger reservations | 6 |
| Request incident | false |

Every response retains its internal request ID, logical request ID, API response ID, duration, raw response, parsed response, schema result, candidate membership result, verifier result, and HTTP/API error type. No API key is stored in these evaluation outputs.

## Case comparison

| Case | Variant | Ground Truth | Prediction | Confidence | Schema | Exact | At least one | All parts | Candidate | Verifier |
|---|---|---|---|---:|---|---|---|---|---|---|
| missingpart-A01 | Reference | `PIN_RED_SHORT` | `EYE_BALL` | 0.95 | invalid | false | false | false | N/A | not run |
| missingpart-A01 | Reference+Candidate | `PIN_RED_SHORT` | `EYE_BALL` | 0.95 | valid | false | false | false | valid membership | conflict; blocked |
| missingpart-B01 | Reference | `WHEEL_BLUE_SMALL` | `PIN_YELLOW` | 0.95 | invalid | false | false | false | N/A | not run |
| missingpart-B01 | Reference+Candidate | `WHEEL_BLUE_SMALL` | `EYE_BALL` | 0.95 | valid | false | false | false | valid membership | uncertain; blocked |
| wrongpart-B01 | Reference | `PIN_RED_SHORT\|PIN_YELLOW` | `PIN_RED_SHORT` | 0.95 | invalid | false | true | false | N/A | not run |
| wrongpart-B01 | Reference+Candidate | `PIN_RED_SHORT\|PIN_YELLOW` | `PIN_RED_SHORT` | 0.95 | valid | false | true | false | valid membership | uncertain; blocked |

Frozen confirmed Ground Truth was joined by exact image ID only after inference. It was not included in any request package or inference input.

## Primary metrics

Only schema-valid responses with exact-image confirmed frozen Ground Truth enter primary metrics. Consequently, Reference has denominator zero and its rates are `null`/N/A rather than misleading zero-percent values.

| Metric | Reference | Reference+Candidate |
|---|---:|---:|
| Schema-valid rate | 0/3 (0%) | 3/3 (100%) |
| Exact Set Match | N/A | 0% |
| At-least-one Recall | N/A | 33.33% |
| All-parts Recall | N/A | 0% |
| Part Precision | N/A | 33.33% |
| Part Recall | N/A | 25.00% |
| Part F1 | N/A | 28.57% |
| Unknown Rate | N/A | 0% |
| False-confident Identity @0.70/@0.80/@0.90 | N/A | 66.67% |
| False-confident Case @0.80 | N/A | 66.67% |
| Candidate Violation Rate | N/A | 0% |
| Verifier Acceptance | N/A | 0% |
| Verifier Conflict | N/A | 33.33% |
| Verifier Unresolved/Uncertain | N/A | 66.67% |
| Wrong Identity Escaped Verifier | not evaluated | 0 |
| Wrong Identity Blocked | not evaluated | 2 |
| Correct Identity Blocked | not evaluated | 0 |

All three Candidate predictions fell in the 0.90-1.00 confidence bin: one identity was correct and two were incorrect, for 33.33% empirical accuracy.

## A01 forensic

- Reference returned `EYE_BALL` at 0.95 and again included `$schema`, `title`, and `type`; schema validation failed and the verifier correctly did not run.
- Reference+Candidate returned `EYE_BALL` at 0.95. Its 15-item candidate set contained both `EYE_BALL` and the Ground Truth `PIN_RED_SHORT`, so membership was valid but identity was wrong.
- Candidate did not output `PIN_RED_SHORT` or an UNKNOWN identity.
- The verifier returned `conflict`, left the verified ID empty, and required manual review. The wrong identity did not escape.

## missingpart-B01 and wrongpart-B01

For missingpart-B01, Ground Truth is `WHEEL_BLUE_SMALL`. Reference predicted `PIN_YELLOW`; Candidate predicted `EYE_BALL`. Both were high-confidence wrong identities, and the Candidate result shows the same visually salient eye bias seen in A01.

For wrongpart-B01, Ground Truth is the composite swap pair `PIN_RED_SHORT|PIN_YELLOW`. Both variants returned only `PIN_RED_SHORT`. This earns at-least-one recall but fails Exact Set Match, All-parts Recall, and composite full recall. Candidate did not recover the missing `PIN_YELLOW` side of the swap; its verifier returned uncertain and did not verify the identity.

## Reference schema validity

Reference schema-valid rate is 0/3. All three failures exactly repeat the earlier metadata-echo pattern: top-level `$schema`, `title`, and `type` violated the current schema's `additionalProperties: false`. The targeted runner successfully preserved raw and parsed responses and did not retry, but it did not remediate the Reference response-format behavior. The production schema should not be loosened to accept schema-definition metadata.

A future experimental format test may avoid placing the full schema definition in ordinary prompt text or use a provider-supported strict structured-output contract. Any analysis-only metadata stripping must remain explicit and excluded from primary metrics unless separately validated and approved.

## Candidate effectiveness

All three Candidate lists contain 15 canonical IDs, exactly 100% of the part inventory. Ground Truth and prediction are members in every case, and Candidate Violation Rate is zero. Each constraint is therefore classified `weak`: membership compliance proves only that the model selected an inventory item, not that the candidate strategy improved semantic identity.

## Historical directional comparison

| Metric | Historical frozen baseline (n=25) | Targeted Candidate (n=3) | Direction only |
|---|---:|---:|---|
| Exact Match | 8.00% | 0% | worse |
| At-least-one Recall | 12.50% | 33.33% | better |
| All-parts Recall | 4.17% | 0% | worse |
| Part F1 | 10.5263% | 28.57% | better |
| False-confident Identity @0.80 | 88.00% | 66.67% | better, but still high |

These denominators and samples are not comparable enough for a significance claim. The mixed direction is supporting context only and does not overturn the targeted case failures.

## Decision and next experiment

Reference cannot be selected because all three responses are schema-invalid. Reference+Candidate cannot be promoted because it achieved zero Exact Match and zero All-parts Recall, retained two high-confidence eye errors, and recovered only half of the wrongpart swap. Zero candidate violations and zero verifier escapes are safety successes, not identity-quality success.

The next experiment should be `LOCALIZATION_GUIDED_ROI`, not a larger one-shot prompt:

1. Compare test and correct-reference localization to identify a visual-delta ROI.
2. Crop the corresponding test and reference ROIs.
3. Build a restricted evidence-derived candidate set without Ground Truth.
4. Classify part identity within that ROI.
5. Aggregate multi-view evidence where available.
6. Pass the result through the existing fail-closed verifier.

Until A01 no longer produces high-confidence `EYE_BALL` and composite wrongpart identity is recovered reliably, Phase 2B remains blocked.

## Artifacts

All machine-readable outputs are under `analysis/vision_prompt_ab/targeted_run_20260809_111248/evaluation/`:

- `request_audit_summary.json`
- `targeted_ab_predictions.csv`
- `targeted_ab_case_comparison.csv`
- `targeted_ab_metrics.json`
- `targeted_ab_metrics_comparison.csv`
- `confidence_bins.csv`
- `candidate_effectiveness.csv`
- `targeted_ab_decision.json`
