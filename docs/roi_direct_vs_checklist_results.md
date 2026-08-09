# ROI Direct vs Checklist Results

## Research question and experiment design

The targeted experiment tested whether component-level checklist decomposition improves affected-part identity over direct classification when both methods receive the same frozen localization package. The cases were missingpart-A01, missingpart-B01, and wrongpart-B01, with one Direct and one Checklist request per case.

The user executed the six authorized Vision requests. This evaluation was entirely offline: it did not resume or retry the experiment, call Azure/OpenAI, use GPT Image, or execute Phase 2B.

## Request audit and response freeze

The ledger contains six logical requests, six physical reservations, six completed statuses, and zero explicit retries. All six raw and parsed artifacts are present and have no HTTP/API error. Direct responses are schema-valid (3/3); Checklist responses are schema-invalid (0/3), so total original schema validity is 3/6.

Before loading confirmed labels, all response files were copied to `evaluation/frozen_responses/` and recorded in a content-hash manifest with `labels_loaded=false`. Evaluation validates the six frozen SHA-256 values before joining exact-image Ground Truth.

Every Checklist response used the same nonconforming format: top-level `results`, `CHECK` or `check_result`, categorical confidence, and no `evidence_summary`. The raw/frozen responses remain untouched and excluded from original schema-valid counts.

The experiment-only `utils/roi_checklist_response_normalizer.py` reparses raw message content and applies only known contract transformations. It enforces exact request-candidate membership, validates against the experiment schema, accepts no Ground Truth parameter, and fails closed on ambiguity. All three responses normalize deterministically and validate (3/3); SHA-256 before/after proves the original response files are unchanged. The normalized semantic fields exactly match the earlier label-free recovery. Thus the two distinct results are: **original model schema compliance 0/3**, and **post-hoc normalized analysis schema compliance 3/3**. The latter is compatibility handling, not successful original API compliance.

## ROI candidate reduction recap

The candidate counts remain 5, 5, and 6 from a 15-part inventory, with mean reduction 64.44%. Confirmed GT coverage is 3/3 and `EYE_BALL` is absent from all three candidate sets. Localization remains unverified and every case requires manual review.

## Affected-part results

| Metric | ROI Direct | ROI Checklist recovered analysis |
|---|---:|---:|
| Exact Set Match | 33.33% | 33.33% |
| At-least-one Recall | 33.33% | 66.67% |
| All-parts Recall | 33.33% | 66.67% |
| Part Precision | 25.00% | 42.86% |
| Part Recall | 25.00% | 75.00% |
| Part F1 | 25.00% | 54.55% |
| Unknown Rate | 0.00% | 33.33% |
| False-confident Identity @0.80 | 100.00% | 57.14% |
| False-confident Case @0.80 | 100.00% | 50.00% |
| Manual Review Rate | 100.00% | 100.00% |
| Wrong Identity Escaped Verifier | 0 | 0 |

Under strict original-schema handling, all three Checklist responses are UNKNOWN/unresolved because none validates. The recovered numbers above measure semantic signal only and must not be presented as 3/3 schema-valid production behavior.

## Checklist component performance

Across 16 candidate checks, six are UNCERTAIN and ten are resolved. The resolved-only confusion matrix is TN=3, FP=4, FN=0, TP=3. Resolved accuracy is 60.00%, precision 42.86%, recall 100.00%, and F1 60.00%; overall UNCERTAIN rate is 37.50%.

UNCERTAIN is not converted to PASS or FAIL. This protects missingpart-B01 from a confident wrong wheel decision, but increases manual review. The zero FN value is paired with four false positive FAIL checks, especially in wrongpart-B01, so recall alone overstates checklist quality.

## Case studies

### missingpart-A01

- Ground Truth: `PIN_RED_SHORT`.
- Direct: `PIN_RED_SHORT`, confidence 0.68, exact match; verifier conflict/manual review.
- Checklist: `PIN_RED_SHORT=FAIL` at recovered confidence 0.90; PIN_YELLOW and LINK_BLUE_5HOLE PASS; ROD_GREEN_LONG and LINK_RED_3HOLE UNCERTAIN.
- Rule result: `PIN_RED_SHORT`, exact match, but still verifier conflict because the frozen ROI is unverified.
- `EYE_BALL` cannot leak into either result because it is absent from the reduced candidate set.

### missingpart-B01

- Ground Truth: `WHEEL_BLUE_SMALL`.
- Direct: incorrectly predicts `PIN_RED_SHORT` at 0.85, a false-confident identity.
- Checklist: `WHEEL_BLUE_SMALL` and three other candidates are UNCERTAIN; only PIN_YELLOW is PASS.
- Rule result: no affected identity, UNKNOWN/unresolved. Checklist avoids another high-confidence wrong answer but does not recover the wheel identity.

### wrongpart-B01

- Ground Truth set: `PIN_RED_SHORT|PIN_YELLOW`.
- Direct: predicts `LINK_BLUE_5HOLE|LINK_GREEN_5HOLE`; Exact/at-least/all-parts all fail.
- Checklist: all six candidates are FAIL; the rule result includes both swap GT identities plus four false positives.
- Checklist therefore has at-least-one and all-parts recall for the swap pair, but not Exact Set Match and not a clean two-part swap conclusion.

## Deterministic annotations and figures

No GPT Image output is needed for this experiment. The deterministic renderer preserves the source hash and only draws a Test-frame correction bbox for verifier-accepted evidence. Since every identity is conflict or unresolved, final correction panels fail closed without asserting a box. Separate ROI panels label frozen proposals as `UNVERIFIED_ROI`; they must not be interpreted as verified localization.

Generated artifacts include the resolved-only confusion matrix, Direct/Checklist metric comparison, three fail-closed annotated Test images, and three four-panel case figures. CSVs use pipe-delimited part sets and empty cells for null values—never Python list strings, `NaN`, or `None` literals.

## Interpretation and decision

Checklist decomposition shows a promising recovered semantic signal: equal Exact Match, higher recall/F1, lower false-confidence, and zero wrong-identity escape. It also increases UNKNOWN/manual-review behavior rather than forcing a confident wheel prediction for missingpart-B01.

However, all three original Checklist responses violate the experiment schema, and wrongpart-B01 over-flags all candidates instead of isolating the swap pair. Therefore this execution does not establish a deployable Checklist improvement.

- Decision: `NO_CLEAR_IMPROVEMENT`
- Recommended production method: `NONE`
- Output strategy: `DETERMINISTIC_ANNOTATION` in fail-closed/manual-review mode
- GPT Image recommendation: do not use for this affected-part result
- Phase 2B: `BLOCK`

Before another request-based experiment, fix and offline-test the Checklist response contract and strengthen paired ROI localization. No production Prompt/Schema change is justified from these three recovered cases.
