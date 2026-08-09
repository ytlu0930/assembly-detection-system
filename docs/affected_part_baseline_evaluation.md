# Affected-part identity baseline evaluation

## Outcome

The offline 2026-07-01 Vision baseline has low affected-part identity accuracy despite high reported confidence. On 25 manually confirmed images, Exact Set Match is 8.00%, part-level F1 is 10.53%, and the false-confident identity rate is 88.00% at thresholds 0.70, 0.80, and 0.90.

Error-type classification accuracy and affected-part identity accuracy measure different tasks. The previously observed approximately 94.83% error-type accuracy must not be presented as affected-part identity accuracy. The former asks whether an image is missing/extra/wrong/correct; the latter asks whether the exact canonical affected-part set is correct.

No Vision API was called. Predictions were extracted from the latest 2026-07-01 parsed JSON for each confirmed source image in `logs/current_parsed_json/`.

## Trusted denominator

`analysis/affected_part_eval_ground_truth.csv` contains only `review_status=confirmed` rows from `analysis/affected_parts_review_template.csv`. It does not promote uncertain cases or infer labels from filename target codes.

| Distribution | Count |
|---|---:|
| Confirmed images | 25 |
| missingpart-A01 | 8 |
| missingpart-B01 | 8 |
| wrongpart-B01 | 8 |
| correct-control | 1 |

View distribution is back 7, top 6, and bottom/front/left/right 3 each. Formal error-type distribution is missing 16, wrongpart 8, correct 1. There are no confirmed extra-part rows, so extra-part primary accuracy is N/A.

The A/B evaluation subset has 19 confirmed images: one image for every available case/view combination, capped below 24. It contains six views each for missingpart-A01, missingpart-B01, and wrongpart-B01, plus one correct-control image. Extrapart-A01 and wrongpart-A01 remain `needs_second_review` and are excluded from the primary denominator.

## Baseline metrics

| Metric | Result |
|---|---:|
| Exact Set Match | 8.00% |
| At-least-one-part recall | 12.50% |
| All-parts recall | 4.17% |
| Part-level precision | 12.00% |
| Part-level recall | 9.38% |
| Part-level F1 | 10.53% |
| Unknown part rate | 4.00% |
| Composite full recall | 0.00% |
| Correct-control false-positive rate | 0.00% |

Per-error-type Exact Set Match is correct 100.00% (1/1), missing 6.25% (1/16), and wrongpart 0.00% (0/8). Per-view Exact Set Match is back 14.29% (1/7), bottom 0.00% (0/3), front 0.00% (0/3), left 33.33% (1/3), right 0.00% (0/3), and top 0.00% (0/6).

## False-confident identity KPI

A false-confident identity is a predicted canonical identity that is not in the confirmed affected-part set and whose model confidence is at least the selected threshold. Identity rate uses high-confidence predicted identities as its denominator. Case rate uses all evaluated images as its denominator.

| Threshold | False identities | High-confidence identities | Identity rate | False-confident cases | Case rate |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 22 | 25 | 88.00% | 21/25 | 84.00% |
| 0.80 | 22 | 25 | 88.00% | 21/25 | 84.00% |
| 0.90 | 22 | 25 | 88.00% | 21/25 | 84.00% |

## Confidence bins

| Confidence | Predictions | Correct | Incorrect | Empirical identity accuracy |
|---|---:|---:|---:|---:|
| 0.00-0.49 | 0 | 0 | 0 | N/A |
| 0.50-0.69 | 0 | 0 | 0 | N/A |
| 0.70-0.79 | 0 | 0 | 0 | N/A |
| 0.80-0.89 | 0 | 0 | 0 | N/A |
| 0.90-1.00 | 25 | 3 | 22 | 12.00% |

All emitted identities are concentrated at 0.95 confidence, but only 12% are correct. Current confidence therefore does not represent empirical identity correctness on this confirmed sample.

## missingpart-A01 trace

Ground Truth is `PIN_RED_SHORT` for eight confirmed views. Only the left view predicts it correctly. Bottom, front, right, top_01, and top_02 predict `EYE_BALL`; back_01 and back_02 predict `BLOCK_YELLOW_CUBE`. Every prediction is confidence 0.95. The front-view `EYE_BALL` result is therefore one instance of a broader multi-view identity problem, not a downstream image-generation problem.

## Limitations

- The sample is small and intentionally limited to 25 confirmed images from four case groups.
- Extrapart-A01 and wrongpart-A01 are excluded pending second review; extra-part accuracy is not yet measurable.
- Only one confirmed correct control is available.
- Confidence calibration statistics are descriptive; no calibration model was fitted.
- Baseline predictions do not contain verifier outcomes, so verifier acceptance/conflict/unresolved and escape metrics are N/A until A/B results run through the verifier.
- Ground Truth labels are evaluation-only and are never supplied to candidate generation or the Vision request.

Machine-readable outputs are under `analysis/affected_part_baseline/`.

Validation: the focused affected-part suite passed 17 tests; verifier/regression tests passed 19 tests; the complete isolated `tests/` suite passed 172 tests and 19 subtests. The originally requested legacy `output/pytest_temp` and `output/pytest_cache` paths have Windows ACL denial and fail during collection, before test execution; a fresh isolated basetemp/cache confirms this is an environment-path issue rather than a regression.
