# Vision Part Failure Analysis — 2026-07-01

## Scope and provenance

The primary population is `logs/current_parsed_json/*20260701*.json`. There are 59 successful files but 58 unique `image_name` values; `model03_step03_correct-01_front_01` was run twice. Metrics use the later filename timestamp for each image. All records point to `prompts/vision_v2.txt`, `schema/vision_output_schema.json`, a matching test image, a correct reference image, and `ground_truth/model03/stepXX.json`. Raw responses exist under `logs/current_raw_responses`; this analysis uses the successfully parsed payloads.

Machine-readable rows and metrics are in `analysis/vision_part_failure_analysis_20260701.csv` and `.json`. `analysis/affected_parts_review_template.csv` separates supported labels from items requiring review.

## Baseline results

| Metric | Result |
|---|---:|
| Unique images | 58 |
| Error-type accuracy | 55/58 = 94.83% |
| Affected-part exact set match | 24/52 = 46.15% |
| At least one affected part hit (error cases with supported labels) | 8/32 = 25.00% |
| All affected parts detected | 6/32 = 18.75% |
| Composite full recall | 0/16 = 0.00% |
| Unknown predicted part rate | 9/43 = 20.93% |
| Hallucinated-part row rate | 26/33 = 78.79% |

The affected-part exact denominator includes 20 correct controls, whose expected error-part set is empty. Extrapart-A01's six views are excluded from affected-part scoring because the extra red rod has no unambiguous canonical `part_id`; all six are flagged for human review. Composite A01 requires both `wrongpart` and `extrapart`; B01 requires both swapped identities. This avoids crediting a one-part guess as full composite recall.

By type: correct 19/20, extrapart 6/6, missingpart 16/16, wrongpart 14/16 for error type. Affected-part exact match is 1/16 for missingpart and 4/16 for wrongpart. Thus type classification and part identity are materially different metrics.

By view, error-type accuracy is back 12/14, front 7/8, and 100% for bottom/left/right/top. Affected-part exact rates among evaluable samples are back 5/13, bottom 4/7, front 3/7, left 4/7, right 3/7, and top 5/11. No angle resolves the identity problem reliably.

## Supported affected-part labels

- `missingpart-A01`: `PIN_RED_SHORT` (red short rod).
- `missingpart-B01`: `WHEEL_BLUE_SMALL` (one small wheel is absent).
- `wrongpart-A01`: `EYE_BALL`, with multiplicity/composite details still needing count-aware review.
- `wrongpart-B01`: `PIN_YELLOW` and `PIN_RED_SHORT` swapped.
- `extrapart-A01`: review required; no canonical long-red-rod identity exists in the current library.

These labels come from the explicit handoff observations plus source/reference image inspection, not from `data/ground_truth.csv`. The formal CSV remains untouched and is not used as online repair knowledge.
