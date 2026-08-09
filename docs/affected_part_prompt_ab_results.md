# Affected-Part Prompt A/B Results

## Experiment Design

- Experiment: `affected-part-prompt-ab-20260809`
- Provider: Azure OpenAI Chat Completions
- Deployment: `gpt-4o`
- API version: `2024-12-01-preview`
- Frozen schema: `schema/vision_output_schema.json`
- Variants: Baseline, Reference, Reference+Candidate
- Demonstration cases: six front-view cases
- Logical packages: 18
- Unique logical artifacts: 18 (12 success, 6 failed)
- Frozen confirmed Ground Truth: 25 images
- Successful confirmed execution denominator: Baseline 2, Reference 0, Reference+Candidate 3
- Excluded from primary metrics: extrapart-A01 and wrongpart-A01 (`needs_second_review`), plus front correct-control (`not_in_frozen_evaluation`)

The run is **PARTIAL**. Five Reference error-case responses returned schema-shaped objects rather than valid schema instances, and missingpart-B01 Baseline was interrupted by the outer runner timeout. The failed Reference rows are not silently converted into predictions.

### API request incident

The first shell command reached its outer timeout, but its Python child continued in the background. A resume process was started under the mistaken assumption that the child had terminated. The two processes overlapped and the resume process issued 13 duplicate physical requests. Reconstructed physical request count is 31, exceeding the intended budget of 18. No further request was made after discovery. The runner now uses an exclusive experiment lock to prevent a second execution process from starting while an orphaned/active process owns the run.

## Baseline Recap

The frozen 25-image historical baseline remains unchanged: Exact Set Match 8.00%, At-least-one Recall 12.50%, All-parts Recall 4.17%, precision 12.00%, recall 9.375%, F1 10.5263%, Unknown Rate 4.00%, Composite Full Recall 0%, Correct FP 0%, false-confident identity 88% at 0.70/0.80/0.90, and false-confident case rate 84% at 0.80.

The Variant A figures below are a new API execution on only the successful confirmed front-view packages. They are not a replacement for the frozen 25-image baseline.

## Overall Results

| Metric | Baseline (n=2) | Reference (n=0) | Reference+Candidate (n=3) |
|---|---:|---:|---:|
| Exact Set Match | 0.00% | N/A | 0.00% |
| At-least-one Recall | 50.00% | N/A | 33.33% |
| All-parts Recall | 0.00% | N/A | 0.00% |
| Part Precision | 33.33% | N/A | 33.33% |
| Part Recall | 33.33% | N/A | 25.00% |
| Part F1 | 33.33% | N/A | 28.57% |
| Unknown Rate | 0.00% | N/A | 0.00% |
| Composite Full Recall | 0.00% | N/A | 0.00% |
| Correct-control FP | N/A | N/A | N/A |

Direct ranking is not statistically valid because successful confirmed denominators differ. Reference has no successful confirmed error response. Candidate does not show a directional improvement over the available Baseline rows: Exact Match stays at 0%, At-least-one Recall and F1 are lower, and false-confident identity is unchanged.

## False-Confident Analysis

| Metric | Baseline | Reference | Reference+Candidate |
|---|---:|---:|---:|
| False-confident identity @0.70 | 66.67% | N/A | 66.67% |
| False-confident identity @0.80 | 66.67% | N/A | 66.67% |
| False-confident identity @0.90 | 66.67% | N/A | 66.67% |
| False-confident case @0.80 | 100.00% | N/A | 66.67% |

The lower Candidate case rate reflects three evaluated cases rather than two and must not be interpreted as a clean improvement. Identity-level false confidence did not decline.

## Confidence Calibration

Baseline has three predictions in the 0.90-1.00 bin: one correct, two incorrect, empirical accuracy 33.33%. Reference has no evaluable predictions; every bin is N/A. Reference+Candidate also has three predictions in the 0.90-1.00 bin: one correct, two incorrect, empirical accuracy 33.33%. All lower bins have zero denominator and remain N/A.

## Verifier Interaction

| Metric | Baseline | Reference | Reference+Candidate |
|---|---:|---:|---:|
| Acceptance | 0.00% | N/A | 0.00% |
| Conflict | 50.00% | N/A | 33.33% |
| Unresolved/uncertain | 50.00% | N/A | 66.67% |
| Verified correct | 0 | 0 | 0 |
| Verified wrong | 0 | 0 | 0 |
| Wrong identity escaped | 0 | 0 | 0 |
| Wrong identity blocked | 2 | 0 | 2 |

The production verifier remains fail closed and blocked every evaluated wrong identity. Its thresholds were not changed. Prompt changes did not increase acceptance because they did not produce independently supported identities.

## missingpart-A01

| Variant | Request status | Prediction | Confidence | Verifier |
|---|---|---|---:|---|
| Baseline | success | `EYE_BALL` | 0.95 | conflict |
| Reference | failed validation | N/A | N/A | not evaluated |
| Reference+Candidate | success | `EYE_BALL` | 0.95 | conflict |

Candidate constraint did not fix A01 and did not return UNKNOWN. It retained the same high-confidence wrong identity. The verifier blocked it, so the result is safe downstream but not an identity-accuracy improvement.

## Qualitative Unconfirmed Cases

These results are excluded from primary metrics.

- extrapart-A01: Baseline and Candidate both returned `UNKNOWN_EXTRA_PART` at 0.95 and were unresolved; Reference failed validation.
- wrongpart-A01: Baseline returned `EYE_BALL|ROD_GREEN_LONG` (0.95/0.90), Candidate returned `BLOCK_GREEN_4HOLE_2PEG` (0.95), and both were uncertain; Reference failed validation.
- front correct-control: all three variants returned `correct`, but this image is not part of the frozen confirmed evaluation CSV, so Correct FP remains N/A.

## Interpretation

- Reference guidance effectiveness: **not measurable**. All four Reference error cases failed schema validation; only its correct-control request succeeded.
- Candidate constraint effectiveness: **no demonstrated improvement**. It constrained IDs but retained high-confidence A01 error and reduced available recall/F1.
- False-confidence reduction: **no** at the identity level.
- UNKNOWN increase: no increase on confirmed cases; UNKNOWN was used only for the unconfirmed extra-part case.
- Safety improvement: the verifier prevented escape, but this is containment already present before the Prompt experiment, not evidence that Candidate improved Vision identity.

## Limitations

- Confirmed extra-part count is zero.
- wrongpart-A01 remains unresolved.
- Only one confirmed correct control exists, and it is a back view rather than the executed front control.
- Successful confirmed denominators differ by variant (2/0/3).
- Sample size is very small and only front-view demonstration packages were executed.
- Confidence remains concentrated at 0.90-1.00.
- Only the current schema was tested.
- Failed validation artifacts from the original runner retained error details but did not retain the received raw response body; this is fixed for future failures but cannot be reconstructed offline.
- Physical requests exceeded budget because the timed-out shell left a child process running; the corrected audit records 31 and the runner now has an exclusive lock.

## Recommendation

Do not update the production Prompt. Fix and test the experiment execution contract first: enforce process locking, preserve failed raw responses, and make the common schema instruction unambiguously request a schema instance without changing the frozen experiment retroactively. Then run a newly authorized, small regression with identical confirmed cases and denominators before considering broader A/B, Schema vNext, or Phase 2B.

Decision: `NO_CLEAR_IMPROVEMENT`. Recommended variant: none. Production Prompt change: inconclusive/not recommended from this run. Phase 2B remains blocked.

## Validation

- Focused affected-part/runner/verifier tests: 33 passed.
- Full test suite: 173 passed, 19 subtests passed.
- `python -m compileall -q scripts utils tests`: PASS.
- `git diff --check`: PASS (line-ending warnings only).
- Production Prompt, Schema, Ground Truth, and source images have no Git status changes.
