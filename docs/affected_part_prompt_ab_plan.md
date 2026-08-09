# Affected-part Prompt A/B plan

## Status

The offline Prompt A/B framework is ready, but the real API experiment has not been run. The dry-run plan contains six demonstration cases, three prompt variants, and 18 estimated requests. Actual API requests in this task: zero.

All conditions use the current production schema at `schema/vision_output_schema.json`, allowing the experiment to isolate prompt and evidence-presentation effects. Production Prompt and Schema files are unchanged.

## Experimental conditions

| Variant | Prompt | Schema | Purpose |
|---|---|---|---|
| A — Baseline | `experiments/prompts/vision_affected_parts_baseline.txt` | Current | Preserve current affected-part decision behavior |
| B — Reference | `experiments/prompts/vision_affected_parts_reference_guided.txt` | Current | Force reference/test delta before identity |
| C — Reference+Candidate | `experiments/prompts/vision_affected_parts_reference_candidate.txt` | Current | Add deterministic canonical candidate constraint |

Variant B requires this order without requesting chain-of-thought: compare structure, locate visual delta, identify the expected part in the region, determine the relation, and output affected parts with a concise evidence summary. Variant C additionally permits only supplied candidate IDs or UNKNOWN/UNRESOLVED.

## Candidate construction

`utils/affected_part_candidate_builder.py` derives candidates from expected state and the canonical part library. Inputs may additionally contain allowed observed canonical IDs and swap pairs for wrongpart experiments.

- Missing: stable, deduplicated expected parts for the model/step.
- Extra: expected parts plus `UNKNOWN_EXTRA_PART`.
- Wrongpart: expected parts plus canonical observed/swap evidence when supplied.
- IDs absent from the part library are discarded, except the explicit unknown sentinel.
- Sorting is deterministic and duplicate-free.
- Human review CSVs, case labels, and Ground Truth target identities are never read.
- There is no missingpart-A01 or `PIN_RED_SHORT` special case. That ID appears for model03/step03 only because it is present in expected state and the canonical library.

Candidate membership is not treated as visual proof. Variant C still requires a localized reference/test difference and may return UNKNOWN/UNRESOLVED.

## Evaluation cases versus dry-run cases

The confirmed evaluation subset contains 19 images across four confirmed case groups: missingpart-A01, missingpart-B01, wrongpart-B01, and correct-control. It contains all six canonical views for each error group and one back-view control.

The dry-run demonstration uses one front-view image for each of six named cases: missingpart-A01, missingpart-B01, extrapart-A01, wrongpart-A01, wrongpart-B01, and correct-control. These packages demonstrate request construction only. Extrapart-A01 and wrongpart-A01 remain `needs_second_review`; their packages do not convert them into confirmed evaluation Ground Truth and contain no human target label.

Dry-run packages live under `analysis/vision_prompt_ab/packages/<variant>/<case>/` and contain `prompt.txt` plus `request_metadata.json`. Metadata records test/reference paths, expected state, schema hash, prompt hash, candidate IDs where applicable, and a future result contract. It contains no API key or confirmed target identity.

## Safety gates and budget

`scripts/run_affected_part_prompt_ab.py` defaults to dry-run, does not load `.env`, imports no API client, and has no network code. `--execute-api` without `--confirm-cost` is rejected. Even with both flags, execution remains intentionally disabled until the user authorizes cost and an audited Vision adapter is connected to the existing project preflight, authentication, request-budget, and no-secret-logging gates.

Dry-run budget: 6 cases × 3 variants = 18 estimated requests. This is an estimate only; no request was sent.

## Evaluation

Future normalized results use: variant, case_id, view_angle, request_id, parsed_output, latency, token usage when available, predicted_part_ids, confidence, and error_type. `scripts/evaluate_affected_part_prompt_ab.py` uses the same evaluator as the baseline and reports each metric plus delta versus baseline.

Primary success criteria for Variant C:

1. Exact Set Match does not decline.
2. At-least-one recall improves.
3. False-confident identity rate at 0.80 declines materially.
4. A moderate increase in UNKNOWN is acceptable when it replaces a high-confidence wrong identity.

Verifier reporting will include acceptance, conflict, unresolved, and wrong-identity-escaped counts. Verifier thresholds must not be relaxed to improve acceptance. For missingpart-A01, UNKNOWN is a safety improvement over confidence-0.95 `EYE_BALL`, but it is not an identity-accuracy success.

## Recommended next experiment

After explicit user authorization, run the fixed 18-request A/B using the current schema and frozen packages, normalize outputs, evaluate all three variants, and manually audit missingpart-A01 plus every verified-but-wrong escape. Do not start Phase 2B until affected-part identity safety improves and the verifier evaluation is acceptable.

Offline verification completed with 17 focused Prompt A/B tests, 19 verifier/regression tests, and 172 full-suite tests plus 19 subtests passing. No API request was made.
