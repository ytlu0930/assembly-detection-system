# ROI Direct vs Checklist Experiment

## Research question

Can a component-level ROI checklist improve affected-part identity over direct ROI classification, especially Exact Set Match, all-parts recall, Part F1, and false-confident errors, while preserving the production verifier's zero wrong-identity escape behavior?

This document describes an experiment framework and preflight only. No Vision request, GPT Image request, or Phase 2B execution occurred while preparing it.

## Experimental design

The fixed confirmed cases are `missingpart-A01`, `missingpart-B01`, and `wrongpart-B01`. Each case has two methods:

- `roi_direct`: directly classifies affected identities from frozen full images, ROI crops, bbox evidence, reduced candidates, view/error metadata, and expected-state summary.
- `roi_checklist`: emits one observation record per reduced candidate. A deterministic rule engine derives the affected-part conclusion afterward.

The six logical requests are `EXP-001` through `EXP-006`, ordered by case and then Direct/Checklist. Ground Truth is absent from prompts and packages. The evaluator refuses to load confirmed labels until all six response artifacts and all six physical reservations are frozen.

## Frozen ROI inputs

Packages consume the existing `analysis/roi_identity_poc/packages/*.json` artifacts without rerunning localization or selecting new boxes. The preflight records source-package, source-image, prompt, schema, runner, and copied-ROI SHA-256 values. Any later change fails preflight.

Candidate sets remain exactly those produced by the ROI PoC:

| Case | Candidates | Count | Reduction | Localization score | Paired ROI |
|---|---|---:|---:|---:|---|
| missingpart-A01 | PIN_RED_SHORT, PIN_YELLOW, ROD_GREEN_LONG, LINK_RED_3HOLE, LINK_BLUE_5HOLE | 5 | 66.67% | 0.7907 | no |
| missingpart-B01 | ROD_GREEN_LONG, PIN_YELLOW, WHEEL_BLUE_LARGE, WHEEL_BLUE_SMALL, PIN_RED_SHORT | 5 | 66.67% | 0.8028 | no |
| wrongpart-B01 | LINK_BLUE_5HOLE, PIN_RED_SHORT, LINK_GREEN_5HOLE, ROD_GREEN_LONG, LINK_RED_3HOLE, PIN_YELLOW | 6 | 60.00% | 0.8541 | yes |

These IDs are not manually injected labels: their provenance is the frozen expected-state + part-library + local ROI evidence pipeline. The package audit rejects review filenames, confirmed-label keys, and evaluation-label sources.

## Direct classification

The experiment prompt is `experiments/prompts/vision_roi_direct_identity.txt`; its output contract is `experiments/schema/vision_roi_direct_output_schema.json`. A part must be in the package candidate list or be an explicit UNKNOWN value. Wrong-part output may contain multiple affected identities when paired evidence supports a swap. The model returns short observable evidence summaries, not chain-of-thought.

## Checklist verification and rule engine

The checklist prompt and experiment-only schema are `experiments/prompts/vision_roi_checklist_verification.txt` and `experiments/schema/vision_roi_checklist_output_schema.json`. They do not modify the production Prompt or Schema.

Every candidate must appear exactly once with presence, count, spatial, appearance, status, confidence, and evidence fields. Runtime membership validation rejects outside IDs, missing checks, and duplicate checks. A response sanitizer removes accidental top-level schema metadata before validation; it does not repair semantic content and does not trigger retry.

`utils/roi_checklist_rule_engine.py` applies deterministic rules:

- Missing: Reference present and Test absent, or Reference count greater than Test.
- Extra: Test present and Reference absent, or Test count greater than Reference.
- Wrong/swap: paired presence/count/appearance mismatch; paired ROI support is mandatory.
- Position: `spatial_match=false`.
- UNCERTAIN, missing checks, membership violations, missing paired evidence, or no supported identity fail closed to manual review.

## Request safety

The final preflight run is `analysis/roi_direct_vs_checklist/run_20260809_preflight`, UUID `d9d0c3f0-7a57-41ed-872c-b2ab42f4db97`.

- Logical request limit: 6
- Physical request hard ceiling: 6
- SDK automatic retry: 0 (`max_retries=0`)
- Schema-validation retry: disabled
- PID-aware exclusive lock: verified
- Reservation and physical counter persistence: before transport
- Completed response: skipped on resume
- Reserved/failed request: fail closed; explicit retry would be required, but this six-request command exposes no retry flag
- Seventh request: blocked before transport
- Old 31/18 and targeted ledgers: never read or reused
- API credentials: readiness booleans only; keys are never persisted or printed

## Evaluation

`scripts/evaluate_roi_direct_vs_checklist.py` joins exact-image confirmed labels only after response freeze. Both methods report Exact Set Match, at-least-one/all-parts recall, part precision/recall/F1, unknown rate, false-confident identity/case rates, verifier acceptance/conflict/unresolved, wrong-identity escaped/blocked, correct-identity blocked, and manual-review rate. A zero denominator is represented as null/N/A, never fabricated as zero.

Checklist-level evaluation treats confirmed affected candidate IDs as `GT MISMATCH`; other candidate checks are `GT NORMAL`. FAIL predicts mismatch and PASS predicts normal. UNCERTAIN is excluded from the resolved-only 2x2 confusion matrix and reported separately.

### Experiment-only response normalization

The completed run returned three Checklist responses with a consistent response-contract drift. Original schema compliance remains 0/3. `utils/roi_checklist_response_normalizer.py` provides post-hoc offline compatibility for this experiment only: `results` becomes `checks`, known status aliases become `status`, string confidence becomes a bounded float, and absent evidence text receives an explicit non-semantic absence marker required by the experiment schema. Exact candidate membership is enforced and malformed or ambiguous input fails closed.

The normalizer does not modify the production parser, Prompt, or Schema; it has no Ground Truth input and does not change identities or infer PASS/FAIL. The original response files and frozen snapshots remain immutable. Normalized analysis compliance is 3/3 and semantic fields are identical to the prior label-free recovery; this does not retroactively make the original responses schema-valid.

## Deterministic annotations and thesis outputs

`utils/deterministic_correction_annotator.py` draws only frozen bbox evidence and rule-engine identities on a copy of the Test image, adds error/confidence/review text, and verifies that the source image hash remains unchanged. It never synthesizes parts or uses GPT Image.

Post-response outputs are generated under the run's `figures/` and `thesis_tables/` directories:

- `checklist_confusion_matrix.png/.csv`
- `method_comparison_metrics.png`
- three four-panel thesis case figures
- `roi_direct_vs_checklist_metrics.csv`
- `roi_direct_vs_checklist_cases.csv`
- `checklist_component_results.csv`
- `research_method_evolution.csv`
- `request_efficiency.csv`

A verified consolidated copy and SHA-256 manifest are stored under `thesis_artifacts/`. The narrative synthesis is `docs/thesis_experiment_summary.md`.

Matplotlib is used when available. The current venv lacks it, so an OpenCV deterministic high-resolution fallback is included and tested; seaborn is not required.

## Success criteria and limitations

Checklist is promising only if Exact Match, at-least-one recall, and all-parts recall are no worse than Direct; Part F1 is higher; false-confident identity at 0.80 is lower; and wrong identity escaped verifier remains zero. More UNCERTAIN results are acceptable only as an explicit safety/manual-review tradeoff.

This is a three-case targeted experiment, not a production-accuracy estimate. The ROI PoC still contains cross-view false positives and marks every package for manual review. Even a successful Checklist result does not authorize production Prompt/Schema changes, GPT Image execution, or Phase 2B.
