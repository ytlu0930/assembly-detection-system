# Affected-Part Identity Verifier

## Purpose

`utils/affected_part_identity_verifier.py` is the evidence gate between ErrorReport localization and Correction SOP generation. It prevents a valid taxonomy ID with high Vision confidence from being treated as a confirmed repair target unless test/reference evidence supports the reported error relation.

The verifier does not read human case-to-part mappings, does not hardcode `missingpart-A01`, and does not map `EYE_BALL` to `PIN_RED_SHORT`.

## Result contract

`IdentityVerificationResult` contains:

- `predicted_part_id`
- `verified_part_id`
- `identity_status`
- `identity_confidence`
- `evidence`
- `alternative_candidates`
- `requires_manual_review`

Allowed status values are exactly:

- `verified`
- `conflict`
- `uncertain`
- `unresolved`

The ErrorReport adapter remains backward compatible and carries these values as `identity_status`, `identity_confidence`, `identity_evidence`, `verified_part_id`, and `alternative_candidates` without performing identity substitution.

## Evidence rules

Taxonomy membership, model confidence, and presence in the complete expected inventory are context only. They are never sufficient verification.

The verifier evaluates:

- expected inventory presence and count;
- predicted error relation (`missingpart`, `extrapart`, `wrongpart`, and related types);
- localization evidence for the same candidate in test and correct-reference images;
- distinct localized counts after score filtering and NMS;
- explicit independently produced relation evidence when supplied;
- cross-view consistency when supplied;
- independently scored alternative candidates.

For `missingpart`, verification requires a reliable reference count greater than the test count. Equal or reversed counts conflict with the claim. For `extrapart`, the test count must exceed the reference count. Wrong-part, position, and composite relations require explicit relation or cross-view support; simple presence is not enough.

Alternative candidates are advisory unless the top candidate has independent difference evidence, exceeds the candidate threshold, and clears the runner-up margin. The verifier never selects a candidate merely because it is listed in expected state.

## Canonical integration

The formal flow is now:

```text
Parsed Vision JSON
  -> ErrorReport adapter
  -> predicted-part localization in the primary image
  -> counterpart localization in test/reference
  -> AffectedPartIdentityVerifier
  -> verified ErrorReport
  -> Correction SOP
  -> Step Prompt Builder V2
  -> Step Image Generator V2
```

`pipeline_smoke_test.process_one()` collects both localization sides and writes `identity_verifications` plus the extended ErrorReports to `results.json`. `main.run_pipeline()` accepts an injectable verifier and passes it into this stage.

## Fail-closed behavior

Only `identity_status=verified` with a non-empty `verified_part_id` may create named repair steps.

For `conflict`, `uncertain`, or `unresolved`:

- Correction SOP uses a generic manual-review instruction;
- the predicted part is excluded from `target_parts`;
- `identity_verification_blocked=true`;
- every SOP step has image generation disabled;
- Step Prompt Builder emits no image tasks and `generation_allowed=false`;
- Step Image Generator honors the hard block even when manual review override is requested;
- no image provider is called.

## missingpart-A01 regression

The saved Parsed Vision artifact remains unchanged and still predicts `EYE_BALL` at confidence 0.95. The semantic regression supplies contrastive evidence showing equal eye counts in test and reference. The verifier returns:

- `predicted_part_id=EYE_BALL`
- `identity_status=conflict`
- `verified_part_id=null`
- `alternative_candidates=[]`
- `requires_manual_review=true`

The resulting SOP has no `EYE_BALL` target/image task, the prompt package is blocked and empty, and the recording provider receives zero calls. The verifier does not infer or inject `PIN_RED_SHORT`; that remains separate human evaluation truth.

The canonical pipeline was also rerun with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` into `output/single_runs/model03_step03_missingpart-A01_front_01_identity_verifier_offline_01`. Grounding DINO produced eight distinct `EYE_BALL` candidates in both the test and correct-reference images (selected confidences 0.1763 and 0.1788). Equal counts contradict a `missingpart` claim, so the real offline run likewise returned `conflict`, `verified_part_id=null`, and no alternatives. Its SOP contains only non-generating review/comparison steps; `step_prompts_v2.json` has zero tasks and `generation_allowed=false`; the manifest records provider `mock`, `execute_api=false`, and zero tasks.

## Tests and safety

Tests cover valid-and-correct, valid-but-wrong high-confidence, unknown, insufficient evidence, conflict, missing, extra, wrong-part, and swap/composite cases. The dedicated missingpart-A01 regression hashes the original Parsed JSON before and after execution and confirms zero provider calls.

The OpenAI smoke dry-run no longer reads `.env`; `.env` loading occurs only on an explicitly requested non-dry execution path. The dry-run test clears relevant variables, replaces the dotenv loader with a failing sentinel, and retains the network mock.

No Vision Prompt, Vision Schema, Ground Truth, or source image is modified by this verifier.

## Remaining work

The existing Prompt/Schema A/B framework can be used after this verifier baseline is accepted:

- `baseline_prompt_current_schema`
- `improved_prompt_current_schema`
- `improved_prompt_schema_vnext`

No A/B API run was performed in this task. Phase 2B remains blocked until offline identity evidence is verified and manually reviewed.

## Offline identity baseline and Prompt A/B preparation

The confirmed 25-image baseline now quantifies the upstream risk the verifier contains: Exact Set Match is 8.00%, part-level F1 is 10.53%, and false-confident identity rate is 88.00% at 0.70/0.80/0.90. All 25 predicted identities fall in the 0.90-1.00 confidence bin, whose empirical identity accuracy is only 12.00%.

The 19-image confirmed evaluation subset and three current-schema Prompt variants are documented in `docs/affected_part_baseline_evaluation.md` and `docs/affected_part_prompt_ab_plan.md`. Candidate generation is evaluation-independent and does not use review labels. A six-case × three-variant dry run produced 18 packages and zero API requests. Verifier thresholds were not changed; real Prompt A/B execution and Phase 2B remain blocked pending explicit authorization and evaluation.

Post-integration validation passed 19 verifier/regression tests and the full isolated suite passed 172 tests plus 19 subtests.
