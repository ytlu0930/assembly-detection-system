# missingpart-A01 semantic trace

Date: 2026-08-08
Mode: offline artifact/source audit only
API requests during this audit: 0

## Executive conclusion

`FIRST_DIVERGENCE_LAYER = Parsed Vision JSON / model prediction`

The canonical pipeline never transforms `PIN_RED_SHORT` into `EYE_BALL`. The selected parsed Vision artifact already says that the missing part is `EYE_BALL` with confidence `0.95`. Every downstream layer preserves that ID and renders its valid aliases (`眼睛`, `white ball with black pupil`). GPT Image 2 then faithfully executes the supplied eye-ball prompt.

Root-cause classification: **A. Vision prediction error**.

There is a secondary validation gap: the pipeline treats a known taxonomy ID plus high model confidence as a reliable identity, without verifying that identity against the test/reference delta or an independent affected-parts verifier. This gap allows the original prediction error to propagate, but it is not evidence of an alias/SOP/Prompt identity conversion bug.

`PHASE_2B_RECOMMENDATION = BLOCK`

## Inputs and separation of truth domains

### Online-inference artifact

`logs/current_parsed_json/model03_step03_missingpart-A01_front_01_parsed_20260701_160358_542567.json`

This artifact is a saved model prediction. Its filename metadata contains `target_part=A01`, but the current online pipeline does not interpret `A01` as a canonical part ID.

### Human Ground Truth

Human review identifies `missingpart-A01` as `PIN_RED_SHORT` / red short rod. Evidence is recorded separately in:

- `analysis/affected_parts_review_template.csv`: the exact front case is confirmed as `PIN_RED_SHORT`.
- `tests/evaluate_vision_part_identification.py`: `missingpart-A01` expects `PIN_RED_SHORT`.
- `docs/affected_parts_annotation_guide.md` and `docs/vision_part_failure_analysis_20260701.md`: the same mapping is documented.

`data/ground_truth.csv` records the formal error class and filename target code `A01`, but does not carry the semantic affected-part ID in its current columns. `ground_truth/model03/step03.json` is an expected assembly inventory containing both multiple `PIN_RED_SHORT` and multiple `EYE_BALL` entries; it does not say which one differs in this particular test image.

The human mapping is evaluation/review truth. It was not injected into this inference trace and must not be silently used as online inference input.

## A. Parsed Vision JSON

| Field | Value |
|---|---|
| `overall_error_type` | `missingpart` |
| Detected `part_id` | `EYE_BALL` |
| Description | `The center eye ball is missing from the test image.` |
| Summary | `The test image is missing the center eye ball part compared to the reference image.` |
| Confidence | `0.95` |
| Affected-parts field | Not present in this Vision schema/artifact |
| Structured evidence field | Not present; description/summary are the only textual evidence |
| `eye` / `ball` / `pupil` | `eye ball` appears in description and summary; `pupil` does not appear |
| `red` / `pin` / `PIN_RED_SHORT` | None appear in `model_response` |

The only `A01` indication is filename metadata (`file_info.target_part=A01`). It is not a semantic prediction and is not resolved to `PIN_RED_SHORT` by the runtime.

This is the first point where the online inference disagrees with human Ground Truth. There is no earlier pipeline artifact containing a correct `PIN_RED_SHORT` prediction that later becomes an eye.

## B. `results.json`

Path: `output/single_runs/model03_step03_missingpart-A01_front_01_openai_e2e/results.json`

### ErrorReport

| Field | Value |
|---|---|
| `part_id` | `EYE_BALL` |
| `description` | `The center eye ball is missing from the test image.` |
| `expected_value` | `null` |
| `actual_value` | `null` |
| `evidence` | Same Vision description |
| `confidence` | `0.95` |
| `unresolved` | `false` |
| `overall_error_type` | `missingpart` |

`utils/error_report_adapter.py:83-104` copies `detected_parts[].part_id` directly. It does not consult an alias table and does not change `EYE_BALL` to or from another ID. Therefore the normalization result is still `EYE_BALL`.

### Localization enrichment

`pipeline_smoke_test.py:253-277` uses the already-selected ErrorReport `part_id` to:

1. look up the same ID in expected state;
2. choose a readable alias from `config/part_library.json`;
3. localize that predicted part in the correct reference image for `missingpart`.

For `EYE_BALL`, this produces:

- canonical ID: `EYE_BALL`
- aliases: `eye`, `white ball with black pupil`, `眼睛`
- localization prompt: `white ball with black pupil`
- expected-state match: `{part_id: EYE_BALL, color: WHITE, position: TOP, orientation: HORIZONTAL}`
- localization role: `reference_missing_part_location`

The expected-state lookup at `pipeline_smoke_test.py:179-186` searches by the predicted ID. Because `EYE_BALL` legitimately exists in the full expected assembly inventory, the lookup confirms its attributes but does not verify whether an eye is actually the missing delta. Expected state therefore enriches the wrong prediction; it does not originate the wrong ID.

The selected localization detection is labelled `ball` with detection score about `0.1788`; the pipeline already warns that localization is unreliable. Localization follows the wrong semantic query and cannot independently correct its identity.

## C. `correction_sop.json`

Path: `output/single_runs/model03_step03_missingpart-A01_front_01_openai_e2e/correction_sop.json`

### Package identity

- `target_parts`: `["EYE_BALL"]`
- `part_identity_reliable`: `true`
- `requires_manual_review`: `true` (caused by unreliable localization)
- repair scope: `local`

The notable logic is `correction_sop_generator.py:263-299`:

- it takes `part_id` directly from the ErrorReport;
- `_find_expected_part()` at lines 882-889 looks up that same ID;
- `_part_identity_reliable()` at lines 849-859 checks only that the ID is in the part library, is not unknown, and confidence is at least the threshold;
- `_part_name()` at lines 861-879 renders a configured name for the same ID.

Because `EYE_BALL` is valid and confidence is 0.95, the SOP marks the wrong prediction as identity-reliable and renders it as `眼睛`. No `PIN_RED_SHORT → EYE_BALL` mapping occurs here.

### SOP steps

| Step | Action | Part identity / affected parts |
|---:|---|---|
| 1 | `prepare_part` | `EYE_BALL`; affected `[EYE_BALL]`; prepare `眼睛` |
| 2 | `locate_installation_point` | `EYE_BALL`; affected `[EYE_BALL]` |
| 3 | `insert_part` | `EYE_BALL`; affected `[EYE_BALL]` |
| 4 | `verify_local_result` | `EYE_BALL`; affected `[EYE_BALL]` |
| 5 | `manual_review` | no target ID |
| 6 | `compare_reference` | generic whole assembly |
| 7 | `retake_photo` | generic whole assembly |
| 8 | `rerun_detection` | generic whole assembly |
| 9 | `finish` | generic whole assembly |

`white ball with black pupil` has already appeared in `results.json` as the localization alias. Within the SOP artifact itself, the ID remains `EYE_BALL` and the localized name is `眼睛`; the full English visual phrase is not introduced by SOP rendering.

## D. `step_prompts_v2.json`

Path: `output/single_runs/model03_step03_missingpart-A01_front_01_openai_e2e/step_prompts_v2.json`

`step_prompt_builder_v2.py:361-392` reads `target_part_id` from each SOP step. `_part_visual_name()` at lines 776-784 maps the same canonical ID to an English alias. `_compose_prompt()` at lines 630-669 inserts that alias into the prompt without changing identity.

| Image task | SOP action | Canonical target | Rendered visual target | Identity changed? |
|---:|---|---|---|---|
| 1 | `prepare_part` | `EYE_BALL` | `white ball with black pupil` | No; alias rendering only |
| 2 | `locate_installation_point` | `EYE_BALL` | `white ball with black pupil` | No |
| 3 | `insert_part` | `EYE_BALL` | `white ball with black pupil` | No |
| 4 | `verify_local_result` | `EYE_BALL` | `white ball with black pupil` | No |
| 5 | `compare_reference` | `null` | `target construction component` | No part-specific identity |

The first image task is the first final image prompt containing the complete phrase `white ball with black pupil`, but it is not the first semantic divergence. It is a faithful textual rendering of the already-wrong `EYE_BALL` ID.

## E. `generation_manifest_v2.json`

Path: `output/single_runs/model03_step03_missingpart-A01_front_01_openai_e2e/generated_steps_v2/generation_manifest_v2.json`

- provider: `openai`
- model: `gpt-image-2`
- executed task: sequence 1 / `prepare_part`
- operation: `images.edit`
- status: `success`
- attempts/request count: 1
- prompt length: 1471 characters
- generated output: `standalone/sequence_01_sop_01_prepare_part.png`

The manifest stores `source_prompts_json` and prompt length, not the full prompt. `step_image_generator_v2.py:243-250` passes `raw["prompt_en"]` from `step_prompts_v2.json` directly to the provider. It contains no part-identity remapping. GPT Image 2 therefore received an explicit instruction to prepare a `white ball with black pupil` and generated an eye-like part accordingly.

`GPT_IMAGE_2_RESPONSIBILITY = NO` for the identity mismatch. The provider/model executed the supplied upstream target. Image fidelity and quality remain separate review questions, but the red-pin-versus-eye error predates image generation.

## Layer-by-layer identity table

| Layer | Input Part | Output Part | Identity Changed? | Evidence |
|---|---|---|---|---|
| Human Ground Truth (offline evaluation) | Observed case | `PIN_RED_SHORT` | N/A | Confirmed affected-parts review; not inference input |
| Parsed Vision model response | Test + reference + expected-state context | `EYE_BALL` | **Yes relative to human truth; first divergence** | `detected_parts[0].part_id`, description, summary |
| ErrorReport adapter | `EYE_BALL` | `EYE_BALL` | No | `error_report_adapter.py:83-104` direct copy |
| Expected-state enrichment | `EYE_BALL` | `EYE_BALL`, WHITE/TOP/HORIZONTAL | No | Lookup by predicted ID; both eye and red pins exist in inventory |
| Localization prompt | `EYE_BALL` | `white ball with black pupil` | No | Valid alias from part library |
| Correction SOP | `EYE_BALL` | `EYE_BALL` / `眼睛` | No | Direct ErrorReport ID plus alias rendering |
| Step Prompt Builder | `EYE_BALL` | `white ball with black pupil` | No | `_part_visual_name()` alias rendering |
| Step Image Generator | Eye prompt | Same eye prompt | No | Passes `raw["prompt_en"]` directly |
| OpenAI GPT Image 2 | Eye prompt | Eye-like image | No semantic substitution shown | Manifest success and produced artifact |

## Search findings

- `config/part_library.json` has separate, correct entries:
  - `EYE_BALL`: `eye`, `white ball with black pupil`, `眼睛`
  - `PIN_RED_SHORT`: `short red cylinder stick`, `紅色短圓柱`, `紅色短棒`
- No alias maps `PIN_RED_SHORT` to an eye or maps `EYE_BALL` to a red pin.
- `config/normalizer_dict.json` handles position/orientation/error-type normalization; it is not used to replace this part ID in the canonical trace.
- `utils/taxonomy.py` defines error-type taxonomy and is not the source of this part substitution.
- The successful isolated OpenAI smoke script hardcodes its own SOP input as `affected_parts=["PIN_RED_SHORT"]` at `scripts/run_openai_image_smoke_test.py:54-60`. It bypasses Parsed Vision → ErrorReport → SOP identity inference, which explains why its result targeted the correct red pin.
- The canonical `main.py` chain is `process_one()` → `CorrectionSOPGenerator` → `StepPromptBuilderV2` → `StepImageGeneratorV2` (`main.py:132-174`). It starts from the saved parsed model response and therefore inherits `EYE_BALL`.

## Root-cause classification

**A. Vision prediction error**

Evidence:

1. The earliest inspected inference artifact already predicts `EYE_BALL`.
2. It contains no `PIN_RED_SHORT`, `red`, or `pin` semantic identity.
3. The adapter copies that ID unchanged.
4. Taxonomy and aliases keep eye and red pin separate.
5. SOP and Prompt Builder retain the ID and only render valid names.
6. GPT Image 2 receives the eye prompt verbatim.

Not selected:

- **B adapter mapping error:** no mapping occurs.
- **C taxonomy/canonical alias error:** aliases are correct and separate.
- **D correction SOP mapping error:** SOP preserves `EYE_BALL`.
- **E Step Prompt Builder error:** builder faithfully renders the `EYE_BALL` alias.
- **F multiple transforming causes:** no second identity-transforming cause was found. A missing verification gate contributes to propagation but does not perform a second transformation.

## Minimal correction proposal (not implemented)

Do not hardcode `missingpart-A01 → PIN_RED_SHORT` into online inference. Keep human Ground Truth evaluation-only.

Recommended order:

1. **Add an expected-state/reference affected-parts verifier before SOP generation.** Compare test/reference evidence to candidate expected parts and return verified, rejected, or unresolved identity with evidence. The natural integration point is around `pipeline_smoke_test.py:397-424`, before localization and before `CorrectionSOPGenerator` consumes `results.json`.
2. **Make SOP identity reliability depend on verification, not only ID membership and model confidence.** Future change location: `correction_sop_generator.py::_part_identity_reliable` (lines 849-859) and the caller around lines 263-299. A high-confidence valid ID must not automatically mean the identity is correct.
3. **Carry verification fields without remapping.** Future change location: `utils/error_report_adapter.py:83-104`; preserve fields such as candidate identity, verifier result, evidence source, and unresolved reason. Do not substitute human labels inside the adapter.
4. **Use correct-reference verification and localization evidence as independent checks.** If the predicted missing part cannot be supported by a reliable reference/test delta, stop image generation and require review.
5. **Run a controlled Vision Prompt A/B** after the verifier/evaluation path is established. This specific prediction error justifies an A/B focused on affected-part identity and reference-difference reasoning, but production Prompt must remain unchanged until measured.
6. **Consider Schema vNext later**, adding structured `expected_part`, `observed_part`, evidence/source, and identity uncertainty. A production Schema change is not required for the immediate containment fix and should not precede evidence from A/B evaluation.
7. Add regression tests where Vision predicts a valid-but-wrong ID with high confidence; the verifier/SOP gate must mark it unresolved and block paid generation.

### Answers on required components

- Vision Prompt A/B needed: **Yes**, as a controlled later evaluation, not the first containment action.
- Production Schema modification needed now: **No**. Schema vNext may improve evidence/uncertainty after evaluation.
- Expected-state verifier needed: **Yes**, but it must compare candidate identity to actual test/reference evidence rather than merely find the candidate in a full assembly inventory.
- Minimal files for a future fix: `pipeline_smoke_test.py`, `correction_sop_generator.py`, `utils/error_report_adapter.py`, and their tests. `step_prompt_builder_v2.py` does not need an identity-mapping fix based on this trace.

## Pytest environment-isolation audit

The failure is an **environment isolation bug in the test setup/interface**.

Evidence:

- `tests/test_openai_image_smoke_cli.py:9` deletes `OPENAI_API_KEY` from `os.environ`.
- `main()` then unconditionally calls `load_dotenv(ROOT / ".env", override=True)` at `scripts/run_openai_image_smoke_test.py:44`.
- The real local key is therefore loaded again after the test deletes it.
- The test expects `api_key_configured is False`, but sees the actual local configuration.
- The socket mock proves no network call occurred, and the key value was not printed, but the test still improperly depends on a user's real `.env`.

Suggested test-only isolation fix:

1. Patch `scripts.run_openai_image_smoke_test.load_dotenv` to a no-op before calling `main()`.
2. Clear all relevant variables with `monkeypatch`: `OPENAI_API_KEY`, `STEP_IMAGE_PROVIDER`, `OPENAI_IMAGE_MODEL`, `ENABLE_OPENAI_IMAGE_API`, and `CONFIRM_OPENAI_IMAGE_API_EXECUTION`.
3. Retain the network mock and redaction assertions.
4. Preferably evolve the CLI to accept an injected environment mapping or env-file path so tests never touch the project `.env`; alternatively, load `.env` only on an explicitly authorized execute path. This is a proposal only.

No real key needs to be read, and no network call should be possible in this test.

## Final decision

The canonical semantic target is wrong before SOP construction. Running three or five sequential paid image tasks would consistently elaborate the wrong eye correction and spend additional budget without validating the intended red-pin repair.

`PHASE_2B_RECOMMENDATION = BLOCK`

Unblock only after an offline verifier or corrected, evaluated Vision artifact identifies the target part with evidence, and a human confirms that the canonical generated prompt matches the intended correction. No API call, Prompt/Schema/Ground Truth change, source-code edit, Phase 2B run, batch run, or Git write operation was performed during this audit.

## Implementation follow-up

The proposed containment is now implemented in `utils/affected_part_identity_verifier.py` and integrated into the canonical pipeline. ErrorReports carry verification fields; the SOP, prompt builder, and image generator fail closed for `conflict`, `uncertain`, and `unresolved` identities.

The regression keeps the original `EYE_BALL` prediction unchanged, supplies independent contrastive evidence with equal eye counts in test/reference, and obtains `identity_status=conflict`, `verified_part_id=null`, manual review, zero eye image tasks, and zero provider calls. It does not replace the identity with the human `PIN_RED_SHORT` answer.

Offline validation after implementation: 155 passed, 19 subtests passed. Formal Prompt, Schema, Ground Truth, and source images remain unchanged. Phase 2B remains blocked.
