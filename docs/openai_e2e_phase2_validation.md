# OpenAI GPT Image 2 canonical E2E — Phase 2A validation

Date: 2026-08-08

## Result

Phase 2A is a **technical PASS**: the canonical `main.py` pipeline completed one OpenAI Platform GPT Image 2 Images Edit request, validated the returned PNG, recorded it in the V2 generation manifest, and embedded it in the instruction book. Semantic image quality remains under human review and is not marked PASS here.

The top-level pipeline status is `partial`, not because a stage or API call failed, but because localization was below its reliability threshold and the case requires manual review.

## Execution scope

- Case: `model03_step03_missingpart-A01_front_01`
- Parsed JSON: `logs/current_parsed_json/model03_step03_missingpart-A01_front_01_parsed_20260701_160358_542567.json`
- Isolated output: `output/single_runs/model03_step03_missingpart-A01_front_01_openai_e2e/`
- Provider: `openai`
- Model: `gpt-image-2`
- Operation: `images.edit`
- Quality: `low`
- Size: `1536x1024`
- Allowed image tasks: 1
- Allowed requests: 1
- Actual requests: 1
- Retries: 0 (one attempt/request recorded)
- Phase 2B: not executed
- Batch: not executed

The run omitted `--overwrite` and used a previously nonexistent output directory. It also omitted `--image-continue-on-error`.

## Request inputs

- Pipeline test image: `input/missingpart/model03_step03/model03_step03_missingpart-A01_front_01.jpg`
- Correct reference image: `input/normal/model03_step03/model03_step03_correct-01_front_01.jpg`
- Executed SOP step: 1, `prepare_part` / `準備正確零件`
- Prompt target: `white ball with black pupil`
- Request source/base for this standalone task: the correct reference image (the task has no assembly base and uses the reference as its image input)
- Correct reference: the same correct reference image
- Output: `generated_steps_v2/standalone/sequence_01_sop_01_prepare_part.png`

Prompt objective:

> Prepare one correct white ball with black pupil. Produce a single isolated part on a clean white background, matching reference geometry, material, color, and hole pattern.

The canonical SOP contains 9 total steps. The prompt package contains 5 image tasks: `prepare_part`, `locate_installation_point`, `insert_part`, `verify_local_result`, and `compare_reference`. Phase 2A intentionally selected only the first task, so it made exactly one request. The current case therefore does not have only three total image tasks; any future Phase 2B command must use the actual five-task package or deliberately cap it.

## Manifest evidence

Generation manifest:

`output/single_runs/model03_step03_missingpart-A01_front_01_openai_e2e/generated_steps_v2/generation_manifest_v2.json`

- `provider`: `openai`
- `image_model`: `gpt-image-2`
- `execute_api`: `true`
- `requested_task_count`: 1
- `successful_task_count`: 1
- `failed_task_count`: 0
- `skipped_task_count`: 0
- Task status: `success`
- Task operation: `images.edit`
- Task attempts/request count: 1
- Task duration: 115.722761 seconds

Pipeline manifest:

`output/single_runs/model03_step03_missingpart-A01_front_01_openai_e2e/pipeline_manifest.json`

- Image generator stage: `success`, 115.732210 seconds
- Instruction-book stage: `success`, 2.112116 seconds
- Total pipeline duration: 134.404966 seconds
- `generated_step_image_paths` contains the real Step 1 output
- `final_instruction_path` points to `assembly_instruction_book.png`
- Errors: none

## Artifact validation

### Generated Step 1 image

- Exists: yes
- Path: `output/single_runs/model03_step03_missingpart-A01_front_01_openai_e2e/generated_steps_v2/standalone/sequence_01_sop_01_prepare_part.png`
- File size: 1,158,848 bytes
- Pillow decodable: yes
- Format: PNG
- Dimensions: 1536 × 1024
- Mode: RGB

### Instruction book

- Exists: yes
- Path: `output/single_runs/model03_step03_missingpart-A01_front_01_openai_e2e/assembly_instruction_book.png`
- File size: 1,068,247 bytes
- Pillow decodable: yes
- Format: PNG
- Dimensions: 1800 × 5644
- `pipeline_manifest.final_instruction_path` matches this file: yes
- Step 1 uses the actual GPT Image 2 output: yes
- Steps not executed in Phase 2A use explicit missing-step-image fallback panels: yes
- A fallback panel was not represented as GPT Image 2 output: yes

## Human image-quality review

`manual_review_pending=true`

Codex performed only artifact/structure checks. The following semantic judgments remain for a human reviewer:

| Criterion | Status |
|---|---|
| `target_edit_correct` | pending |
| `target_position_correct` | pending |
| `non_target_preservation` | pending |
| `geometry_preservation` | pending |
| `view_consistency` | pending |
| `background_consistency` | pending |
| `hallucinated_parts` | pending |
| `instruction_value` | pending |

Visual inspection shows a clean standalone white eyeball-like part on a white background, consistent with the generated prompt. However, the pipeline's localization target (`white ball with black pupil`) may not match the intended missing red-pin correction described by the earlier isolated smoke test. This is precisely why the semantic result must remain under manual review.

## Localization warning

`localization_warning=true`

Recorded warnings:

- `Localization 分數未達可靠門檻，bbox 僅供參考。`
- `此案例需要人工確認。`

The HF Hub also emitted an unauthenticated-download warning while loading localization weights. No `HF_TOKEN` was added because this warning did not block the run and credential changes were out of scope.

## Offline validation

- `compileall`: PASS
- Pytest: 139 passed, 19 subtests passed, 1 failed
- Failed test: `tests/test_openai_image_smoke_cli.py::test_smoke_cli_defaults_to_dry_run_without_network`
- Failure cause: the test deletes `OPENAI_API_KEY` from process environment, but the current dry-run script reloads the configured project `.env`; expected `api_key_configured=false`, observed `true`.
- The failed test verified that no network connection was made.
- No source/test fix was made during this Phase 2A run.
- Output is ignored by Git via the repository's `output/` rule.

## Phase decision

- Phase 2A: **technical PASS**
- Image-quality review: pending
- Phase 2B: not executed
- Batch: not executed

Do not proceed to Phase 2B until a human confirms the target-part mismatch is understood and accepts or corrects the upstream localization/SOP target.

If explicitly approved afterward, the requested three-request Phase 2B command is:

```powershell
.\venv\Scripts\python.exe main.py `
  --parsed-json "logs\current_parsed_json\model03_step03_missingpart-A01_front_01_parsed_20260701_160358_542567.json" `
  --output-dir "output\single_runs\model03_step03_missingpart-A01_front_01_openai_full_e2e" `
  --generate-images `
  --image-provider openai `
  --execute-image-api `
  --confirm-cost `
  --allow-manual-review `
  --image-quality low `
  --image-size 1536x1024 `
  --image-max-tasks 3 `
  --image-max-requests 3
```

This command is documented only; it was not executed.

## Post-validation semantic containment

The historical Phase 2A API/transport result remains a technical PASS, but its semantic target is a FAIL: the saved Vision prediction was `EYE_BALL`, while separate human evaluation identifies the affected part as `PIN_RED_SHORT`. GPT Image 2 was not responsible; it followed the supplied eye prompt.

`AffectedPartIdentityVerifier` is now integrated before SOP generation. The missingpart-A01 regression classifies equal test/reference eye evidence as `conflict`, leaves `verified_part_id=null`, requires manual review, produces no eye repair prompt, and makes zero image-provider calls. The earlier pytest `.env` isolation failure is also fixed; the full offline suite now reports 155 passed and 19 subtests passed.

`PHASE_2B_RECOMMENDATION = BLOCK`
