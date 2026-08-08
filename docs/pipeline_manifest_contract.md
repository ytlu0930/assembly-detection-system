# Pipeline Manifest Contract

`main.run_pipeline(...)` returns `PipelineManifest` and writes it to `<case_output>/pipeline_manifest.json`.

| Field | Type | Meaning |
|---|---|---|
| `final_instruction_path` | string or null | Final `assembly_instruction_book.png`; primary UI artifact |
| `results_path` | string or null | Multi-ErrorReport and Localization result JSON |
| `correction_sop_path` | string or null | Canonical structured correction SOP JSON |
| `step_prompts_path` | string or null | V2 prompt package JSON |
| `generated_steps_dir` | string or null | V2 generated-step root and generation manifest |
| `generated_step_image_paths` | list[string] | Existing standalone, assembly, and comparison outputs |
| `annotated_image_path` | string or null | First available Localization annotation |
| `status` | string | `running`, `success`, `partial`, or `failed` |
| `warnings` | list[string] | Recoverable localization/image/review issues |
| `errors` | list[string] | Fatal pipeline errors |
| `manual_review_required` | boolean | SOP-level manual-review flag |
| `image_provider` | string | `mock` or `openai` |
| `execute_image_api` | boolean | Whether all code-level execution conditions were met |
| `stages` | list[StageRecord] | Stage status, artifact path, timing, and error data |

The UI consumes `final_instruction_path` and may display annotation, warnings, and status. Localization failures remain warnings and preserve text SOP. Image failures remain warnings; the instruction book uses textual/image placeholders. Fatal parsed-input, SOP, prompt, or book errors set `status=failed` and populate `errors`.
