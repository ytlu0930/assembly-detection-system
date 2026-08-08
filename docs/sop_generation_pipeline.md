# Correction SOP and Step Images

`utils.correction_sop_generator.generate_correction_sop` accepts ErrorReport lists and expected state. It returns `repair_scope`, `source_step_id`, `rollback_to_step`, `target_parts`, and ordered `steps`. Every step contains `step_number`, action, instruction, visual instruction, affected/preserved parts, reference step, and whether an image is required.

Small errors use local repair. Missing parts use locate → insert → verify; extra parts use locate → remove → verify; a verified two-part wrongpart pair uses locate → remove → swap → verify. Explicit rollback metadata selects partial rollback; critical errors select full rollback. Ground-truth CSV is not read.

`utils.step_prompt_builder` requires the background, camera, lighting, vehicle, colors, counts, shapes, and non-target parts to remain unchanged, while showing a red action arrow. Every step still references the correct image and may chain the previous output. `utils.step_image_generator` isolates provider calls and records prompt, input paths, output, status, duration, warning, error, and metadata.

The formal provider is OpenAI Image API / GPT Image 2 (`gpt-image-2`) using Images Edit (`client.images.edit`, `/v1/images/edits`). Runtime still defaults to `MockStepImageProvider`. The guarded adapter is implemented but disabled by default; it requires `ENABLE_OPENAI_IMAGE_API=true`, `CONFIRM_OPENAI_IMAGE_API_EXECUTION=true`, and `execute_api=True`. A disabled or failed provider stops later image requests while text SOP and other pipeline outputs remain available. The first edit uses the test image; each later successful step uses the prior output, and every step retains the same correct reference.

Per-step instruction images are the core project output. `generate_sop_flowchart` is only a compact overview and consumes the structured SOP, never raw Vision JSON.

As of 2026-08-08, root `correction_sop_generator.py` is canonical. Its JSON exposes `repair_scope`, `source_step_id`, `rollback_to_step`, `target_parts`, and `steps`, while retaining `correction_plan` aliases for A-version compatibility. `step_prompt_builder_v2.py` is the sole formal prompt builder. `instruction_book_generator.py`, not the deprecated flowchart, is the formal visual output.
