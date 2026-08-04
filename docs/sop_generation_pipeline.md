# Correction SOP and Step Images

`utils.correction_sop_generator.generate_correction_sop` accepts ErrorReport lists and expected state. It returns `repair_scope`, `source_step_id`, `rollback_to_step`, `target_parts`, and ordered `steps`. Every step contains `step_number`, action, instruction, visual instruction, affected/preserved parts, reference step, and whether an image is required.

Small errors use local repair. Missing parts use locate → insert → verify; extra parts use locate → remove → verify; a verified two-part wrongpart pair uses locate → remove → swap → verify. Explicit rollback metadata selects partial rollback; critical errors select full rollback. Ground-truth CSV is not read.

`utils.step_prompt_builder` requires the background, camera, lighting, vehicle, colors, counts, shapes, and non-target parts to remain unchanged, while showing a red action arrow. Every step still references the correct image and may chain the previous output. `utils.step_image_generator` isolates provider calls and records prompt, input paths, output, status, duration, and error. The default integrated provider is an offline assembly-card mock; no production provider or credential is assumed.

Per-step instruction images are the core project output. `generate_sop_flowchart` is only a compact overview and consumes the structured SOP, never raw Vision JSON.
