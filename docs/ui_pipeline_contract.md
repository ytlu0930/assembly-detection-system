# UI Pipeline Contract

The UI calls only `utils.ui_pipeline_adapter.run_analysis_for_ui(image_path, model_id, step_id, view_angle, ...)`. It must not read Vision bbox or connect individual modules itself.

The fixed result contains `success`, `analysis_json`, `annotated_image`, `sop_steps`, `sop_gallery`, `flowchart`, `correction_text`, `confidence`, `warnings`, `error_message`, and `raw_result`. Gallery items are `(generated_image_path, "步驟 N：instruction")`.

`app.py` now uses this adapter. Uploaded files must retain a parseable project filename until the UI gains explicit model/step/view controls; otherwise the adapter should be called with those values from UI controls. API failures appear in `error_message`; optional-stage failures appear in `warnings` while usable outputs remain available.

The UI must not select a provider from environment secrets. Runtime remains `MockStepImageProvider`. If the guarded OpenAI Image API / `gpt-image-2` adapter is injected without all execution gates, the first step is marked disabled, later image steps are skipped, and the message is returned in `warnings`; UI result keys and text SOP remain unchanged.

## Canonical UI contract (2026-08-08)

`app.py` imports `run_pipeline` from `main`. Raw uploads first use the existing analyzer to create a parsed JSON artifact, which is then passed to the formal pipeline. The primary UI output is `manifest.final_instruction_path`. The UI may show annotation, warnings, and status, but does not call Localization, image annotation, SOP generation, OpenAI, or the deprecated flowchart directly.
