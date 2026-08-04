# UI Pipeline Contract

The UI calls only `utils.ui_pipeline_adapter.run_analysis_for_ui(image_path, model_id, step_id, view_angle, ...)`. It must not read Vision bbox or connect individual modules itself.

The fixed result contains `success`, `analysis_json`, `annotated_image`, `sop_steps`, `sop_gallery`, `flowchart`, `correction_text`, `confidence`, `warnings`, `error_message`, and `raw_result`. Gallery items are `(generated_image_path, "步驟 N：instruction")`.

`app.py` now uses this adapter. Uploaded files must retain a parseable project filename until the UI gains explicit model/step/view controls; otherwise the adapter should be called with those values from UI controls. API failures appear in `error_message`; optional-stage failures appear in `warnings` while usable outputs remain available.
