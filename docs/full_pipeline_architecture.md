# Full Pipeline Architecture

The unified entry is `utils.integration_pipeline.run_full_pipeline(...)`. It validates paths; invokes the authoritative analyzer or replays an injected result; converts every detected error to `ErrorReport[]`; localizes each report independently; attaches bbox after Vision; annotates available boxes; generates a deterministic structured SOP; builds one visual task per SOP step; invokes a provider-neutral image generator; produces an optional SOP overview; and returns one UI-ready result.

`localization_pipeline` is only the Grounding DINO → selector → annotation subflow. It is not the full system. Localization failure preserves Vision, ErrorReports, and text SOP. Step-image failure preserves text SOP, annotated image, and flowchart. Flowchart failure preserves the core SOP outputs. Correct cases return no ErrorReports and no repair steps.

`ErrorReport` fields are `part_id`, `error_type`, `expected_value`, `actual_value`, `description`, `severity`, `confidence`, `evidence`, `bbox`, `unresolved`, `role`, `overall_error_type`, and `error_components`. Vision never owns bbox; localization adds it later.

All external-heavy components are injectable. Unit tests use fixtures, a no-detection localizer, and a mock image provider, so they do not load Grounding DINO, use a GPU, or call an API.
