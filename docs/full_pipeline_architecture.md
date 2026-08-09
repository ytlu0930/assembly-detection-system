# Full Pipeline Architecture

The unified entry is `utils.integration_pipeline.run_full_pipeline(...)`. It validates paths; invokes the authoritative analyzer or replays an injected result; converts every detected error to `ErrorReport[]`; localizes each report independently; attaches bbox after Vision; annotates available boxes; generates a deterministic structured SOP; builds one visual task per SOP step; invokes a provider-neutral image generator; produces an optional SOP overview; and returns one UI-ready result.

`localization_pipeline` is only the Grounding DINO → selector → annotation subflow. It is not the full system. Localization failure preserves Vision, ErrorReports, and text SOP. Step-image failure preserves text SOP, annotated image, and flowchart. Flowchart failure preserves the core SOP outputs. Correct cases return no ErrorReports and no repair steps.

`ErrorReport` fields are `part_id`, `error_type`, `expected_value`, `actual_value`, `description`, `severity`, `confidence`, `evidence`, `bbox`, `unresolved`, `role`, `overall_error_type`, and `error_components`. Vision never owns bbox; localization adds it later.

All external-heavy components are injectable. Unit tests use fixtures, a no-detection localizer, and a mock image provider, so they do not load Grounding DINO, use a GPU, or call an API.

The selected formal step-image provider is OpenAI Image API / `gpt-image-2` in image-edit mode, but it is not the runtime default. `create_step_image_provider()` defaults to mock; selecting `openai` only constructs a guarded adapter and never performs a request. `run_full_pipeline` retains the `image_provider` injection point and also accepts `step_image_provider` plus `execute_image_api`. All three authorization gates must agree before lazy client construction. Disabled and failed results become warnings and preserve text SOP, annotation, flowchart, and UI output.

## Canonical architecture after convergence (2026-08-08)

The sole formal entry is `main.run_pipeline`. Parsed Vision JSON flows through multi-ErrorReport Localization, the root correction SOP generator, Step Prompt Builder V2, provider-backed Step Image Generator V2, Instruction Book Generator, and PipelineManifest. `batch_pipeline.py` and `app.py` delegate to this entry. `utils.integration_pipeline` is deprecated compatibility code, and `flowchart_generator.py` is not a formal runtime dependency. The instruction book replaces the flowchart as the final visual SOP.

The provider layer now contains Mock, alternate OpenAI Platform, and formal Azure-hosted OpenAI adapters. `azure_openai` uses the teacher deployment's single-image multipart edit contract. Sequential source chaining remains in Step Image Generator V2, while reference paths stay in metadata and prompt context until Azure multi-image support is verified.

## Affected-part identity gate (2026-08-08)

The canonical flow now inserts `AffectedPartIdentityVerifier` after paired test/reference localization and before Correction SOP generation. Extended ErrorReports carry `identity_status`, `identity_confidence`, `identity_evidence`, `verified_part_id`, and `alternative_candidates`. A known taxonomy ID, high Vision confidence, or expected-inventory membership is not sufficient verification.

Only a `verified` identity can create named repair/image tasks. `conflict`, `uncertain`, and `unresolved` identities force manual review, remove the predicted identity from formal SOP targets, set `identity_verification_blocked=true`, and block Step Prompt/Image generation even when a manual-review override is supplied. The verifier never reads case-specific human answers as inference input.
