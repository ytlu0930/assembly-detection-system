# Vision Error Report and Correction SOP Integration

## Completed

- Systematic analysis of 2026/07/01 Vision parsed results
- Separation of error-type accuracy and affected-part accuracy
- Multi-part ErrorReport adapter
- Structured correction SOP pipeline
- SOP-driven flowchart generation
- Full integration pipeline
- UI adapter
- Mock step-image generation
- Offline replay and fallback tests

## Key Findings

- Error type accuracy: 94.83%
- At least one affected part match: 25.00%
- All affected parts detected: 18.75%
- Composite full recall: 0%
- Unknown part rate: 20.93%

## Not Completed

- Prompt／Schema A/B validation
- Real OpenAI Image API smoke execution and formal image-quality evaluation
- Real Vision API regression
- Real image generation smoke test
- Complete composite-error recognition

## Important Notes

- Step images currently use MockStepImageProvider
- Formal provider selected: OpenAI Image API / GPT Image 2 (model id: `gpt-image-2`) using Image Editing
- OpenAIImageProvider is a guarded GPT Image 2 Images Edit adapter; lazy client creation, retries, budgets, validation, and structured results are implemented
- The adapter is not configured, API-smoke-tested, or image-quality-validated; this batch executed no image API request
- Prompt and Schema were not modified
- Formal Ground Truth and source images were not modified
- Branch must be reviewed before merge
- UI should use ui_pipeline_adapter.py
- bbox remains outside Vision Schema

## Review Checklist

- ErrorReport format
- SOP structure
- Flowchart input contract
- Full pipeline fallback behavior
- UI adapter return format
- Mock provider boundary
- Backward compatibility

## Pipeline convergence addendum

- Canonical single entry: `main.run_pipeline`
- Canonical batch: `batch_pipeline.run_batch`, delegating every case to `run_pipeline`
- Canonical prompt/image path: Step Prompt Builder V2 -> provider-backed Step Image Generator V2
- Canonical final visual: `assembly_instruction_book.png`
- Deprecated runtime artifacts: `flowchart_generator.py`, `utils.integration_pipeline.py`, and the previous UI adapter
- API safety: mock default, separate provider/execute/cost flags, environment gates, request budget, and malformed-key preflight
