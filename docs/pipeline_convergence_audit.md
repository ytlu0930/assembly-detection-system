# Pipeline Convergence Audit

Updated: 2026-08-08

The repository now has one formal full-pipeline entry: `main.run_pipeline`. No third full pipeline was created. Earlier `utils.integration_pipeline` remains only as a deprecated compatibility module.

| Function | A latest implementation | Previous branch implementation | Canonical decision |
|---|---|---|---|
| Pipeline entry | `main.py / run_pipeline` | `utils/integration_pipeline.py` | A `main.run_pipeline`; utils entry deprecated |
| Parsed JSON loading | `pipeline_smoke_test.py` | ErrorReport adapter accepted wrappers | A loader plus branch adapter normalization |
| ErrorReport | raw `error_parts` | `utils/error_report_adapter.py` | Branch multi-ErrorReport adapter, emitted in `results.json` |
| Localization | primary error only | tested `LocalizationPipeline` stack | A control flow, tested detector/selector/annotator stack, one call per ErrorReport |
| SOP generation | root `correction_sop_generator.py` | `utils/correction_sop_generator.py` | Root generator, enriched with canonical aliases and swap/multi-part support |
| Prompt generation | `step_prompt_builder_v2.py` | `utils/step_prompt_builder.py` | V2 only; old builder compatibility-only |
| Step image generation | root V2 with direct OpenAI client | provider-neutral utils generator | V2 orchestration using provider contract |
| Provider abstraction | absent | Mock/OpenAI provider factory | Branch provider contract and factory |
| GPT Image 2 | direct SDK calls in V2 | guarded Images Edit adapter | `utils/openai_image_provider.py` only |
| Instruction book | `instruction_book_generator.py` | flowchart overview | A instruction book is formal visual output |
| Batch pipeline | duplicated stages | none | Thin coordinator calling `run_pipeline` |
| UI adapter | old adapter/flowchart | `utils/ui_pipeline_adapter.py` | `app.py` calls `main.run_pipeline`; old adapter compatibility-only |
| Output management | A case folders | reusable output manager | A case/manifest layout for formal pipeline; manager retained for experiments |

Canonical runtime: Parsed Vision JSON -> ErrorReport/Localization -> Correction SOP -> Step Prompt Builder V2 -> Step Image Generator V2 -> Mock/OpenAI provider -> Instruction Book -> Manifest -> Gradio UI.

`flowchart_generator.py` is deprecated and retained only for historical compatibility. It is not imported by `main.py`, `batch_pipeline.py`, or `app.py`.
