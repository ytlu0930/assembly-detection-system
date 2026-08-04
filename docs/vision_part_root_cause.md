# Vision Part Identification Root Cause

| Cause | Evidence For | Evidence Against | Confidence | Recommended Action |
|---|---|---|---|---|
| Prompt suppresses composite reporting | Rule 5 says to choose the single most important issue; it never asks for ALL affected parts or per-part evidence | `detected_parts` is described as an array | High | Run a small, budget-approved v2.1 A/B that requests exhaustive comparison; do not switch production before results |
| Schema lacks composite relations | No `error_components`, expected/observed pair, role, evidence, swap relation, or count; `overall_error_type` is singular | Array items permit more than one part | High | Design a versioned schema plus adapter only after A/B; do not destructively replace v1 |
| Schema `minItems: 1` encourages filler | Correct cases must output at least one part and Prompt repeats this constraint | It does not by itself explain wrong IDs on error cases | Medium | In a future version allow zero error parts for correct cases |
| Expected state/library ambiguity | Counts are encoded only by repeated rows; positions are coarse; `PIN_YELLOW` is visually trumpet-like; no canonical extra long red rod | Target IDs for the four named cases exist and colors/shapes are present | High | Add reviewed count/location metadata in a new expected-state version; add aliases only after part owner review |
| Parser/analyzer truncation | None in `current_state_analyzer`; it validates and returns the full array | N/A | Low | Preserve analyzer behavior |
| Downstream truncation | `pipeline_smoke_test.py` explicitly uses `error_parts[0]`; older UI assembled reports itself | New adapter and pipeline now iterate every report | High | Use only `utils.ui_pipeline_adapter` from UI |
| Model visual salience/occlusion | Missing cases frequently guess `EYE_BALL`; errors persist across views; back/front are weaker | Prompt/schema/data confounders have not been experimentally removed | Medium, not yet a final model-limit finding | After Prompt A/B, consider second-stage part verification and Grounding DINO-assisted localization |

The root cause is multi-factor. The strongest current causes are an explicitly single-issue Prompt, insufficient composite semantics, expected-state identity/count ambiguity, and a first-item-only prototype. A model limitation cannot yet be declared because no controlled minimal Prompt A/B has been run. High-resolution detail was enabled and references were supplied, so those factors are not absent.

Production Prompt and Schema were therefore not changed in this branch. The new adapter is forward-compatible with `error_components`, `expected_part`, `observed_part`, `role`, and `evidence`, while preserving all legacy detected parts.
