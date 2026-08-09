# Localization-Guided ROI Identity PoC

## Scope and safety

This is an offline proof of concept for `Test + Correct Reference -> delta ROI -> ROI-level candidate reduction -> ROI identity package`. It does not run a VLM, Azure/OpenAI API, GPT Image, or Phase 2B. It does not change the production Vision Prompt or Schema, Ground Truth, or source images.

Inference-time inputs are limited to the test/reference images, `ground_truth/model03/step03.json`, `config/part_library.json`, view/error metadata, local color/shape/position evidence, optional cached Grounding DINO detections, and the existing bbox selector. Confirmed Ground Truth is joined only after all identity packages have been frozen and is used only for offline coverage evaluation.

## Pipeline

`utils/roi_identity_pipeline.py` resizes each confirmed view, extracts HSV connected components, represents positions relative to the colored assembly extent, and pairs same-color Test/Reference components. Missing-part evidence is reference-only: it describes a region present in the correct reference but absent from the test image. Wrong-part evidence retains both reference-only and test-only regions so a swap/composite can be represented. Grounding DINO is optional offline corroboration on the priority front view; it does not override the deterministic delta evidence.

`utils/roi_candidate_builder.py` intersects reliable ROI color/family evidence with expected-state instances and the part library. It has no Ground Truth, review CSV, or case-ID input and contains no A01/PIN_RED_SHORT exception. Ordering is deterministic. If no evidence reaches the localization threshold, it emits no candidates, marks localization insufficient, and requires manual review.

The source/reference images have meaningful pose and scale differences. The PoC therefore keeps every output under manual review. A localization score is an evidence-ranking score, not calibrated identity confidence. Visual inspection found useful target ROIs alongside cross-view false positives, so these packages are suitable for a later ROI-level experiment only after the manual-review/ROI-selection boundary is retained.

## Results

| Case | Full | Reduced candidates | Count | Reduction | Confirmed GT coverage | Key ROI finding |
|---|---:|---|---:|---:|---|---|
| missingpart-A01 | 15 | `PIN_RED_SHORT`, `PIN_YELLOW`, `ROD_GREEN_LONG`, `LINK_RED_3HOLE`, `LINK_BLUE_5HOLE` | 5 | 66.67% | yes | Top-view reference-only evidence includes the red short-pin location absent from Test; other views also contain red false positives. |
| missingpart-B01 | 15 | `ROD_GREEN_LONG`, `PIN_YELLOW`, `WHEEL_BLUE_LARGE`, `WHEEL_BLUE_SMALL`, `PIN_RED_SHORT` | 5 | 66.67% | yes | Bottom-view wheel-shaped blue ROI retains `WHEEL_BLUE_SMALL`. |
| wrongpart-B01 | 15 | `LINK_BLUE_5HOLE`, `PIN_RED_SHORT`, `LINK_GREEN_5HOLE`, `ROD_GREEN_LONG`, `LINK_RED_3HOLE`, `PIN_YELLOW` | 6 | 60.00% | yes | Both `PIN_RED_SHORT` and `PIN_YELLOW` survive; reference-only and test-only ROIs support paired/swap packaging. |

Mean candidate reduction is 64.44%; confirmed-GT set coverage is 3/3 (100%); package-level localization failure rate is 0/3. `EYE_BALL` is excluded from all three reduced sets. Every package still has `requires_manual_review=true` because the current classical matching does not establish a uniquely correct bbox.

For missingpart-A01 specifically, the top-view ROI proposal covers the Reference red short pin that is absent from Test, `PIN_RED_SHORT` is retained, `EYE_BALL` is excluded, and the inventory falls from 15 to 5. The primary-view summary remains front-first for traceability, while the useful identity evidence is multi-view.

For missingpart-B01, `WHEEL_BLUE_SMALL` is retained, `EYE_BALL` is excluded, and the inventory falls from 15 to 5. For wrongpart-B01, both swap identities are retained in a six-part set and paired ROI output is supported; it is no longer constrained to only one of the two identities.

## Artifacts and decision

- `analysis/roi_identity_poc/roi_inventory.csv`: per-view localization inventory
- `analysis/roi_identity_poc/candidate_reduction.csv`: evaluation-only GT coverage join
- `analysis/roi_identity_poc/case_summary.csv`: case summary
- `analysis/roi_identity_poc/packages/`: frozen inference packages without GT
- `output/roi_identity_poc/`: crops and annotated ROI images; ignored by Git

Candidate reduction is effective and preserves confirmed GT in this three-case PoC. ROI Vision A/B is ready only as a small, manually reviewed offline/guarded experiment; it must not be promoted to production inference or Phase 2B. Phase 2B remains blocked.
