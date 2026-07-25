# Phase 8.1 Formal Ground Truth

The canonical evaluation metadata is `data/ground_truth.csv`. It is generated
deterministically from the frozen inventory at
`output/dataset_audit/20260722_170328/dataset_inventory.csv`; source images are
read-only inputs.

Generate it with the project environment:

```powershell
.\venv\Scripts\python.exe scripts\build_ground_truth.py
```

## Identity and fields

`image_id` is the project-relative `image_path`, for example
`input/.../image.jpg` or `regression_subset/image.jpg`. A filename alone cannot
be the identifier because 10 filenames occur in both source splits. The
`image_name` column preserves the basename for display and legacy integration.

Required evaluation fields are:

- `image_id`, `image_path`, `model_id`, `step_id`, `view_angle`
- `is_error`, `error_type`, `error_detail`, `evaluation_scope`
- `source_split`

Traceability fields include `raw_label`, `filename_valid`,
`inventory_filename_valid`, `inventory_validation_errors`,
`duplicate_group_id`, and `sha256`. `filename_valid` uses the current shared
taxonomy; the `inventory_*` fields preserve the frozen audit's historical
result. `schema_error_type` is the adapter value used by the existing Vision
output schema and batch comparison contract.

Load and validate the file with:

```python
from utils.ground_truth_loader import load_ground_truth

rows = load_ground_truth()
```

For lookup, use the source-qualified `image_id`. Filename fallback is supported
only when the filename is unique; an ambiguous filename raises `KeyError`.

## Central taxonomy

The single source of truth is `utils/taxonomy.py`.

| Raw filename label | Formal `error_type` | Vision schema value | Scope |
|---|---|---|---|
| `correct` | `correct` | `correct` | in |
| `positionerror` | `position` | `positionerror` | in |
| `missingpart` | `missing` | `missingpart` | in |
| `extrapart` | `extra` | `extrapart` | in |
| `wrongpart` | `wrongpart` | `wrongpart` | in |
| `criticalerror` | `criticalerror` | `criticalerror` | in |
| `orientationerror` | `orientation` | unavailable | out |

`criticalerror` is retained without semantic remapping because the current
Vision schema explicitly supports it. `orientation` remains a formal taxonomy
member, but the schema has no corresponding enum and the frozen dataset has no
orientation sample; it is therefore out of evaluation scope.

## Frozen dataset result

The formal file contains 158 rows: 61 correct and 97 error images.

| Formal type | Count |
|---|---:|
| correct | 61 |
| missing | 36 |
| wrongpart | 28 |
| extra | 15 |
| position | 12 |
| criticalerror | 6 |
| orientation | 0 |

All 158 collected rows are in scope. The aggregate targets of at least 30
correct and 80 error images are met. Per-class targets are not all met:
position is 12/20, orientation is 0/20, and extra is 15/20. The requirement of
coverage across at least three steps per error class is also not met. These
shortfalls are recorded as dataset limitations, not filled with synthetic rows
or label remapping.

The legacy repository-root `ground_truth.csv` is not the Phase 8.1 canonical
file and is intentionally left unchanged.
