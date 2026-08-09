# Candidate Constraint Failure Analysis

## missingpart-A01 evidence

This analysis uses only the persisted package and response artifacts. No API was called.

| Item | Observed value |
|---|---|
| Candidate package | `analysis/vision_prompt_ab/packages/reference_candidate/missingpart-A01/` |
| Candidate IDs | `BLOCK_GREEN_4HOLE_2PEG`, `BLOCK_YELLOW_CUBE`, `CONNECTOR_ORANGE`, `EYE_BALL`, `JOINT_BLUE_Y`, `JOINT_YELLOW_H`, `LINK_BLUE_5HOLE`, `LINK_GREEN_5HOLE`, `LINK_RED_3HOLE`, `PIN_RED_SHORT`, `PIN_YELLOW`, `PLATE_BLUE_TRIANGLE`, `ROD_GREEN_LONG`, `WHEEL_BLUE_LARGE`, `WHEEL_BLUE_SMALL` |
| Candidate count | 15 |
| Canonical library count | 15 |
| Actual prompt rule | Select only an ID from the supplied list or `UNKNOWN`/`UNRESOLVED`; never invent an ID |
| Raw predicted part | `EYE_BALL` |
| Confidence | 0.95 |
| Human ground truth used only for evaluation | `PIN_RED_SHORT` |

Direct answers:

1. **Was `EYE_BALL` in the candidate list?** Yes.
2. **Was `PIN_RED_SHORT` in the candidate list?** Yes.
3. **How many candidates were supplied?** 15.
4. **Was the list too broad to constrain identity effectively?** Yes. It covered 15/15 canonical inventory IDs because the missing-part builder used the complete step expected inventory.
5. **Did the prompt forbid IDs outside the list?** Yes, except explicit unknown/unresolved sentinels.
6. **Does the current production schema allow arbitrary `part_id` strings?** Yes. It requires a non-empty string and has no candidate membership enum.
7. **Did the original runner perform post-response membership validation?** No. It schema-validated structure only.

Therefore A01's `EYE_BALL` was not a candidate-membership violation. It was a semantically false but in-set prediction. Runtime enforcement prevents out-of-set identities from escaping, but cannot make a full-inventory candidate list discriminate between `EYE_BALL` and `PIN_RED_SHORT`.

## Candidate breadth audit

The deterministic audit is `analysis/vision_prompt_ab/results/candidate_set_audit.csv`.

| case_id | candidate_count | inventory coverage | constraint_effectiveness |
|---|---:|---:|---|
| missingpart-A01 | 15 | 15/15 | weak |
| missingpart-B01 | 15 | 15/15 | weak |
| wrongpart-B01 | 15 | 15/15 | weak |

The same weak designation applies to `wrongpart-A01`; `extrapart-A01` has the full inventory plus `UNKNOWN_EXTRA_PART`. No candidate list was manually reduced to match Ground Truth.

## Deterministic runtime enforcement

The experimental A/B runner now assigns every response one of:

- `valid`: Variant C identity is in the supplied candidate set or is an explicitly allowed unknown category.
- `violation`: Variant C identity is outside that set.
- `not_applicable`: non-Candidate variants.

For `violation`, finalization sets `verified_part_id` to null/blank, sets `requires_manual_review=true`, and leaves the verifier unresolved. It never maps the invalid ID to a nearest candidate. Historical persisted results are deterministically backfilled by the same membership function.

Metrics now include Candidate Violation Rate and High-confidence Candidate Violation Rate. In the already completed six Candidate artifacts both rates are 0%; that result only means all outputs were members of their broad lists. It does not establish semantic identity correctness.

Implementation locations:

- `scripts/run_affected_part_prompt_ab.py`: `enforce_candidate_constraint` and response-time recording.
- `scripts/finalize_affected_part_prompt_ab.py`: historical backfill and violation quarantine.
- `scripts/evaluate_affected_part_identity.py`: aggregate violation metrics.
- `scripts/audit_affected_part_ab_failures.py`: deterministic candidate breadth audit.

## Root-cause statement

`CANDIDATE_CONSTRAINT_ROOT_CAUSE = missingpart-A01 supplied the complete 15-part canonical inventory, including both EYE_BALL and PIN_RED_SHORT; the schema allowed any non-empty part_id and the original runner had no post-response membership check, so the prompt constraint was structurally unenforced and semantically weak.`
