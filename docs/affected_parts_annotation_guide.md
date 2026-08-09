# Affected-Parts Annotation Guide

## Purpose

Error-type accuracy and affected-part accuracy measure different capabilities. The 2026-07-01 run classified error types well but frequently named the wrong part, so a reviewed part-level reference is required before Prompt/Schema A/B evaluation.

## Evidence to inspect

For every row, reviewers must inspect the Test Image, same-view Correct Reference Image, matching expected-state JSON, and `config/part_library.json`. Assembly instructions or existing human notes may be supporting evidence. Vision predictions must never be copied into the Ground Truth columns or treated as evidence.

## CSV conventions

- Multiple values use semicolons, never Python/JSON list syntax.
- `annotation_status` is exactly `confirmed`, `uncertain`, or `needs_second_review`.
- `relation_type` is exactly `single`, `swap`, `replacement`, `composite`, or `unknown`.
- Canonical IDs must already exist in expected state or the part library. Do not invent an ID.
- `unresolved_extra_part` is a workflow placeholder, not a newly approved canonical ID.
- Leave a value empty when the evidence cannot support it and select `needs_second_review`.

## Relation rules

A swap means two known parts occupy each other's expected locations; record both canonical IDs and use `relation_type=swap`. A composite case contains two or more coexisting error components, which may involve one identity with count/multiplicity differences or several parts; list every supported component and use `relation_type=composite`. A replacement is one observed identity substituted for a different expected identity without a confirmed reciprocal swap.

## Review workflow

1. Reviewer 1 checks all four evidence sources and fills `reviewer` plus `review_notes`.
2. Rows marked `needs_second_review` are independently checked by Reviewer 2 without seeing Vision predictions.
3. Reviewer 2 confirms a canonical ID only when the same identity is supported across applicable views and by expected state or the library.
4. Disagreement remains `uncertain`; it is not resolved by majority model output.
5. After approval, freeze a reviewed copy with reviewer names and date. Do not overwrite `data/ground_truth.csv` automatically.

## Current special cases

- missingpart-A01 maps to formal ID `PIN_RED_SHORT`.
- missingpart-B01 maps to formal ID `WHEEL_BLUE_SMALL`.
- wrongpart-B01 is a `PIN_YELLOW`/`PIN_RED_SHORT` swap pair.
- wrongpart-A01 remains a possible eye-related composite and requires count-aware second review.
- All extrapart-A01 views share placeholder `unresolved_extra_part` until one canonical ID is approved.

## A/B evaluation use

The A/B evaluator must read only rows whose `annotation_status=confirmed` for primary metrics. It parses semicolon-delimited affected IDs and components as sets, while retaining part multiplicity in reviewer notes until a count-aware schema is approved. Report error-type accuracy, affected-part exact match, at-least-one recall, all-parts recall, composite recall, and unknown-part rate with explicit denominators.
