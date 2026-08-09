# Reference Variant Schema Failure Analysis

## Scope and conclusion

This is an offline forensic analysis of the five failed Reference artifacts already present under `analysis/vision_prompt_ab/results/raw/`. No API request was made.

All five model calls returned content that had already been parsed into a Python/JSON object before `jsonschema` validation failed. The old runner then persisted only the exception text and set both `raw_response` and `parsed_output` to null. Fortunately, the validator's `On instance:` payload retained the parsed object and made analysis-only recovery possible.

The common root cause is **schema metadata emitted as instance properties**. The current schema has top-level `additionalProperties: false`, so `$schema`, `title`, and `type` are invalid response fields. `missingpart-A01` additionally returned a JSON-Schema-shaped wrapper whose response values were nested under `properties`. This behavior is consistent with the request presenting the complete JSON Schema in the system message and asking for an object conforming to it; it is not evidence of a transport or JSON parser failure.

## Per-case forensic table

| case_id | raw_response_available | json_parse | schema_validation | failed_field | expected | actual | root_cause |
|---|---|---|---|---|---|---|---|
| missingpart-A01 | Model response received; old artifact discarded body | success before validation; recoverable from exception | failed | top-level `$schema`, `title`, `type`, `additionalProperties`, `required`, `properties` | Only the seven response fields allowed by the current schema | JSON-Schema-shaped wrapper; response values nested under `properties` | Schema definition echoed/wrapped instead of returning only an instance |
| missingpart-B01 | Model response received; old artifact discarded body | success before validation; recoverable from exception | failed | top-level `$schema`, `title`, `type` | No fields outside the current schema | Valid response instance plus three schema metadata fields | Schema metadata copied into instance |
| extrapart-A01 | Model response received; old artifact discarded body | success before validation; recoverable from exception | failed | top-level `$schema`, `title`, `type` | No fields outside the current schema | Valid response instance plus three schema metadata fields | Schema metadata copied into instance |
| wrongpart-A01 | Model response received; old artifact discarded body | success before validation; recoverable from exception | failed | top-level `$schema`, `title`, `type` | No fields outside the current schema | Valid response instance plus three schema metadata fields | Schema metadata copied into instance |
| wrongpart-B01 | Model response received; old artifact discarded body | success before validation; recoverable from exception | failed | top-level `$schema`, `title`, `type` | No fields outside the current schema | Valid response instance plus three schema metadata fields | Schema metadata copied into instance |

There were no missing required response fields after recovery, enum mismatches, unexpected nulls, malformed `detected_parts`, malformed evidence objects, invalid confidence types/ranges, or nested `additionalProperties` failures. The only validation problem was the top-level schema material. The current schema does not contain separate `affected_parts` or `evidence` fields; the affected identity is represented by `detected_parts`, with evidence expressed in each item's `description`.

## Analysis-only recovery

The recovery script is `scripts/audit_affected_part_ab_failures.py`. It extracts the already parsed `On instance:` value without contacting a service, unwraps `missingpart-A01`'s `properties`, strips only schema metadata from the other four, and revalidates each result against the unchanged current schema.

| case_id | recovered part IDs | recovered validation |
|---|---|---|
| missingpart-A01 | `JOINT_YELLOW_H` | valid |
| missingpart-B01 | `PIN_YELLOW` | valid |
| extrapart-A01 | `PIN_RED_SHORT` | valid |
| wrongpart-A01 | `BLOCK_GREEN_4HOLE_2PEG`, `EYE_BALL` | valid |
| wrongpart-B01 | `PIN_RED_SHORT`, `PIN_YELLOW` | valid |

The five recovered records are under `analysis/vision_prompt_ab/recovered/reference/`. Every record is explicitly marked:

```json
{
  "recovered_for_analysis": true,
  "excluded_from_primary_metrics": true
}
```

These records must not be silently substituted into the primary A/B metrics. Future experimental persistence should retain the raw model response and parsed object even when validation fails, so a validation error never destroys the evidence needed for diagnosis.

## Root-cause statement

`REFERENCE_SCHEMA_ROOT_CAUSE = Reference responses echoed JSON Schema metadata into the response instance; top-level additionalProperties=false rejected those fields, while the old failure artifact discarded the raw/parsed body.`
