"""Deterministic affected-part candidates derived only from inference assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def build_affected_part_candidates(
    *,
    model_id: str,
    step_id: str,
    expected_state: Mapping[str, Any] | str | Path,
    part_library: Mapping[str, Any] | str | Path,
    error_type: str,
    observed_part_ids: Sequence[str] = (),
    swap_candidate_pairs: Sequence[Sequence[str]] = (),
) -> dict[str, Any]:
    expected_payload = _load(expected_state)
    library_payload = _load(part_library)
    if str(expected_payload.get("model_id")) != model_id or str(expected_payload.get("step_id")) != step_id:
        raise ValueError("expected_state model_id/step_id does not match the requested case")
    canonical = {str(item).upper() for item in library_payload}
    expected = {str(part.get("part_id") or "").upper() for part in expected_payload.get("expected_parts", [])}
    expected.discard("")
    observed = {str(item).upper() for item in observed_part_ids if str(item).upper() in canonical}
    swap = {str(item).upper() for pair in swap_candidate_pairs for item in pair if str(item).upper() in canonical}
    normalized_type = str(error_type).lower().replace("_", "")
    candidates = set(expected)
    if normalized_type in {"extrapart", "extra"}:
        candidates.add("UNKNOWN_EXTRA_PART")
    elif normalized_type in {"wrongpart", "wrong"}:
        candidates.update(observed)
        candidates.update(swap)
    ordered = sorted(item for item in candidates if item in canonical)
    if "UNKNOWN_EXTRA_PART" in candidates:
        ordered.append("UNKNOWN_EXTRA_PART")
    return {
        "candidate_part_ids": ordered,
        "candidate_metadata": {
            "model_id": model_id,
            "step_id": step_id,
            "error_type": error_type,
            "source": "expected_state+part_library",
            "expected_candidate_count": len(expected),
            "observed_candidate_count": len(observed),
            "swap_candidate_count": len(swap),
            "human_review_source_used": False,
        },
    }
