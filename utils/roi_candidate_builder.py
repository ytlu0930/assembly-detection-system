"""Deterministic ROI candidate reduction using inference-time assets only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _load(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def part_family(part_id: str, aliases: list[str] | tuple[str, ...]) -> str:
    text = " ".join([part_id.replace("_", " "), *[str(item) for item in aliases]]).lower()
    if "wheel" in text or "tire" in text:
        return "wheel"
    if "eye" in text or "pupil" in text:
        return "eye"
    if "pin" in text or "nail" in text or "cylinder stick" in text:
        return "pin"
    if "flat bar" in text or " link " in f" {text} ":
        return "bar"
    if "rod" in text or "long green cylinder" in text:
        return "rod"
    if "joint" in text:
        return "joint"
    if "plate" in text:
        return "plate"
    if "connector" in text or "clip" in text:
        return "connector"
    if "block" in text or "cube" in text:
        return "block"
    return "unknown"


def build_roi_candidates(
    *,
    expected_state: Mapping[str, Any] | str | Path,
    part_library: Mapping[str, Any] | str | Path,
    roi_evidence: list[dict[str, Any]],
    error_type: str,
    view_angle: str,
    minimum_localization_score: float = 0.45,
) -> dict[str, Any]:
    """Reduce inventory by ROI color/family/position without review labels or case IDs."""
    expected = _load(expected_state)
    library = _load(part_library)
    canonical = sorted(str(item).upper() for item in library)
    expected_rows = [item for item in expected.get("expected_parts", []) if isinstance(item, dict)]
    expected_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in expected_rows:
        part_id = str(row.get("part_id") or "").upper()
        if part_id in library:
            expected_by_id.setdefault(part_id, []).append(row)

    reliable = [
        item for item in roi_evidence
        if float(item.get("score") or 0.0) >= minimum_localization_score
        and str(item.get("status") or "success") == "success"
    ]
    if not reliable:
        return {
            "candidate_part_ids": [], "candidate_count": 0,
            "full_candidate_count": len(canonical), "reduction_ratio": 1.0,
            "status": "localization_insufficient", "candidate_evidence": [],
            "human_review_source_used": False,
        }

    colors = {str(item.get("color") or "").upper() for item in reliable if item.get("color")}
    families = {str(item.get("shape_family") or "unknown") for item in reliable}
    positions = {str(item.get("position") or "").upper() for item in reliable if item.get("position")}
    by_color = {
        part_id for part_id, rows in expected_by_id.items()
        if any(str(row.get("color") or "").upper() in colors for row in rows)
    }
    family_matches = {
        part_id for part_id in by_color
        if part_family(part_id, list(library.get(part_id, []))) in families
    }
    selected = family_matches or by_color

    normalized_view = str(view_angle).lower()
    if normalized_view in {"front", "back"} and positions:
        positioned = {
            part_id for part_id in selected
            if any(str(row.get("position") or "").upper() in positions for row in expected_by_id[part_id])
        }
        if positioned:
            selected = positioned

    evidence_rows = []
    for part_id in sorted(selected):
        family = part_family(part_id, list(library.get(part_id, [])))
        matched = [
            item for item in reliable
            if str(item.get("color") or "").upper() in {
                str(row.get("color") or "").upper() for row in expected_by_id[part_id]
            }
            and (family == str(item.get("shape_family")) or not family_matches)
        ]
        score = max((float(item.get("score") or 0.0) for item in matched), default=minimum_localization_score)
        evidence_rows.append({
            "part_id": part_id, "score": score, "family": family,
            "colors": sorted({str(item.get("color") or "").upper() for item in matched}),
            "difference_supported": bool(matched),
            "roi_evidence": sorted(matched, key=lambda item: (-float(item.get("score") or 0.0), item.get("bbox", [])))[:2],
            "source": "expected_state+part_library+roi_spatial_local_evidence",
        })
    ordered = [item["part_id"] for item in sorted(evidence_rows, key=lambda item: (-item["score"], item["part_id"]))]
    return {
        "candidate_part_ids": ordered,
        "candidate_count": len(ordered),
        "full_candidate_count": len(canonical),
        "reduction_ratio": 1.0 - len(ordered) / len(canonical) if canonical else None,
        "status": "success" if ordered else "no_candidate",
        "candidate_evidence": evidence_rows,
        "error_type": str(error_type),
        "human_review_source_used": False,
    }
