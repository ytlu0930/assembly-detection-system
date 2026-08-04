"""Normalize legacy and future Vision JSON into independent error reports.

Bounding boxes deliberately do not belong to the Vision contract.  They remain
``None`` until the localization stage enriches each report.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


VALID_ERROR_TYPES = {
    "missingpart",
    "extrapart",
    "positionerror",
    "wrongpart",
    "criticalerror",
    "uncertain",
}


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _severity(error_type: str) -> str:
    if error_type == "criticalerror":
        return "critical"
    if error_type in {"wrongpart", "positionerror"}:
        return "high"
    if error_type in {"missingpart", "extrapart"}:
        return "medium"
    return "unknown"


def adapt_vision_result(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one serializable ErrorReport for every erroneous detected part.

    ``payload`` may be a raw model response, a ``current_parsed_json`` wrapper,
    or an analyzer return value.  Legacy fields are retained where useful and
    future ``expected_part``, ``observed_part``, ``evidence``, ``role`` and
    ``error_components`` fields are accepted without requiring a schema switch.
    Correct assemblies return an empty list; unknown identities are retained and
    marked unresolved instead of being discarded.
    """
    if not isinstance(payload, dict):
        raise TypeError("Vision payload must be a dictionary")

    model = payload.get("model_response", payload)
    if not isinstance(model, dict):
        raise ValueError("model_response must be a dictionary")

    overall = str(model.get("overall_error_type", "uncertain")).lower()
    is_error = bool(model.get("is_error", overall != "correct"))
    if not is_error or overall == "correct":
        return []

    detected = model.get("detected_parts", [])
    if detected is None:
        detected = []
    if not isinstance(detected, list):
        raise ValueError("detected_parts must be a list")

    components = model.get("error_components", [])
    if not isinstance(components, list):
        components = []
    normalized_components = [str(item).lower() for item in components]

    reports: list[dict[str, Any]] = []
    for index, item in enumerate(detected):
        if not isinstance(item, dict):
            raise ValueError(f"detected_parts[{index}] must be a dictionary")
        error_type = str(item.get("error_type", overall)).lower()
        if error_type == "correct":
            continue
        if error_type not in VALID_ERROR_TYPES:
            error_type = "uncertain"

        part_id = str(item.get("part_id") or "unknown_part").strip()
        unresolved = part_id.lower().startswith("unknown") or not part_id
        expected = item.get("expected_part", item.get("expected_state"))
        observed = item.get("observed_part", item.get("observed_state"))
        evidence = str(item.get("evidence") or item.get("description") or "").strip()

        reports.append(
            {
                "part_id": part_id or "unknown_part",
                "error_type": error_type,
                "expected_value": deepcopy(expected),
                "actual_value": deepcopy(observed),
                "description": str(item.get("description") or "").strip(),
                "severity": _severity(error_type),
                "confidence": _confidence(item.get("confidence")),
                "evidence": evidence,
                "bbox": None,
                "unresolved": unresolved,
                "role": item.get("role"),
                "overall_error_type": overall,
                "error_components": list(normalized_components),
            }
        )

    if not reports:
        reports.append(
            {
                "part_id": "unknown_part",
                "error_type": overall if overall in VALID_ERROR_TYPES else "uncertain",
                "expected_value": None,
                "actual_value": None,
                "description": str(model.get("summary") or "Error reported without a part identity"),
                "severity": _severity(overall),
                "confidence": 0.0,
                "evidence": str(model.get("summary") or ""),
                "bbox": None,
                "unresolved": True,
                "role": None,
                "overall_error_type": overall,
                "error_components": list(normalized_components),
            }
        )
    return reports


def attach_bbox(
    reports: list[dict[str, Any]], index: int, bbox: list[float] | None
) -> list[dict[str, Any]]:
    """Return a copy with one localization bbox attached."""
    if not 0 <= index < len(reports):
        raise IndexError("ErrorReport index out of range")
    result = deepcopy(reports)
    if bbox is not None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox must contain [x1, y1, x2, y2]")
        result[index]["bbox"] = [float(value) for value in bbox]
    return result
