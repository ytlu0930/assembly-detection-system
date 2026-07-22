from __future__ import annotations

from math import hypot
from typing import Any


TARGET_POSITIONS = {
    "center": (0.50, 0.50),
    "top": (0.50, 0.15),
    "bottom": (0.50, 0.85),
    "left": (0.15, 0.50),
    "right": (0.85, 0.50),
    "top_left": (0.15, 0.15),
    "top_right": (0.85, 0.15),
    "bottom_left": (0.15, 0.85),
    "bottom_right": (0.85, 0.85),
    "any": None,
}

DEFAULT_EXPECTED_AREA_RANGE = (0.01, 0.08)
DEFAULT_WEIGHTS = {
    "confidence": 0.20,
    "position": 0.25,
    "area": 0.35,
    "oversized": 0.15,
    "boundary": 0.05,
}


class BBoxCandidateSelector:
    """Select an explainable best bbox without depending on a detector model."""

    def __init__(
        self,
        expected_area_range: tuple[float, float] = DEFAULT_EXPECTED_AREA_RANGE,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.expected_area_range = self._validate_area_range(expected_area_range)
        self.weights = self._merge_weights(weights)

    @staticmethod
    def _validate_area_range(values: tuple[float, float]) -> tuple[float, float]:
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("expected_area_range must contain (minimum, maximum)")
        minimum, maximum = (float(value) for value in values)
        if not 0.0 < minimum <= maximum <= 1.0:
            raise ValueError("expected_area_range must satisfy 0 < minimum <= maximum <= 1")
        return minimum, maximum

    @staticmethod
    def _merge_weights(overrides: dict[str, float] | None) -> dict[str, float]:
        result = dict(DEFAULT_WEIGHTS)
        if overrides is None:
            return result
        if not isinstance(overrides, dict):
            raise ValueError("weights must be a dictionary")
        unknown = set(overrides) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown weight names: {sorted(unknown)}")
        for name, value in overrides.items():
            number = float(value)
            if number < 0.0:
                raise ValueError(f"weight '{name}' must be non-negative")
            result[name] = number
        if not any(result.values()):
            raise ValueError("at least one weight must be positive")
        return result

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(value, upper))

    @classmethod
    def _prepare_bbox(
        cls,
        bbox: Any,
        image_width: int,
        image_height: int,
    ) -> tuple[list[float], bool]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox must contain [x1, y1, x2, y2]")
        try:
            raw = [float(value) for value in bbox]
        except (TypeError, ValueError) as exc:
            raise ValueError("bbox coordinates must be numeric") from exc
        if not all(value == value and abs(value) != float("inf") for value in raw):
            raise ValueError("bbox coordinates must be finite")

        x1, y1, x2, y2 = raw
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")

        clipped = [
            cls._clamp(x1, 0.0, float(image_width)),
            cls._clamp(y1, 0.0, float(image_height)),
            cls._clamp(x2, 0.0, float(image_width)),
            cls._clamp(y2, 0.0, float(image_height)),
        ]
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            raise ValueError("bbox has no visible area after clipping")
        return clipped, clipped != raw

    @staticmethod
    def _area_score(area_ratio: float, minimum: float, maximum: float) -> float:
        if minimum <= area_ratio <= maximum:
            return 1.0
        if area_ratio < minimum:
            return area_ratio / minimum
        return maximum / area_ratio

    @staticmethod
    def _oversized_penalty(area_ratio: float, expected_maximum: float) -> float:
        if area_ratio <= expected_maximum:
            return 0.0
        return min(1.0, (area_ratio - expected_maximum) / expected_maximum)

    @staticmethod
    def _boundary_penalty(
        bbox: list[float],
        image_width: int,
        image_height: int,
    ) -> float:
        x1, y1, x2, y2 = bbox
        x_margin = image_width * 0.01
        y_margin = image_height * 0.01
        near_edges = sum(
            (
                x1 <= x_margin,
                y1 <= y_margin,
                x2 >= image_width - x_margin,
                y2 >= image_height - y_margin,
            )
        )
        return near_edges / 4.0

    def select(
        self,
        detections: list[dict[str, Any]],
        image_width: int,
        image_height: int,
        target_position: str = "any",
        expected_area_range: tuple[float, float] | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Score all valid detections and return a deterministic selection."""
        if not isinstance(detections, list):
            raise ValueError("detections must be a list")
        if isinstance(image_width, bool) or int(image_width) <= 0:
            raise ValueError("image_width must be a positive integer")
        if isinstance(image_height, bool) or int(image_height) <= 0:
            raise ValueError("image_height must be a positive integer")
        image_width = int(image_width)
        image_height = int(image_height)

        target_position = str(target_position).strip().lower()
        if target_position not in TARGET_POSITIONS:
            raise ValueError(f"unsupported target_position: {target_position!r}")
        area_range = self._validate_area_range(
            expected_area_range if expected_area_range is not None else self.expected_area_range
        )
        active_weights = (
            self._merge_weights(weights) if weights is not None else dict(self.weights)
        )
        target_point = TARGET_POSITIONS[target_position]
        image_area = float(image_width * image_height)

        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for candidate_index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                rejected.append(
                    {
                        "candidate_index": candidate_index,
                        "status": "rejected",
                        "reason": "detection must be a dictionary",
                    }
                )
                continue
            try:
                bbox, was_clipped = self._prepare_bbox(
                    detection.get("bbox"), image_width, image_height
                )
            except ValueError as exc:
                rejected.append(
                    {
                        "candidate_index": candidate_index,
                        "status": "rejected",
                        "reason": str(exc),
                    }
                )
                continue

            try:
                detection_score = float(detection.get("score", 0.0))
            except (TypeError, ValueError):
                detection_score = 0.0
            if detection_score != detection_score or abs(detection_score) == float("inf"):
                detection_score = 0.0
            normalized_detection_score = self._clamp(detection_score, 0.0, 1.0)

            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            bbox_area_ratio = bbox_area / image_area
            bbox_center_x = (x1 + x2) / 2.0
            bbox_center_y = (y1 + y2) / 2.0
            normalized_center_x = bbox_center_x / image_width
            normalized_center_y = bbox_center_y / image_height

            if target_point is None:
                position_distance = 0.0
                position_score = 1.0
            else:
                raw_distance = hypot(
                    normalized_center_x - target_point[0],
                    normalized_center_y - target_point[1],
                )
                position_distance = min(1.0, raw_distance / (2.0**0.5))
                position_score = 1.0 - position_distance

            area_score = self._area_score(bbox_area_ratio, *area_range)
            oversized_penalty = self._oversized_penalty(
                bbox_area_ratio, area_range[1]
            )
            boundary_penalty = self._boundary_penalty(
                bbox, image_width, image_height
            )
            selection_score = (
                active_weights["confidence"] * normalized_detection_score
                + active_weights["position"] * position_score
                + active_weights["area"] * area_score
                - active_weights["oversized"] * oversized_penalty
                - active_weights["boundary"] * boundary_penalty
            )

            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "bbox": bbox,
                    "bbox_was_clipped": was_clipped,
                    "label": str(detection.get("label", "unknown") or "unknown"),
                    "detection_score": detection_score,
                    "normalized_detection_score": normalized_detection_score,
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "bbox_area": bbox_area,
                    "bbox_area_ratio": bbox_area_ratio,
                    "bbox_center_x": bbox_center_x,
                    "bbox_center_y": bbox_center_y,
                    "normalized_center_x": normalized_center_x,
                    "normalized_center_y": normalized_center_y,
                    "target_position_distance": position_distance,
                    "position_distance": position_distance,
                    "center_score": position_score,
                    "position_score": position_score,
                    "area_score": area_score,
                    "oversized_penalty": oversized_penalty,
                    "boundary_penalty": boundary_penalty,
                    "final_selection_score": selection_score,
                    "selection_score": selection_score,
                    "selected": False,
                }
            )

        if not candidates:
            return {
                "selected": None,
                "candidates": [],
                "rejected_candidates": rejected,
                "selection_reason": "No valid detection candidate was available.",
                "status": "no_detection",
                "target_position": target_position,
                "expected_area_range": list(area_range),
                "weights": active_weights,
            }

        selected = max(
            candidates,
            key=lambda item: (
                item["selection_score"],
                item["detection_score"],
                -item["candidate_index"],
            ),
        )
        selected["selected"] = True
        selected_summary = {
            "bbox": selected["bbox"],
            "label": selected["label"],
            "detection_score": selected["detection_score"],
            "selection_score": selected["selection_score"],
            "candidate_index": selected["candidate_index"],
        }
        reason = (
            f"Selected candidate {selected['candidate_index']} with score "
            f"{selected['selection_score']:.4f}: detection="
            f"{selected['normalized_detection_score']:.4f}, position="
            f"{selected['position_score']:.4f}, area={selected['area_score']:.4f}, "
            f"oversized_penalty={selected['oversized_penalty']:.4f}, "
            f"boundary_penalty={selected['boundary_penalty']:.4f}."
        )
        return {
            "selected": selected_summary,
            "candidates": candidates,
            "rejected_candidates": rejected,
            "selection_reason": reason,
            "status": "success",
            "target_position": target_position,
            "expected_area_range": list(area_range),
            "weights": active_weights,
        }
