"""Offline paired-image ROI localization and candidate-package construction."""

from __future__ import annotations

import json
from math import hypot, log
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from utils.bbox_candidate_selector import BBoxCandidateSelector
from utils.image_annotator import annotate_image
from utils.roi_candidate_builder import build_roi_candidates, part_family


COLOR_RANGES = {
    "RED": (((0, 10), 100, 55), ((170, 180), 100, 55)),
    "ORANGE": (((8, 22), 100, 55),),
    "YELLOW": (((18, 40), 90, 55),),
    "GREEN": (((38, 95), 65, 40),),
    "BLUE": (((95, 138), 65, 45),),
    "WHITE": (((0, 180), 0, 155),),
}


class ROIIdentityPipeline:
    def __init__(
        self, *, detector: Any | None = None,
        selector: BBoxCandidateSelector | None = None,
        minimum_localization_score: float = 0.45,
        working_size: int = 800,
    ) -> None:
        self.detector = detector
        self.selector = selector or BBoxCandidateSelector(expected_area_range=(0.0003, 0.08))
        self.minimum_localization_score = float(minimum_localization_score)
        self.working_size = int(working_size)

    @staticmethod
    def _load(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        return json.loads(Path(value).read_text(encoding="utf-8"))

    @staticmethod
    def _position(cx: float, cy: float) -> str:
        if cx < 0.38:
            return "LEFT"
        if cx > 0.62:
            return "RIGHT"
        if cy < 0.30:
            return "TOP"
        if cy > 0.70:
            return "BOTTOM"
        return "CENTER"

    @staticmethod
    def _shape_family(width: int, height: int, area_ratio: float, circularity: float) -> str:
        aspect = width / max(height, 1)
        if circularity >= 0.62 and 0.62 <= aspect <= 1.62:
            return "wheel" if area_ratio >= 0.012 else "pin"
        if max(aspect, 1.0 / max(aspect, 0.001)) >= 1.55:
            return "bar" if area_ratio >= 0.018 else "pin"
        return "pin" if area_ratio < 0.006 else "unknown"

    def _components(self, image: np.ndarray) -> tuple[dict[str, list[dict[str, Any]]], tuple[int, int]]:
        resized = cv2.resize(image, (self.working_size, self.working_size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        saturation, value = hsv[:, :, 1], hsv[:, :, 2]
        colored = ((saturation > 60) & (value > 40)).astype(np.uint8) * 255
        points = cv2.findNonZero(colored)
        assembly = cv2.boundingRect(points) if points is not None else (0, 0, self.working_size, self.working_size)
        ax, ay, aw, ah = assembly
        assembly_area = float(max(1, aw * ah))
        result: dict[str, list[dict[str, Any]]] = {}
        for color, ranges in COLOR_RANGES.items():
            mask = np.zeros((self.working_size, self.working_size), dtype=np.uint8)
            for (low_h, high_h), minimum_saturation, minimum_value in ranges:
                if color == "WHITE":
                    selected = (saturation < 45) & (value > minimum_value)
                else:
                    selected = (
                        (hsv[:, :, 0] >= low_h) & (hsv[:, :, 0] <= high_h)
                        & (saturation >= minimum_saturation) & (value >= minimum_value)
                    )
                mask[selected] = 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            components = []
            for index in range(1, count):
                x, y, width, height, area = (int(value) for value in stats[index])
                area_ratio = area / assembly_area
                if area < 80 or area_ratio > 0.16 or width < 6 or height < 6:
                    continue
                component_mask = (labels == index).astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4.0 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter) if perimeter else 0.0
                cx = ((x + width / 2.0) - ax) / max(aw, 1)
                cy = ((y + height / 2.0) - ay) / max(ah, 1)
                components.append({
                    "color": color, "bbox": [x, y, x + width, y + height],
                    "center": [cx, cy], "area_ratio": area_ratio,
                    "circularity": float(circularity),
                    "shape_family": self._shape_family(width, height, area_ratio, float(circularity)),
                    "position": self._position(cx, cy),
                    "component_score": min(1.0, 0.35 + min(area_ratio / 0.02, 1.0) * 0.25 + min(float(circularity), 1.0) * 0.20),
                })
            result[color] = sorted(components, key=lambda item: (item["center"][0], item["center"][1], item["area_ratio"]))
        return result, (image.shape[1], image.shape[0])

    @staticmethod
    def _match_cost(left: dict[str, Any], right: dict[str, Any]) -> float:
        position = hypot(left["center"][0] - right["center"][0], left["center"][1] - right["center"][1])
        area = abs(log(max(left["area_ratio"], 1e-6) / max(right["area_ratio"], 1e-6)))
        family = 0.0 if left["shape_family"] == right["shape_family"] else 0.20
        return position + 0.12 * area + family

    def _delta_evidence(
        self, reference: dict[str, list[dict[str, Any]]], test: dict[str, list[dict[str, Any]]], error_type: str,
    ) -> list[dict[str, Any]]:
        evidence = []
        normalized = str(error_type).lower().replace("_", "")
        for color in sorted(COLOR_RANGES):
            ref_items, test_items = reference[color], test[color]
            pairs = sorted(
                ((self._match_cost(left, right), li, ri) for li, left in enumerate(ref_items) for ri, right in enumerate(test_items)),
                key=lambda item: (item[0], item[1], item[2]),
            )
            used_left: set[int] = set()
            used_right: set[int] = set()
            for cost, left_index, right_index in pairs:
                if cost > 0.32 or left_index in used_left or right_index in used_right:
                    continue
                used_left.add(left_index); used_right.add(right_index)
            unmatched_ref = [item for index, item in enumerate(ref_items) if index not in used_left]
            unmatched_test = [item for index, item in enumerate(test_items) if index not in used_right]
            selected: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
            if normalized in {"missingpart", "missing"}:
                selected = [("reference_only", item, None) for item in unmatched_ref]
            elif normalized in {"wrongpart", "wrong"}:
                selected = [("reference_only", item, None) for item in unmatched_ref]
                selected += [("test_only", item, None) for item in unmatched_test]
            for relation, item, counterpart in selected:
                nearest = min((self._match_cost(item, other) for other in (test_items if relation == "reference_only" else ref_items)), default=1.0)
                score = min(1.0, float(item["component_score"]) * 0.65 + min(nearest / 0.45, 1.0) * 0.35)
                evidence.append({
                    **item, "status": "success", "score": score,
                    "difference_relation": relation,
                    "reference_bbox": item["bbox"] if relation == "reference_only" else None,
                    "test_bbox": item["bbox"] if relation == "test_only" else None,
                    "matching_source": "paired_color_component_delta",
                })
        return sorted(evidence, key=lambda item: (-item["score"], item["color"], item["position"], item["bbox"]))

    def _dino_corroboration(
        self, image_path: Path, prompt: str, target_position: str,
    ) -> dict[str, Any]:
        if self.detector is None or not prompt:
            return {"status": "not_run", "selected_bbox": None, "score": 0.0}
        try:
            detections = self.detector.detect(
                image_path=str(image_path), text_prompt=prompt,
                box_threshold=0.15, text_threshold=0.10, max_detections=20,
            )
        except Exception as exc:
            return {"status": "error", "selected_bbox": None, "score": 0.0, "error": f"{type(exc).__name__}: {exc}"}
        image = cv2.imread(str(image_path))
        selection = self.selector.select(detections, image.shape[1], image.shape[0], target_position=target_position.lower())
        selected = selection.get("selected")
        return {
            "status": "success" if selected else "no_detection",
            "selected_bbox": selected.get("bbox") if selected else None,
            "score": selected.get("selection_score", 0.0) if selected else 0.0,
            "detection_score": selected.get("detection_score", 0.0) if selected else 0.0,
            "selection": selection,
        }

    @staticmethod
    def _scale_bbox(bbox: list[float], original_size: tuple[int, int], working_size: int) -> list[int]:
        width, height = original_size
        return [
            int(round(bbox[0] * width / working_size)), int(round(bbox[1] * height / working_size)),
            int(round(bbox[2] * width / working_size)), int(round(bbox[3] * height / working_size)),
        ]

    def run(
        self, *, test_image: str | Path, reference_image: str | Path,
        model_id: str, step_id: str, view_angle: str, error_type: str,
        expected_state: Mapping[str, Any] | str | Path,
        part_library: Mapping[str, Any] | str | Path,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        test_path, reference_path = Path(test_image).resolve(), Path(reference_image).resolve()
        test_image_data, reference_image_data = cv2.imread(str(test_path)), cv2.imread(str(reference_path))
        if test_image_data is None or reference_image_data is None:
            raise ValueError("Test/reference image could not be decoded")
        test_components, test_size = self._components(test_image_data)
        reference_components, reference_size = self._components(reference_image_data)
        evidence = self._delta_evidence(reference_components, test_components, error_type)
        candidate = build_roi_candidates(
            expected_state=expected_state, part_library=part_library, roi_evidence=evidence,
            error_type=error_type, view_angle=view_angle,
            minimum_localization_score=self.minimum_localization_score,
        )
        reliable = [item for item in evidence if item["score"] >= self.minimum_localization_score]
        candidate_roi_items = []
        for candidate_item in candidate["candidate_evidence"]:
            for roi_item in candidate_item.get("roi_evidence", [])[:1]:
                candidate_roi_items.append({**roi_item, "candidate_part_id": candidate_item["part_id"]})
        localization_score = max((item["score"] for item in reliable), default=0.0)
        localization_status = "success" if candidate["candidate_part_ids"] and candidate_roi_items else "insufficient_evidence"
        output = Path(output_dir).resolve() if output_dir is not None else None
        test_rois: list[str] = []
        reference_rois: list[str] = []
        bboxes = []
        if localization_status == "success" and output is not None:
            output.mkdir(parents=True, exist_ok=True)
            annotations_test, annotations_reference = [], []
            for index, item in enumerate(candidate_roi_items[:6], start=1):
                label = item["candidate_part_id"]
                if item.get("reference_bbox"):
                    bbox = self._scale_bbox(item["reference_bbox"], reference_size, self.working_size)
                    annotations_reference.append({"part_id": label, "bbox": bbox, "status": "error", "error_type": error_type})
                    crop = reference_image_data[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    path = output / f"reference_roi_{index:02d}.png"; cv2.imwrite(str(path), crop); reference_rois.append(str(path))
                    bboxes.append({"role": "reference", "bbox": bbox, "color": item["color"], "score": item["score"], "candidate_part_id": label})
                if item.get("test_bbox"):
                    bbox = self._scale_bbox(item["test_bbox"], test_size, self.working_size)
                    annotations_test.append({"part_id": label, "bbox": bbox, "status": "error", "error_type": error_type})
                    crop = test_image_data[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    path = output / f"test_roi_{index:02d}.png"; cv2.imwrite(str(path), crop); test_rois.append(str(path))
                    bboxes.append({"role": "test", "bbox": bbox, "color": item["color"], "score": item["score"], "candidate_part_id": label})
            if annotations_reference:
                annotate_image(str(reference_path), annotations_reference, str(output))
            if annotations_test:
                annotate_image(str(test_path), annotations_test, str(output))
        library = self._load(part_library)
        prompt = ". ".join(
            max((str(alias) for alias in library.get(part_id, []) if str(alias).isascii()), key=len, default=part_id)
            for part_id in candidate["candidate_part_ids"]
        )
        target_position = reliable[0]["position"] if reliable else "CENTER"
        dino = self._dino_corroboration(reference_path, prompt, target_position)
        return {
            "model_id": model_id, "step_id": step_id, "view_angle": view_angle, "error_type": error_type,
            "test_image": str(test_path), "reference_image": str(reference_path),
            "test_roi": test_rois, "reference_roi": reference_rois, "bbox": bboxes,
            "localization_score": localization_score, "localization_status": localization_status,
            "candidate_part_ids": candidate["candidate_part_ids"], "candidate_count": candidate["candidate_count"],
            "full_candidate_count": candidate["full_candidate_count"], "reduction_ratio": candidate["reduction_ratio"],
            "roi_evidence": reliable, "candidate_evidence": candidate["candidate_evidence"],
            "dino_corroboration": dino,
            "supports_paired_roi": str(error_type).lower().replace("_", "") in {"wrongpart", "wrong"},
            "requires_manual_review": True,
            "human_ground_truth_used": False,
        }
