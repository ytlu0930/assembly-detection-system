from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from utils.bbox_candidate_selector import BBoxCandidateSelector
from utils.grounding_detector import DEFAULT_MODEL_ID, GroundingDetector
from utils.image_annotator import annotate_image
from utils.output_manager import create_run_output, write_run_summary


class LocalizationPipeline:
    """Independent Grounding DINO -> selector -> annotator PoC pipeline."""

    def __init__(
        self,
        detector: GroundingDetector | None = None,
        selector: BBoxCandidateSelector | None = None,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
    ) -> None:
        normalized_device = (
            None
            if device is None or str(device).strip().lower() == "auto"
            else str(device).strip()
        )
        self.detector = detector or GroundingDetector(
            model_id=model_id, device=normalized_device
        )
        self.selector = selector or BBoxCandidateSelector()

    def localize(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.15,
        text_threshold: float = 0.10,
        target_position: str = "center",
        max_detections: int | None = 10,
        output_dir: str | None = None,
        expected_area_range: tuple[float, float] | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run localization once and always return a serializable status result."""
        total_started = perf_counter()
        internal_paths = None
        annotation_output_dir = output_dir
        if output_dir is None:
            internal_paths = create_run_output(
                "pipeline",
                "localization_pipeline",
                output_root=Path(__file__).resolve().parents[1] / "output",
            )
            annotation_output_dir = str(internal_paths.images_dir)
        result: dict[str, Any] = {
            "all_detections": [],
            "image_width": None,
            "image_height": None,
            "selection_result": None,
            "selected_bbox": None,
            "selected_label": None,
            "selected_detection_score": None,
            "selected_selection_score": None,
            "annotated_image_path": None,
            "model_load_time": self.detector.model_load_time,
            "inference_time": None,
            "selection_time": None,
            "total_time": None,
            "status": "error",
            "error_message": None,
        }
        try:
            source_path = Path(image_path).expanduser().resolve()
            with Image.open(source_path) as image:
                image_width, image_height = image.size
            result["image_width"] = image_width
            result["image_height"] = image_height

            inference_started = perf_counter()
            detections = self.detector.detect(
                image_path=str(source_path),
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_detections=max_detections,
            )
            result["inference_time"] = perf_counter() - inference_started
            result["all_detections"] = detections

            selection_started = perf_counter()
            selection_result = self.selector.select(
                detections=detections,
                image_width=image_width,
                image_height=image_height,
                target_position=target_position,
                expected_area_range=expected_area_range,
                weights=weights,
            )
            result["selection_time"] = perf_counter() - selection_started
            result["selection_result"] = selection_result

            selected = selection_result["selected"]
            if selected is None:
                result["status"] = "no_detection"
                return result

            result["selected_bbox"] = selected["bbox"]
            result["selected_label"] = selected["label"]
            result["selected_detection_score"] = selected["detection_score"]
            result["selected_selection_score"] = selected["selection_score"]

            annotation = {
                "part_id": f"selector_{selected['candidate_index']:02d}",
                "bbox": selected["bbox"],
                "status": "correct",
                "error_type": "correct",
            }
            result["annotated_image_path"] = annotate_image(
                image_path=str(source_path),
                annotations=[annotation],
                output_dir=annotation_output_dir,
            )
            result["status"] = "success"
        except Exception as exc:
            result["error_message"] = f"{type(exc).__name__}: {exc}"
        finally:
            result["total_time"] = perf_counter() - total_started
            if internal_paths is not None:
                with internal_paths.json_path.open("w", encoding="utf-8") as handle:
                    json.dump(result, handle, ensure_ascii=False, indent=2)
                failed = result["status"] not in {"success", "no_detection"}
                write_run_summary(
                    internal_paths,
                    status="failed" if failed else "completed",
                    input_count=1,
                    success_count=0 if failed else 1,
                    failure_count=1 if failed else 0,
                    parameters={
                        "prompt": text_prompt,
                        "box_threshold": box_threshold,
                        "text_threshold": text_threshold,
                        "target_position": target_position,
                        "max_detections": max_detections,
                    },
                    timing={
                        "model_load_seconds": result["model_load_time"],
                        "inference_seconds": result["inference_time"],
                        "selection_seconds": result["selection_time"],
                        "total_seconds": result["total_time"],
                    },
                    output_paths={
                        "results_json": str(internal_paths.json_path),
                        "annotated_image": result["annotated_image_path"],
                    },
                    runtime={"device": self.detector.device},
                    notes=[result["error_message"]] if result["error_message"] else [],
                )
        return result
