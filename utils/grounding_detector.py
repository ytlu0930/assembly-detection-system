from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any


DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"


class GroundingDetector:
    """Small, reusable Transformers wrapper for Grounding DINO inference."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Grounding DINO dependencies are missing. Install requirements.txt first."
            ) from exc

        model_id = str(model_id).strip()
        if not model_id:
            raise ValueError("model_id must not be empty")

        self.model_id = model_id
        self.device = self._resolve_device(device, torch)

        load_started = perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        self.model_load_time = perf_counter() - load_started

    @staticmethod
    def _resolve_device(device: str | None, torch_module: Any) -> str:
        if device is None:
            return "cuda" if torch_module.cuda.is_available() else "cpu"

        requested = str(device).strip().lower()
        if not requested:
            raise ValueError("device must not be empty")
        if requested.startswith("cuda") and not torch_module.cuda.is_available():
            raise ValueError("CUDA was requested, but torch.cuda.is_available() is False")
        return requested

    @staticmethod
    def _validate_threshold(name: str, value: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number between 0 and 1") from exc
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return number

    def detect(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        max_detections: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return score-sorted pixel-coordinate detections for one image."""
        import torch
        from PIL import Image, UnidentifiedImageError

        source_path = Path(image_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Image not found: {source_path}")
        if not source_path.is_file():
            raise ValueError(f"Image path is not a file: {source_path}")

        prompt = str(text_prompt).strip()
        if not prompt:
            raise ValueError("text_prompt must not be empty")
        box_threshold = self._validate_threshold("box_threshold", box_threshold)
        text_threshold = self._validate_threshold("text_threshold", text_threshold)
        if max_detections is not None:
            if isinstance(max_detections, bool) or not isinstance(max_detections, int):
                raise ValueError("max_detections must be a positive integer or None")
            if max_detections <= 0:
                raise ValueError("max_detections must be a positive integer or None")

        try:
            with Image.open(source_path) as opened_image:
                image = opened_image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Pillow could not decode image: {source_path}") from exc

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        postprocess_kwargs = {
            "threshold": box_threshold,
            "text_threshold": text_threshold,
            "target_sizes": [image.size[::-1]],
        }
        try:
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.input_ids,
                **postprocess_kwargs,
            )[0]
        except TypeError:
            # Transformers <= 4.47 used positional input_ids and box_threshold.
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]

        labels = processed.get("text_labels")
        if labels is None:
            labels = processed.get("labels", [])

        image_width, image_height = image.size
        detections: list[dict[str, Any]] = []
        for box, score, label in zip(processed["boxes"], processed["scores"], labels):
            raw_box = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
            x1, y1, x2, y2 = (float(value) for value in raw_box)
            clamped_box = [
                max(0.0, min(x1, float(image_width))),
                max(0.0, min(y1, float(image_height))),
                max(0.0, min(x2, float(image_width))),
                max(0.0, min(y2, float(image_height))),
            ]
            score_value = float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)
            detections.append(
                {
                    "label": str(label),
                    "score": score_value,
                    "bbox": clamped_box,
                }
            )

        detections.sort(key=lambda item: item["score"], reverse=True)
        if max_detections is not None:
            detections = detections[:max_detections]
        return detections
