from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from utils.output_manager import create_run_output, write_run_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALID_STATUSES = {"correct", "error"}


def _validate_bbox(bbox: Any, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """
    Validate and clamp a bounding box.

    Expected bbox format:
        [x1, y1, x2, y2]

    Returns:
        A clamped integer tuple: (x1, y1, x2, y2)

    Raises:
        ValueError: If bbox is invalid or has no visible area.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be a list or tuple with four values: [x1, y1, x2, y2]")

    try:
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox values must be numeric") from exc

    # Accept reversed coordinates and normalize them.
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    # Clamp coordinates to the image boundary.
    left = max(0, min(left, image_width - 1))
    right = max(0, min(right, image_width - 1))
    top = max(0, min(top, image_height - 1))
    bottom = max(0, min(bottom, image_height - 1))

    if left >= right or top >= bottom:
        raise ValueError(
            f"bbox has no visible area after clamping: [{left}, {top}, {right}, {bottom}]"
        )

    return left, top, right, bottom


def _build_label(annotation: dict[str, Any]) -> str:
    """
    Build the label shown above the bounding box.

    correct:
        part_id

    error:
        part_id: error_type
    """
    part_id = str(annotation.get("part_id", "unknown_part")).strip() or "unknown_part"
    status = str(annotation.get("status", "")).strip().lower()

    if status == "correct":
        return part_id

    error_type = str(annotation.get("error_type", "error")).strip() or "error"
    return f"{part_id}: {error_type}"


def _draw_label(
    image: Any,
    label: str,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    """
    Draw a readable filled label background and text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    padding = 5

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    label_top = max(0, y1 - text_height - baseline - padding * 2)
    label_bottom = min(image.shape[0] - 1, y1)
    label_right = min(image.shape[1] - 1, x1 + text_width + padding * 2)

    cv2.rectangle(
        image,
        (x1, label_top),
        (label_right, label_bottom),
        color,
        thickness=-1,
    )

    text_x = x1 + padding
    text_y = max(
        text_height + padding,
        label_bottom - baseline - padding,
    )

    cv2.putText(
        image,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def annotate_image(
    image_path: str,
    annotations: list[dict[str, Any]],
    output_dir: str | None = None,
) -> str:
    """
    Draw bounding boxes and labels on an image.

    Args:
        image_path:
            Path to the source image.

        annotations:
            A list of annotation dictionaries.

            Required fields:
                part_id: str
                bbox: [x1, y1, x2, y2]
                status: "correct" or "error"

            Optional fields:
                error_type: str
                    Used in the label when status == "error".

        output_dir:
            Optional output directory.
            Defaults to a standard output/vision/annotations/<run_id>/images run.

    Returns:
        Absolute path of the annotated image.

    Raises:
        FileNotFoundError:
            If the input image does not exist.

        ValueError:
            If annotations are invalid or the image cannot be decoded.

        OSError:
            If the output image cannot be written.
    """
    source_path = Path(image_path).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Image not found: {source_path}")

    if not source_path.is_file():
        raise ValueError(f"Image path is not a file: {source_path}")

    if not isinstance(annotations, list):
        raise ValueError("annotations must be a list")

    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {source_path}")

    image_height, image_width = image.shape[:2]

    # OpenCV uses BGR.
    status_colors = {
        "correct": (0, 180, 0),
        "error": (0, 0, 255),
    }

    for index, annotation in enumerate(annotations, start=1):
        if not isinstance(annotation, dict):
            raise ValueError(f"annotation #{index} must be a dictionary")

        status = str(annotation.get("status", "")).strip().lower()
        if status not in VALID_STATUSES:
            raise ValueError(
                f"annotation #{index} has invalid status '{status}'. "
                f"Expected one of: {sorted(VALID_STATUSES)}"
            )

        bbox = _validate_bbox(
            annotation.get("bbox"),
            image_width=image_width,
            image_height=image_height,
        )

        x1, y1, x2, y2 = bbox
        color = status_colors[status]
        label = _build_label(annotation)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=3,
        )
        _draw_label(image, label, x1, y1, color)

    run_paths = None
    if output_dir:
        target_dir = Path(output_dir).expanduser().resolve()
    else:
        run_paths = create_run_output(
            "vision",
            "annotations",
            output_root=PROJECT_ROOT / "output",
        )
        target_dir = run_paths.images_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / f"{source_path.stem}_annotated{source_path.suffix}"

    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to write annotated image: {output_path}")

    if run_paths is not None:
        write_run_summary(
            run_paths,
            status="completed",
            input_count=1,
            success_count=1,
            failure_count=0,
            parameters={"annotation_count": len(annotations)},
            output_paths={"annotated_image": str(output_path)},
        )

    return str(output_path)


if __name__ == "__main__":
    # Replace this path with an actual project image when testing.
    test_image = PROJECT_ROOT / "regression_subset" / "model03_step01_correct-01_front_01.jpg"

    test_annotations = [
        {
            "part_id": "part_01",
            "bbox": [60, 60, 220, 190],
            "status": "correct",
            "error_type": "correct",
        },
        {
            "part_id": "part_02",
            "bbox": [240, 100, 390, 240],
            "status": "error",
            "error_type": "wrongpart",
        },
    ]

    try:
        result_path = annotate_image(
            image_path=str(test_image),
            annotations=test_annotations,
        )
        print(f"Annotated image saved to: {result_path}")
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Annotation failed: {exc}")
