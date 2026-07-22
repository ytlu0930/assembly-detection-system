from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import transformers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from utils.bbox_candidate_selector import BBoxCandidateSelector
from utils.grounding_detector import DEFAULT_MODEL_ID, GroundingDetector
from utils.image_annotator import annotate_image
from utils.output_manager import resolve_run_output, write_run_summary


DEFAULT_INPUT_DIR = PROJECT_ROOT / "regression_subset"
FIELDNAMES = [
    "image_path",
    "prompt",
    "box_threshold",
    "text_threshold",
    "target_position",
    "detection_count",
    "top1_bbox",
    "top1_score",
    "selected_bbox",
    "selected_detection_score",
    "selected_selection_score",
    "selected_candidate_index",
    "bbox_area_ratio",
    "position_distance",
    "inference_time",
    "selection_time",
    "top1_output_image",
    "selector_output_image",
    "status",
    "error_message",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare top-1 with bbox selection.")
    parser.add_argument("--image", type=Path, action="append")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--prompt", default="lime green rectangular block in the center"
    )
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.10)
    parser.add_argument("--target-position", default="center")
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    return parser


def _image_paths(args: argparse.Namespace) -> list[Path]:
    if args.image:
        return [path.expanduser().resolve() for path in args.image]
    return sorted(args.input_dir.expanduser().resolve().glob("*.jpg"))


def _write_annotation(
    image_path: Path,
    bbox: list[float],
    part_id: str,
    output_dir: Path,
    final_name: str,
) -> str:
    generated = Path(
        annotate_image(
            str(image_path),
            [
                {
                    "part_id": part_id,
                    "bbox": bbox,
                    "status": "correct",
                    "error_type": "correct",
                }
            ],
            str(output_dir),
        )
    )
    final_path = generated.with_name(final_name + generated.suffix)
    generated.replace(final_path)
    return str(final_path)


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_run_output(
        "localization",
        "phase08_bbox_selection",
        output_dir=args.output_dir,
        output_root=PROJECT_ROOT / "output",
        image_subdirs=["detections", "selected_bbox"],
    )
    detection_output_dir = paths.image_subdirs["detections"]
    selection_output_dir = paths.image_subdirs["selected_bbox"]

    device = None if args.device.strip().lower() == "auto" else args.device
    detector = GroundingDetector(model_id=args.model_id, device=device)
    selector = BBoxCandidateSelector()
    rows: list[dict[str, Any]] = []

    for index, image_path in enumerate(_image_paths(args), start=1):
        row: dict[str, Any] = {name: None for name in FIELDNAMES}
        row.update(
            {
                "image_path": str(image_path),
                "prompt": args.prompt,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "target_position": args.target_position,
                "detection_count": 0,
                "status": "error",
                "error_message": None,
            }
        )
        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size

            inference_started = perf_counter()
            detections = detector.detect(
                str(image_path),
                args.prompt,
                args.box_threshold,
                args.text_threshold,
                args.max_detections,
            )
            row["inference_time"] = perf_counter() - inference_started
            row["detection_count"] = len(detections)

            selection_started = perf_counter()
            selection = selector.select(
                detections,
                image_width,
                image_height,
                target_position=args.target_position,
            )
            row["selection_time"] = perf_counter() - selection_started

            if detections:
                top1 = detections[0]
                row["top1_bbox"] = top1["bbox"]
                row["top1_score"] = top1["score"]
                row["top1_output_image"] = _write_annotation(
                    image_path,
                    top1["bbox"],
                    "top1",
                    detection_output_dir,
                    f"{image_path.stem}_top1",
                )

            selected = selection["selected"]
            if selected is not None:
                selected_detail = next(
                    item
                    for item in selection["candidates"]
                    if item["candidate_index"] == selected["candidate_index"]
                )
                row["selected_bbox"] = selected["bbox"]
                row["selected_detection_score"] = selected["detection_score"]
                row["selected_selection_score"] = selected["selection_score"]
                row["selected_candidate_index"] = selected["candidate_index"]
                row["bbox_area_ratio"] = selected_detail["bbox_area_ratio"]
                row["position_distance"] = selected_detail["position_distance"]
                row["selector_output_image"] = _write_annotation(
                    image_path,
                    selected["bbox"],
                    f"selector_{selected['candidate_index']}",
                    selection_output_dir,
                    f"{image_path.stem}_selector",
                )
                row["status"] = "success"
            else:
                row["status"] = "no_detection"
        except Exception as exc:
            row["error_message"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(
            f"[{index:02d}] {image_path.name}: status={row['status']} "
            f"detections={row['detection_count']} selected={row['selected_candidate_index']}"
        )

    json_path = paths.json_path
    csv_path = paths.csv_path
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["top1_bbox"] = json.dumps(row["top1_bbox"])
            csv_row["selected_bbox"] = json.dumps(row["selected_bbox"])
            writer.writerow(csv_row)

    failure_count = sum(row["status"] == "error" for row in rows)
    inference_times = [
        float(row["inference_time"])
        for row in rows
        if row["inference_time"] is not None
    ]
    selection_times = [
        float(row["selection_time"])
        for row in rows
        if row["selection_time"] is not None
    ]
    selector_changed = sum(
        row["selected_candidate_index"] not in {None, 0} for row in rows
    )
    write_run_summary(
        paths,
        status="completed" if not failure_count else "partial",
        input_count=len(rows),
        success_count=len(rows) - failure_count,
        failure_count=failure_count,
        parameters={
            "prompt": args.prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "target_position": args.target_position,
            "max_detections": args.max_detections,
        },
        timing={
            "average_inference_seconds": (
                sum(inference_times) / len(inference_times) if inference_times else None
            ),
            "average_selection_milliseconds": (
                1000 * sum(selection_times) / len(selection_times)
                if selection_times
                else None
            ),
        },
        runtime={
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": detector.device,
            "selector_changed_top1_count": selector_changed,
        },
        output_paths={
            "results_json": str(json_path),
            "results_csv": str(csv_path),
            "detections": str(detection_output_dir),
            "selected_bbox": str(selection_output_dir),
        },
    )

    print(f"json_results: {json_path}")
    print(f"csv_results: {csv_path}")
    print(f"run_summary: {paths.summary_path}")
    return 0 if all(row["status"] != "error" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
