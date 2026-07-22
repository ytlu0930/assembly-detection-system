from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import transformers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.grounding_detector import DEFAULT_MODEL_ID, GroundingDetector
from utils.image_annotator import annotate_image
from utils.output_manager import resolve_run_output, write_run_summary


DEFAULT_IMAGE = (
    PROJECT_ROOT / "regression_subset" / "model03_step01_correct-01_front_01.jpg"
)
PROMPTS = [
    "green vertical rectangular block",
    "light green vertical rectangular plastic block",
    "lime green rectangular block in the center",
    "central light green plastic plate",
    "green toy construction piece",
]
THRESHOLDS = [(0.15, 0.10), (0.25, 0.20), (0.35, 0.25)]
FIELDNAMES = [
    "image_path",
    "prompt",
    "box_threshold",
    "text_threshold",
    "model_id",
    "device",
    "model_load_time",
    "inference_time",
    "detection_count",
    "detections",
    "top_score",
    "top_bbox",
    "output_image",
    "status",
    "error_message",
]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:50]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 5x3 Grounding DINO PoC matrix.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-detections", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    return parser


def _json_ready(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row)


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_run_output(
        "localization",
        "phase07_grounding_experiments",
        output_dir=args.output_dir,
        output_root=PROJECT_ROOT / "output",
    )
    image_dir = paths.images_dir

    detector = GroundingDetector(model_id=args.model_id, device=args.device)
    rows: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(PROMPTS, start=1):
        for box_threshold, text_threshold in THRESHOLDS:
            row: dict[str, Any] = {
                "image_path": str(args.image.expanduser().resolve()),
                "prompt": prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "model_id": detector.model_id,
                "device": detector.device,
                "model_load_time": round(detector.model_load_time, 6),
                "inference_time": None,
                "detection_count": 0,
                "detections": [],
                "top_score": None,
                "top_bbox": None,
                "output_image": None,
                "status": "error",
                "error_message": None,
            }
            started = perf_counter()
            try:
                detections = detector.detect(
                    image_path=str(args.image),
                    text_prompt=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    max_detections=args.max_detections,
                )
                row["inference_time"] = round(perf_counter() - started, 6)
                row["detection_count"] = len(detections)
                row["detections"] = detections
                row["status"] = "success" if detections else "no_detections"

                if detections:
                    row["top_score"] = detections[0]["score"]
                    row["top_bbox"] = detections[0]["bbox"]
                    filename_stem = (
                        f"p{prompt_index}_{_slugify(prompt)}_"
                        f"b{box_threshold:.2f}_t{text_threshold:.2f}"
                    )
                    annotations = [
                        {
                            "part_id": f"grounding_{index:02d}",
                            "bbox": detection["bbox"],
                            "status": "correct",
                            "error_type": "correct",
                        }
                        for index, detection in enumerate(detections, start=1)
                    ]
                    generated_path = Path(
                        annotate_image(
                            image_path=str(args.image),
                            annotations=annotations,
                            output_dir=str(image_dir),
                        )
                    )
                    final_path = generated_path.with_name(filename_stem + generated_path.suffix)
                    generated_path.replace(final_path)
                    row["output_image"] = str(final_path)
            except Exception as exc:  # Keep the matrix running and record each failed case.
                row["inference_time"] = round(perf_counter() - started, 6)
                row["error_message"] = f"{type(exc).__name__}: {exc}"

            rows.append(row)
            print(
                f"[{len(rows):02d}/15] prompt={prompt!r} "
                f"box={box_threshold:.2f} text={text_threshold:.2f} "
                f"status={row['status']} detections={row['detection_count']}"
            )

    json_path = paths.json_path
    csv_path = paths.csv_path
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([_json_ready(row) for row in rows], handle, ensure_ascii=False, indent=2)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["detections"] = json.dumps(row["detections"], ensure_ascii=False)
            csv_row["top_bbox"] = json.dumps(row["top_bbox"])
            writer.writerow(csv_row)

    failure_count = sum(row["status"] == "error" for row in rows)
    inference_times = [
        float(row["inference_time"])
        for row in rows
        if row["inference_time"] is not None
    ]
    write_run_summary(
        paths,
        status="completed" if not failure_count else "partial",
        input_count=len(rows),
        success_count=len(rows) - failure_count,
        failure_count=failure_count,
        parameters={
            "prompts": PROMPTS,
            "thresholds": THRESHOLDS,
            "max_detections": args.max_detections,
            "image": str(args.image.expanduser().resolve()),
        },
        timing={
            "average_inference_seconds": (
                sum(inference_times) / len(inference_times) if inference_times else None
            )
        },
        runtime={
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": detector.device,
        },
        output_paths={
            "results_json": str(json_path),
            "results_csv": str(csv_path),
            "images": str(image_dir),
        },
    )

    print(f"json_results: {json_path}")
    print(f"csv_results: {csv_path}")
    print(f"run_summary: {paths.summary_path}")
    return 0 if all(row["status"] != "error" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
