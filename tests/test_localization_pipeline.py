from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import transformers

from utils.grounding_detector import DEFAULT_MODEL_ID
from utils.localization_pipeline import LocalizationPipeline
from utils.output_manager import resolve_run_output, write_run_summary


DEFAULT_IMAGE = (
    PROJECT_ROOT / "regression_subset" / "model03_step01_correct-01_front_01.jpg"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 8 localization pipeline.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
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


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_run_output(
        "pipeline",
        "localization_pipeline",
        output_dir=args.output_dir,
        output_root=PROJECT_ROOT / "output",
    )
    pipeline = LocalizationPipeline(model_id=args.model_id, device=args.device)
    result = pipeline.localize(
        image_path=str(args.image),
        text_prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_position=args.target_position,
        max_detections=args.max_detections,
        output_dir=str(paths.images_dir),
    )

    with paths.json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    with paths.csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(
            {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in result.items()
            }
        )

    failed = result["status"] not in {"success", "no_detection"}
    write_run_summary(
        paths,
        status="failed" if failed else "completed",
        input_count=1,
        success_count=0 if failed else 1,
        failure_count=1 if failed else 0,
        parameters={
            "image": str(args.image.expanduser().resolve()),
            "prompt": args.prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "target_position": args.target_position,
            "max_detections": args.max_detections,
        },
        timing={
            "model_load_seconds": result["model_load_time"],
            "inference_seconds": result["inference_time"],
            "selection_milliseconds": (
                result["selection_time"] * 1000
                if result["selection_time"] is not None
                else None
            ),
            "total_seconds": result["total_time"],
        },
        runtime={
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": pipeline.detector.device,
        },
        output_paths={
            "results_json": str(paths.json_path),
            "results_csv": str(paths.csv_path),
            "annotated_image": result["annotated_image_path"],
        },
        notes=[result["error_message"]] if result["error_message"] else [],
    )

    detections = result["all_detections"]
    selection = result["selection_result"] or {}
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"torch_version: {torch.__version__}")
    print(f"transformers_version: {transformers.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_version: {torch.version.cuda}")
    print(
        "gpu_name: "
        + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    )
    print(f"selected_device: {pipeline.detector.device}")
    print(f"model_id: {pipeline.detector.model_id}")
    print(f"image: {args.image.expanduser().resolve()}")
    print(f"prompt: {args.prompt}")
    print(f"box_threshold: {args.box_threshold}")
    print(f"text_threshold: {args.text_threshold}")
    print(f"target_position: {args.target_position}")
    print(f"detection_count: {len(detections)}")
    print(f"top1_detection: {json.dumps(detections[0] if detections else None)}")
    print(f"selected_candidate: {json.dumps(selection.get('selected'))}")
    print(f"selection_reason: {selection.get('selection_reason')}")
    print(f"model_load_time: {result['model_load_time']:.6f}s")
    print(f"inference_time: {result['inference_time']}")
    print(f"selection_time: {result['selection_time']}")
    print(f"output_image: {result['annotated_image_path']}")
    print(f"results_json: {paths.json_path}")
    print(f"results_csv: {paths.csv_path}")
    print(f"run_summary: {paths.summary_path}")
    print(f"status: {result['status']}")
    print(f"error_message: {result['error_message']}")
    return 0 if result["status"] in {"success", "no_detection"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
