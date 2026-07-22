from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single-image Grounding DINO PoC.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default="green vertical rectangular block")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-detections", type=int, default=5)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_run_output(
        "localization",
        "phase07_grounding_poc",
        output_dir=args.output_dir,
        output_root=PROJECT_ROOT / "output",
    )
    detector = GroundingDetector(model_id=args.model_id, device=args.device)
    inference_started = perf_counter()
    detections = detector.detect(
        image_path=str(args.image),
        text_prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_detections=args.max_detections,
    )
    inference_time = perf_counter() - inference_started

    print(f"model_id: {detector.model_id}")
    print(f"device: {detector.device}")
    print(f"image: {args.image.resolve()}")
    print(f"prompt: {args.prompt}")
    print(f"box_threshold: {args.box_threshold}")
    print(f"text_threshold: {args.text_threshold}")
    print(f"model_load_time: {detector.model_load_time:.3f}s")
    print(f"inference_time: {inference_time:.3f}s")
    print(f"detection_count: {len(detections)}")

    for index, detection in enumerate(detections, start=1):
        print(
            f"detection_{index}: label={detection['label']!r}, "
            f"score={detection['score']:.4f}, bbox={detection['bbox']}"
        )

    if not detections:
        print("No object detected; no annotated image was written.")
        write_run_summary(
            paths,
            status="completed",
            input_count=1,
            success_count=1,
            failure_count=0,
            parameters={
                "prompt": args.prompt,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "max_detections": args.max_detections,
            },
            timing={"inference_seconds": inference_time},
            runtime={
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": detector.device,
            },
            notes=["No detections; no annotated image was written."],
        )
        return 0

    annotations = [
        {
            "part_id": f"grounding_{index:02d}",
            "bbox": detection["bbox"],
            "status": "correct",
            "error_type": "correct",
        }
        for index, detection in enumerate(detections, start=1)
    ]
    output_image = annotate_image(
        image_path=str(args.image),
        annotations=annotations,
        output_dir=str(paths.images_dir),
    )
    write_run_summary(
        paths,
        status="completed",
        input_count=1,
        success_count=1,
        failure_count=0,
        parameters={
            "prompt": args.prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "max_detections": args.max_detections,
        },
        timing={"inference_seconds": inference_time},
        runtime={
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": detector.device,
        },
        output_paths={"annotated_image": output_image},
    )
    print(f"output_image: {output_image}")
    print(f"run_summary: {paths.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
