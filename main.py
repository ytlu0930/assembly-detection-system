"""
main.py

單張圖片的正式完整 Pipeline 入口。

流程
----
既有 Vision parsed JSON
    ↓
pipeline_smoke_test.process_one()
    ↓
results.json
    ↓
CorrectionSOPGenerator
    ↓
correction_sop.json
    ↓
StepPromptBuilderV2
    ↓
step_prompts_v2.json
    ↓
step_image_generator_v2.py
    ↓
generated_steps_v2/
    ↓
InstructionBookGenerator
    ↓
assembly_instruction_book.png

本程式不再使用舊版 flowchart_generator，也不在 main.py 內直接呼叫 Vision。
Vision 與 Localization 模組由組員程式負責；本程式負責把它們的既有結果
串接到 SOP、Prompt、GPT Image 與最終說明書。

單張測試
--------
指定 parsed JSON：

python main.py ^
  --parsed-json "logs\\current_parsed_json\\xxx_parsed_*.json"

只測前處理，不呼叫圖片 API：

python main.py ^
  --parsed-json "logs\\current_parsed_json\\xxx.json"

做圖片 dry-run：

python main.py ^
  --parsed-json "logs\\current_parsed_json\\xxx.json" ^
  --image-dry-run

正式生圖：

python main.py ^
  --parsed-json "logs\\current_parsed_json\\xxx.json" ^
  --generate-images ^
  --allow-manual-review

只測第一個生圖任務：

python main.py ^
  --parsed-json "logs\\current_parsed_json\\xxx.json" ^
  --generate-images ^
  --allow-manual-review ^
  --image-max-tasks 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from correction_sop_generator import CorrectionSOPGenerator
from instruction_book_generator import InstructionBookGenerator
from pipeline_smoke_test import (
    find_latest_parsed_json,
    image_stem_from_parsed_json,
    process_one,
)
from step_prompt_builder_v2 import StepPromptBuilderV2


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "single_runs"
STEP_IMAGE_GENERATOR_PATH = PROJECT_ROOT / "step_image_generator_v2.py"


@dataclass
class StageRecord:
    name: str
    status: str
    output_path: Optional[str] = None
    elapsed_seconds: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class PipelineManifest:
    schema_version: str
    created_at: str
    finished_at: Optional[str]
    parsed_json_path: str
    image_stem: str
    output_dir: str
    generate_images: bool
    image_dry_run: bool
    allow_manual_review: bool
    overwrite: bool
    status: str
    final_instruction_path: Optional[str]
    elapsed_seconds: float
    stages: list[StageRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def stage_success(
    name: str,
    output_path: Optional[Path],
    started_at: float,
    *,
    status: str = "success",
) -> StageRecord:
    return StageRecord(
        name=name,
        status=status,
        output_path=str(output_path) if output_path else None,
        elapsed_seconds=round(time.perf_counter() - started_at, 3),
    )


def stage_failure(
    name: str,
    exc: Exception,
    started_at: float,
) -> StageRecord:
    return StageRecord(
        name=name,
        status="failed",
        elapsed_seconds=round(time.perf_counter() - started_at, 3),
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


def run_step_image_generator(
    *,
    prompts_json_path: Path,
    output_dir: Path,
    dry_run: bool,
    allow_manual_review: bool,
    overwrite: bool,
    quality: str,
    size: str,
    max_tasks: Optional[int],
    continue_on_error: bool,
) -> Path:
    if not STEP_IMAGE_GENERATOR_PATH.is_file():
        raise FileNotFoundError(
            f"step_image_generator_v2.py not found: {STEP_IMAGE_GENERATOR_PATH}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(STEP_IMAGE_GENERATOR_PATH),
        "--prompts-json",
        str(prompts_json_path),
        "--output-dir",
        str(output_dir),
        "--quality",
        quality,
        "--size",
        size,
    ]

    if dry_run:
        command.append("--dry-run")

    if allow_manual_review:
        command.append("--allow-manual-review")

    if overwrite:
        command.append("--overwrite")

    if max_tasks is not None:
        command.extend(["--max-tasks", str(max_tasks)])

    if continue_on_error:
        command.append("--continue-on-error")

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    stdout_path = output_dir / "main_image_generator_stdout.txt"
    stderr_path = output_dir / "main_image_generator_stderr.txt"

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    if completed.stdout.strip():
        print(completed.stdout)

    if completed.returncode != 0:
        error_tail = completed.stderr.strip()[-2500:]
        raise RuntimeError(
            "step_image_generator_v2.py failed with "
            f"exit code {completed.returncode}.\n{error_tail}"
        )

    manifest_path = output_dir / "generation_manifest_v2.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Image generator completed, but generation_manifest_v2.json "
            f"was not created: {manifest_path}"
        )

    return manifest_path


def run_pipeline(
    *,
    parsed_json_path: str | Path,
    output_dir: str | Path,
    generate_images: bool = False,
    image_dry_run: bool = False,
    allow_manual_review: bool = False,
    overwrite: bool = False,
    image_quality: str = "medium",
    image_size: str = "1024x1024",
    image_max_tasks: Optional[int] = None,
    image_continue_on_error: bool = False,
    book_columns: int = 1,
    create_book_with_placeholders: bool = False,
) -> PipelineManifest:
    if generate_images and image_dry_run:
        raise ValueError(
            "generate_images and image_dry_run cannot both be enabled."
        )

    parsed_path = Path(parsed_json_path).expanduser().resolve()
    case_dir = Path(output_dir).expanduser().resolve()
    case_dir.mkdir(parents=True, exist_ok=True)

    image_stem = image_stem_from_parsed_json(parsed_path)
    manifest_path = case_dir / "pipeline_manifest.json"

    manifest = PipelineManifest(
        schema_version="1.0",
        created_at=datetime.now().astimezone().isoformat(),
        finished_at=None,
        parsed_json_path=str(parsed_path),
        image_stem=image_stem,
        output_dir=str(case_dir),
        generate_images=generate_images,
        image_dry_run=image_dry_run,
        allow_manual_review=allow_manual_review,
        overwrite=overwrite,
        status="running",
        final_instruction_path=None,
        elapsed_seconds=0.0,
        stages=[],
    )

    pipeline_started = time.perf_counter()

    try:
        # --------------------------------------------------
        # Stage 1: 組員既有 Vision JSON + Localization
        # --------------------------------------------------
        started = time.perf_counter()
        try:
            results_path = process_one(
                parsed_json_path=parsed_path,
                output_dir=case_dir,
                overwrite=overwrite,
            )
            manifest.stages.append(
                stage_success(
                    "pipeline_smoke_test",
                    results_path,
                    started,
                )
            )
        except Exception as exc:
            manifest.stages.append(
                stage_failure(
                    "pipeline_smoke_test",
                    exc,
                    started,
                )
            )
            raise

        # --------------------------------------------------
        # Stage 2: Correction SOP
        # --------------------------------------------------
        started = time.perf_counter()
        sop_path = case_dir / "correction_sop.json"

        try:
            if sop_path.is_file() and not overwrite:
                manifest.stages.append(
                    stage_success(
                        "correction_sop_generator",
                        sop_path,
                        started,
                        status="existing",
                    )
                )
            else:
                sop_generator = CorrectionSOPGenerator()
                sop = sop_generator.generate_from_results(results_path)
                sop_path, _ = sop_generator.save(sop, case_dir)

                manifest.stages.append(
                    stage_success(
                        "correction_sop_generator",
                        sop_path,
                        started,
                    )
                )
        except Exception as exc:
            manifest.stages.append(
                stage_failure(
                    "correction_sop_generator",
                    exc,
                    started,
                )
            )
            raise

        # --------------------------------------------------
        # Stage 3: Step Prompt Builder V2
        # --------------------------------------------------
        started = time.perf_counter()
        prompts_path = case_dir / "step_prompts_v2.json"

        try:
            if prompts_path.is_file() and not overwrite:
                manifest.stages.append(
                    stage_success(
                        "step_prompt_builder_v2",
                        prompts_path,
                        started,
                        status="existing",
                    )
                )
            else:
                prompt_builder = StepPromptBuilderV2(
                    block_on_manual_review=not allow_manual_review
                )
                package = prompt_builder.build_from_sop(sop_path)
                prompts_path, _ = prompt_builder.save(package, case_dir)

                manifest.stages.append(
                    stage_success(
                        "step_prompt_builder_v2",
                        prompts_path,
                        started,
                    )
                )
        except Exception as exc:
            manifest.stages.append(
                stage_failure(
                    "step_prompt_builder_v2",
                    exc,
                    started,
                )
            )
            raise

        # --------------------------------------------------
        # Stage 4: GPT Image 2 / Dry Run
        # --------------------------------------------------
        generated_dir = case_dir / "generated_steps_v2"

        if generate_images or image_dry_run:
            started = time.perf_counter()

            try:
                image_manifest_path = run_step_image_generator(
                    prompts_json_path=prompts_path,
                    output_dir=generated_dir,
                    dry_run=image_dry_run,
                    allow_manual_review=allow_manual_review,
                    overwrite=overwrite,
                    quality=image_quality,
                    size=image_size,
                    max_tasks=image_max_tasks,
                    continue_on_error=image_continue_on_error,
                )

                manifest.stages.append(
                    stage_success(
                        "step_image_generator_v2",
                        image_manifest_path,
                        started,
                        status="dry_run" if image_dry_run else "success",
                    )
                )

            except Exception as exc:
                manifest.stages.append(
                    stage_failure(
                        "step_image_generator_v2",
                        exc,
                        started,
                    )
                )
                raise
        else:
            manifest.stages.append(
                StageRecord(
                    name="step_image_generator_v2",
                    status="skipped",
                    output_path=str(generated_dir),
                    error_message=(
                        "Image generation disabled. Use --generate-images "
                        "or --image-dry-run."
                    ),
                )
            )

        # --------------------------------------------------
        # Stage 5: 一頁式說明書
        # --------------------------------------------------
        should_generate_book = (
            generate_images
            or create_book_with_placeholders
            or (generated_dir / "generation_manifest_v2.json").is_file()
        )

        if should_generate_book:
            started = time.perf_counter()
            book_path = case_dir / "assembly_instruction_book.png"

            try:
                if book_path.is_file() and not overwrite:
                    manifest.stages.append(
                        stage_success(
                            "instruction_book_generator",
                            book_path,
                            started,
                            status="existing",
                        )
                    )
                else:
                    book_generator = InstructionBookGenerator(
                        columns=book_columns
                    )
                    book_path = book_generator.generate(
                        prompts_json_path=prompts_path,
                        output_path=book_path,
                        overwrite=overwrite,
                    )

                    manifest.stages.append(
                        stage_success(
                            "instruction_book_generator",
                            book_path,
                            started,
                        )
                    )

                manifest.final_instruction_path = str(book_path)

            except Exception as exc:
                manifest.stages.append(
                    stage_failure(
                        "instruction_book_generator",
                        exc,
                        started,
                    )
                )
                raise
        else:
            manifest.stages.append(
                StageRecord(
                    name="instruction_book_generator",
                    status="skipped",
                    error_message=(
                        "No generated images. Use "
                        "--create-book-with-placeholders to preview layout."
                    ),
                )
            )

        manifest.status = "success"

    except Exception:
        manifest.status = "failed"

    finally:
        manifest.finished_at = (
            datetime.now()
            .astimezone()
            .isoformat()
        )
        manifest.elapsed_seconds = round(
            time.perf_counter() - pipeline_started,
            3,
        )
        save_json(manifest_path, manifest.to_dict())

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete single-image assembly instruction pipeline."
        )
    )

    parser.add_argument(
        "--parsed-json",
        type=Path,
        default=None,
        help="Path to one existing *_parsed_*.json file.",
    )

    parser.add_argument(
        "--image-stem",
        type=str,
        default=None,
        help=(
            "Find the latest parsed JSON matching this original image stem."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Case output directory. Default: "
            "output/single_runs/<image_stem>/"
        ),
    )

    parser.add_argument(
        "--generate-images",
        action="store_true",
        help="Actually call gpt-image-2.",
    )

    parser.add_argument(
        "--image-dry-run",
        action="store_true",
        help="Validate image prompts and paths without calling the API.",
    )

    parser.add_argument(
        "--allow-manual-review",
        action="store_true",
        help=(
            "Override generation_allowed=false after manual confirmation."
        ),
    )

    parser.add_argument(
        "--image-quality",
        choices=["low", "medium", "high", "auto"],
        default="medium",
    )

    parser.add_argument(
        "--image-size",
        choices=[
            "1024x1024",
            "1024x1536",
            "1536x1024",
            "auto",
        ],
        default="1024x1024",
    )

    parser.add_argument(
        "--image-max-tasks",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--image-continue-on-error",
        action="store_true",
    )

    parser.add_argument(
        "--book-columns",
        type=int,
        choices=[1, 2],
        default=1,
    )

    parser.add_argument(
        "--create-book-with-placeholders",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.parsed_json is not None:
        parsed_path = args.parsed_json.expanduser().resolve()
    else:
        parsed_path = find_latest_parsed_json(
            image_stem=args.image_stem
        )

    image_stem = image_stem_from_parsed_json(parsed_path)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / image_stem
    )

    manifest = run_pipeline(
        parsed_json_path=parsed_path,
        output_dir=output_dir,
        generate_images=args.generate_images,
        image_dry_run=args.image_dry_run,
        allow_manual_review=args.allow_manual_review,
        overwrite=args.overwrite,
        image_quality=args.image_quality,
        image_size=args.image_size,
        image_max_tasks=args.image_max_tasks,
        image_continue_on_error=args.image_continue_on_error,
        book_columns=args.book_columns,
        create_book_with_placeholders=(
            args.create_book_with_placeholders
        ),
    )

    print("=" * 78)
    print("SINGLE-IMAGE PIPELINE FINISHED")
    print("=" * 78)
    print(f"Image stem:       {manifest.image_stem}")
    print(f"Status:           {manifest.status}")
    print(f"Output directory: {manifest.output_dir}")
    print(f"Final instruction:{manifest.final_instruction_path}")
    print(f"Elapsed:          {manifest.elapsed_seconds:.1f}s")
    print("=" * 78)

    for stage in manifest.stages:
        print(
            f"{stage.name:<30} "
            f"{stage.status:<10} "
            f"{stage.output_path or ''}"
        )
        if stage.error_message:
            print(
                f"  {stage.error_type}: "
                f"{stage.error_message}"
            )

    return 0 if manifest.status == "success" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
