"""Canonical single-case assembly-correction pipeline.

Parsed Vision JSON -> multi-ErrorReport localization -> correction SOP -> V2
prompts -> provider-backed V2 images -> instruction book -> manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from correction_sop_generator import CorrectionSOPGenerator
from instruction_book_generator import InstructionBookGenerator
from pipeline_smoke_test import find_latest_parsed_json, image_stem_from_parsed_json, process_one
from step_image_generator_v2 import StepImageGeneratorV2
from step_prompt_builder_v2 import StepPromptBuilderV2
from utils.step_image_provider_contract import StepImageProvider
from utils.step_image_provider_factory import create_step_image_provider


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "single_runs"


@dataclass
class StageRecord:
    name: str
    status: str
    output_path: str | None = None
    elapsed_seconds: float = 0.0
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class PipelineManifest:
    schema_version: str
    created_at: str
    finished_at: str | None
    parsed_json_path: str
    image_stem: str
    output_dir: str
    status: str
    results_path: str | None = None
    correction_sop_path: str | None = None
    step_prompts_path: str | None = None
    generated_steps_dir: str | None = None
    generated_step_image_paths: list[str] = field(default_factory=list)
    annotated_image_path: str | None = None
    final_instruction_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manual_review_required: bool = False
    image_provider: str = "mock"
    execute_image_api: bool = False
    elapsed_seconds: float = 0.0
    stages: list[StageRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _stage(name: str, status: str, path: Path | None, started: float, exc: Exception | None = None) -> StageRecord:
    return StageRecord(
        name=name, status=status, output_path=str(path) if path else None,
        elapsed_seconds=round(perf_counter() - started, 6),
        error_type=type(exc).__name__ if exc else None,
        error_message=str(exc) if exc else None,
    )


def run_pipeline(
    *,
    parsed_json_path: str | Path,
    output_dir: str | Path | None = None,
    generate_images: bool = False,
    image_provider: str = "mock",
    execute_image_api: bool = False,
    confirm_cost: bool = False,
    provider: StepImageProvider | None = None,
    localizer: Any | None = None,
    allow_manual_review: bool = False,
    overwrite: bool = False,
    image_quality: str = "low",
    image_size: str = "1536x1024",
    image_max_tasks: int | None = None,
    image_max_requests: int = 1,
    image_continue_on_error: bool = False,
    book_columns: int = 1,
    create_book_with_placeholders: bool = True,
    image_dry_run: bool = False,
) -> PipelineManifest:
    """Run the only formal full pipeline. Mock is always the safe default."""
    parsed_path = Path(parsed_json_path).expanduser().resolve()
    image_stem = image_stem_from_parsed_json(parsed_path)
    case_dir = Path(output_dir).expanduser().resolve() if output_dir else (DEFAULT_OUTPUT_ROOT / image_stem).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = case_dir / "pipeline_manifest.json"
    external_authorized = bool(
        image_provider == "openai" and generate_images and execute_image_api and confirm_cost
    )
    active_provider = provider or create_step_image_provider(
        image_provider,
        enable_external_api=external_authorized,
        **({
            "quality": image_quality,
            "size": image_size,
            "max_requests_per_run": image_max_requests,
        } if image_provider == "openai" else {}),
    )
    manifest = PipelineManifest(
        schema_version="2.0", created_at=datetime.now().astimezone().isoformat(),
        finished_at=None, parsed_json_path=str(parsed_path), image_stem=image_stem,
        output_dir=str(case_dir), status="running", image_provider=active_provider.name,
        execute_image_api=external_authorized,
    )
    pipeline_started = perf_counter()

    try:
        started = perf_counter()
        results_path = process_one(parsed_path, case_dir, overwrite=overwrite, localizer=localizer)
        manifest.results_path = str(results_path)
        manifest.stages.append(_stage("localization_and_error_reports", "success", results_path, started))
        results = json.loads(results_path.read_text(encoding="utf-8"))
        localizations = results.get("localizations", [])
        annotated = next((item.get("annotated_image_path") for item in localizations if isinstance(item, dict) and item.get("annotated_image_path")), None)
        manifest.annotated_image_path = str(annotated) if annotated else None
        for item in localizations:
            if isinstance(item, dict) and item.get("error_message"):
                manifest.warnings.append(str(item["error_message"]))

        started = perf_counter()
        sop_generator = CorrectionSOPGenerator()
        sop = sop_generator.generate_from_results(results_path)
        sop_path, _ = sop_generator.save(sop, case_dir)
        manifest.correction_sop_path = str(sop_path)
        manifest.manual_review_required = sop.requires_manual_review
        manifest.warnings.extend(sop.warnings)
        manifest.stages.append(_stage("correction_sop_generator", "success", sop_path, started))

        started = perf_counter()
        prompt_builder = StepPromptBuilderV2(block_on_manual_review=not allow_manual_review)
        package = prompt_builder.build_from_sop(sop_path)
        prompts_path, _ = prompt_builder.save(package, case_dir)
        manifest.step_prompts_path = str(prompts_path)
        manifest.warnings.extend(package.warnings)
        manifest.stages.append(_stage("step_prompt_builder_v2", "success", prompts_path, started))

        started = perf_counter()
        generated_dir = case_dir / "generated_steps_v2"
        image_generator = StepImageGeneratorV2(
            provider=active_provider, execute_api=external_authorized,
            confirm_cost=confirm_cost, model="gpt-image-2", quality=image_quality,
            size=image_size, max_requests_per_run=image_max_requests,
        )
        image_manifest = image_generator.run(
            prompts_json_path=prompts_path, output_dir=generated_dir,
            allow_manual_review=allow_manual_review or active_provider.name == "mock",
            dry_run=image_dry_run, max_tasks=image_max_tasks,
            continue_on_error=image_continue_on_error, overwrite=overwrite,
        )
        manifest.generated_steps_dir = str(generated_dir)
        manifest.generated_step_image_paths = image_manifest.standalone_outputs + image_manifest.assembly_outputs + image_manifest.comparison_outputs
        manifest.warnings.extend(image_manifest.warnings)
        image_status = "success" if image_manifest.failed_task_count == 0 else "partial"
        manifest.stages.append(_stage("step_image_generator_v2", image_status, generated_dir / "generation_manifest_v2.json", started))

        started = perf_counter()
        book_path = case_dir / "assembly_instruction_book.png"
        book_generator = InstructionBookGenerator(columns=book_columns)
        if book_path.exists() and overwrite:
            book_path.unlink()
        if not book_path.exists():
            book_generator.generate(prompts_path, book_path, overwrite=False)
        manifest.final_instruction_path = str(book_path)
        manifest.stages.append(_stage("instruction_book_generator", "success", book_path, started))
        manifest.status = "partial" if manifest.warnings else "success"
    except Exception as exc:
        manifest.status = "failed"
        manifest.errors.append(f"{type(exc).__name__}: {exc}")
        manifest.stages.append(_stage("pipeline", "failed", None, pipeline_started, exc))
    finally:
        manifest.finished_at = datetime.now().astimezone().isoformat()
        manifest.elapsed_seconds = round(perf_counter() - pipeline_started, 6)
        _save(manifest_path, manifest.to_dict())
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical single-case correction pipeline.")
    parser.add_argument("--parsed-json", type=Path)
    parser.add_argument("--image-stem")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--generate-images", action="store_true", help="Generate images with the selected provider; provider defaults to mock.")
    parser.add_argument("--image-provider", choices=["mock", "openai"], default="mock")
    parser.add_argument("--execute-image-api", action="store_true")
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--image-dry-run", action="store_true")
    parser.add_argument("--allow-manual-review", action="store_true")
    parser.add_argument("--image-quality", default="low")
    parser.add_argument("--image-size", default="1536x1024")
    parser.add_argument("--image-max-tasks", type=int)
    parser.add_argument("--image-max-requests", type=int, default=1)
    parser.add_argument("--image-continue-on-error", action="store_true")
    parser.add_argument("--book-columns", type=int, choices=[1, 2], default=1)
    parser.add_argument("--create-book-with-placeholders", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parsed = args.parsed_json.resolve() if args.parsed_json else find_latest_parsed_json(args.image_stem)
    if args.execute_image_api:
        load_dotenv(PROJECT_ROOT / ".env")
    manifest = run_pipeline(
        parsed_json_path=parsed, output_dir=args.output_dir, generate_images=args.generate_images,
        image_provider=args.image_provider, execute_image_api=args.execute_image_api,
        confirm_cost=args.confirm_cost, allow_manual_review=args.allow_manual_review,
        overwrite=args.overwrite, image_quality=args.image_quality, image_size=args.image_size,
        image_max_tasks=args.image_max_tasks, image_max_requests=args.image_max_requests,
        image_continue_on_error=args.image_continue_on_error, book_columns=args.book_columns,
        create_book_with_placeholders=args.create_book_with_placeholders,
        image_dry_run=args.image_dry_run,
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
    return 0 if manifest.status in {"success", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
