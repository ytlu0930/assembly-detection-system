"""Canonical provider-backed V2 step-image generator.

This module never constructs an OpenAI client itself. All model-backed edits go
through the audited StepImageProvider contract; Python-only comparison tasks do
not use a provider.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from dotenv import load_dotenv
from PIL import Image, ImageDraw

from utils.step_image_provider_contract import StepImageProvider
from utils.step_image_provider_factory import create_step_image_provider


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_SIZE = "1536x1024"
DEFAULT_QUALITY = "low"
DEFAULT_OUTPUT_FORMAT = "png"


@dataclass
class TaskRecord:
    sequence_index: int
    sop_step_no: int
    action: str
    title: str
    api_mode: str
    branch: str
    status: str
    model: str
    size: str
    quality: str
    output_format: str
    base_image_path: str | None = None
    reference_image_path: str | None = None
    annotated_image_path: str | None = None
    mask_path: str | None = None
    previous_assembly_output: str | None = None
    output_path: str | None = None
    actual_api_operation: str | None = None
    attempts: int = 0
    elapsed_seconds: float | None = None
    prompt_characters: int = 0
    error_type: str | None = None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class GenerationManifestV2:
    schema_version: str
    created_at: str
    source_prompts_json: str
    model_id: str
    assembly_step_id: str
    source_image_name: str
    image_model: str
    size: str
    quality: str
    output_format: str
    input_fidelity: str
    dry_run: bool
    use_mask: bool
    include_annotated_image: bool
    package_generation_allowed: bool
    package_requires_manual_review: bool
    identity_verification_blocked: bool
    manual_review_override: bool
    requested_task_count: int
    successful_task_count: int
    existing_task_count: int
    failed_task_count: int
    skipped_task_count: int
    output_directory: str
    standalone_outputs: list[str]
    assembly_outputs: list[str]
    comparison_outputs: list[str]
    final_assembly_image_path: str | None
    final_comparison_image_path: str | None
    tasks: list[TaskRecord]
    provider: str = "mock"
    execute_api: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("step_prompts_v2.json must contain an object")
    return payload


def _input_path(task: dict[str, Any], role: str) -> str | None:
    for item in task.get("image_inputs", []):
        if isinstance(item, dict) and item.get("role") == role and item.get("path"):
            return str(item["path"])
    return None


def _comparison(left_path: str | None, right_path: str | None, output: Path) -> None:
    panels: list[Image.Image] = []
    for path, color in ((left_path, "#eef3f6"), (right_path, "#f6f1e8")):
        try:
            image = Image.open(str(path)).convert("RGB") if path else None
        except OSError:
            image = None
        if image is None:
            image = Image.new("RGB", (640, 480), color)
            ImageDraw.Draw(image).text((40, 40), "Image unavailable - textual fallback", fill="black")
        image.thumbnail((640, 640))
        panels.append(image)
    canvas = Image.new("RGB", (1320, max(panel.height for panel in panels) + 80), "white")
    x = 20
    for panel in panels:
        canvas.paste(panel, (x, 60))
        x += 660
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), "Corrected state", fill="black")
    draw.text((680, 20), "Correct reference", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


class StepImageGeneratorV2:
    """Execute V2 tasks through MockStepImageProvider or OpenAIImageProvider."""

    def __init__(
        self,
        *,
        provider: StepImageProvider | None = None,
        provider_name: str = "mock",
        execute_api: bool = False,
        confirm_cost: bool = False,
        model: str = DEFAULT_MODEL,
        size: str = DEFAULT_SIZE,
        quality: str = DEFAULT_QUALITY,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        max_requests_per_run: int = 1,
        **_: Any,
    ) -> None:
        self.execute_api = bool(execute_api and confirm_cost)
        self.provider = provider or create_step_image_provider(
            provider_name,
            enable_external_api=self.execute_api,
            **({
                "model": model,
                "size": size,
                "quality": quality,
                "output_format": output_format,
                "max_requests_per_run": max_requests_per_run,
            } if provider_name in {"openai", "azure_openai"} else {}),
        )
        self.model = model
        self.size = size
        self.quality = quality
        self.output_format = output_format
        self.max_requests_per_run = max(0, int(max_requests_per_run))

    def run(
        self,
        *,
        prompts_json_path: str | Path,
        output_dir: str | Path,
        allow_manual_review: bool = False,
        dry_run: bool = False,
        max_tasks: int | None = None,
        branch_filter: Literal["all", "standalone", "assembly", "composition"] = "all",
        continue_on_error: bool = False,
        overwrite: bool = False,
    ) -> GenerationManifestV2:
        started = perf_counter()
        prompts_path = Path(prompts_json_path).expanduser().resolve()
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        package = _load_json(prompts_path)
        raw_tasks = [item for item in package.get("step_prompts", []) if isinstance(item, dict)]
        if branch_filter != "all":
            raw_tasks = [item for item in raw_tasks if item.get("image_task", {}).get("branch") == branch_filter]
        if max_tasks is not None:
            raw_tasks = raw_tasks[: max(0, int(max_tasks))]

        identity_verification_blocked = bool(package.get("identity_verification_blocked", False))
        allowed = not identity_verification_blocked and (
            bool(package.get("generation_allowed", True)) or allow_manual_review
        )
        records: list[TaskRecord] = []
        standalone: list[str] = []
        assembly: list[str] = []
        comparisons: list[str] = []
        warnings: list[str] = []
        last_assembly: str | None = None
        stopped = False

        for raw in raw_tasks:
            task_started = perf_counter()
            image_task = raw.get("image_task", {})
            mode = str(image_task.get("api_mode", "edit"))
            branch = str(image_task.get("branch", "assembly"))
            filename = str(raw.get("output_filename") or f"sequence_{len(records)+1:02d}.png")
            branch_dir = "comparison" if branch == "composition" else branch
            target = output / branch_dir / filename
            source = last_assembly if image_task.get("use_previous_output") and last_assembly else _input_path(raw, "base_image")
            reference = _input_path(raw, "reference_image")
            if not source:
                source = reference
            record = TaskRecord(
                sequence_index=int(raw.get("sequence_index", len(records) + 1)),
                sop_step_no=int(raw.get("sop_step_no", raw.get("step_number", 0))),
                action=str(raw.get("action", "")), title=str(raw.get("title", "")),
                api_mode=mode, branch=branch, status="pending", model=self.model,
                size=self.size, quality=self.quality, output_format=self.output_format,
                base_image_path=source, reference_image_path=reference,
                annotated_image_path=_input_path(raw, "annotated_image"),
                previous_assembly_output=last_assembly,
                prompt_characters=len(str(raw.get("prompt_en", ""))),
            )
            try:
                if target.exists() and not overwrite:
                    record.status, record.output_path = "existing", str(target)
                elif dry_run:
                    record.status = "dry_run"
                elif not allowed:
                    record.status, record.error_message = "skipped", "Manual review is required."
                elif stopped:
                    record.status, record.error_message = "skipped", "A previous image task failed or was disabled."
                elif mode == "compose_python":
                    _comparison(last_assembly, reference, target)
                    record.status, record.output_path = "success", str(target)
                    record.actual_api_operation = "python_composition"
                else:
                    result = self.provider.generate_step_image(
                        source_image_path=str(source or ""),
                        reference_image_path=str(reference or source or ""),
                        prompt=str(raw.get("prompt_en", "")),
                        output_path=str(target),
                        metadata={**raw, "step_number": raw.get("sop_step_no")},
                        execute_api=self.execute_api,
                        mask_path=raw.get("mask_path"),
                    )
                    record.status = result.status
                    record.output_path = result.output_path
                    record.actual_api_operation = (
                        "azure_http_images_edit" if self.provider.name == "azure_openai"
                        else "images.edit" if self.provider.name == "openai"
                        else "offline_mock"
                    )
                    record.attempts = result.request_count
                    record.error_type = result.last_error_type
                    record.error_message = result.error
                    if result.warning:
                        record.warnings.append(result.warning)
                    if not result.success:
                        stopped = True
                        warnings.append(f"Step {record.sequence_index}: {result.warning or result.error}")
                if record.output_path and branch == "assembly":
                    last_assembly = record.output_path
                if record.output_path:
                    (standalone if branch == "standalone" else comparisons if branch == "composition" else assembly).append(record.output_path)
            except Exception as exc:
                record.status = "failed"
                record.error_type = type(exc).__name__
                record.error_message = str(exc)
                warnings.append(f"Step {record.sequence_index}: {type(exc).__name__}: {exc}")
                stopped = not continue_on_error
            record.elapsed_seconds = round(perf_counter() - task_started, 6)
            records.append(record)

        success_count = sum(record.status == "success" for record in records)
        existing_count = sum(record.status == "existing" for record in records)
        failed_count = sum(record.status in {"failed", "api_error", "timeout", "rate_limited", "invalid_response", "output_validation_failed", "not_configured", "invalid_configuration", "invalid_request", "authentication_error", "permission_error", "deployment_or_endpoint_not_found", "service_error", "connection_error"} for record in records)
        skipped_count = len(records) - success_count - existing_count - failed_count
        manifest = GenerationManifestV2(
            schema_version="2.2", created_at=datetime.now().astimezone().isoformat(),
            source_prompts_json=str(prompts_path), model_id=str(package.get("model_id", "")),
            assembly_step_id=str(package.get("step_id", "")), source_image_name=str(package.get("image_name", "")),
            image_model=self.model, size=self.size, quality=self.quality, output_format=self.output_format,
            input_fidelity="provider_managed", dry_run=dry_run, use_mask=False, include_annotated_image=False,
            package_generation_allowed=bool(package.get("generation_allowed", True)),
            package_requires_manual_review=bool(package.get("requires_manual_review", False)),
            identity_verification_blocked=identity_verification_blocked,
            manual_review_override=allow_manual_review, requested_task_count=len(records),
            successful_task_count=success_count, existing_task_count=existing_count,
            failed_task_count=failed_count, skipped_task_count=skipped_count,
            output_directory=str(output), standalone_outputs=standalone, assembly_outputs=assembly,
            comparison_outputs=comparisons, final_assembly_image_path=last_assembly,
            final_comparison_image_path=comparisons[-1] if comparisons else None,
            tasks=records, provider=self.provider.name, execute_api=self.execute_api, warnings=warnings,
        )
        (output / "generation_manifest_v2.json").write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "generation_manifest_v2.md").write_text(
            f"# Step Image Generation V2\n\n- Provider: `{manifest.provider}`\n- API execution: `{manifest.execute_api}`\n- Tasks: `{len(records)}`\n- Elapsed: `{perf_counter() - started:.3f}s`\n",
            encoding="utf-8",
        )
        return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run provider-backed V2 step image generation.")
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-provider", choices=["mock", "openai", "azure_openai"], default="mock")
    parser.add_argument("--execute-image-api", action="store_true")
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-manual-review", action="store_true")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--branch", choices=["all", "standalone", "assembly", "composition"], default="all")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--quality", default=DEFAULT_QUALITY)
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute_image_api:
        load_dotenv(PROJECT_ROOT / ".env")
    output = args.output_dir or args.prompts_json.resolve().parent / "generated_steps_v2"
    generator = StepImageGeneratorV2(
        provider_name=args.image_provider, execute_api=args.execute_image_api,
        confirm_cost=args.confirm_cost, model=args.model, size=args.size,
        quality=args.quality, output_format=args.output_format,
        max_requests_per_run=args.max_requests,
    )
    manifest = generator.run(
        prompts_json_path=args.prompts_json, output_dir=output,
        allow_manual_review=args.allow_manual_review, dry_run=args.dry_run,
        max_tasks=args.max_tasks, branch_filter=args.branch,
        continue_on_error=args.continue_on_error, overwrite=args.overwrite,
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
    return 0 if manifest.failed_task_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
