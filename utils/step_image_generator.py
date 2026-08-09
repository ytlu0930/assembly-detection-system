"""Provider-neutral SOP step image generation with a deterministic mock."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image, ImageDraw

from utils.step_image_provider_contract import StepImageProvider, StepImageResult


class MockStepImageProvider:
    """Create an offline assembly-card placeholder; never calls an API."""

    name = "mock"

    def generate(self, task: dict[str, Any], output_path: Path) -> None:
        canvas = Image.new("RGB", (1280, 720), "white")
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((40, 40, 1240, 680), radius=30, outline="#1f4e79", width=6)
        draw.text((80, 75), f"STEP {task['step_number']}", fill="#1f4e79", stroke_width=1)
        draw.text((80, 150), str(task.get("action", "repair")).upper(), fill="#c00000")
        text = str(task.get("instruction") or task.get("visual_instruction") or "")
        # Pillow's default font may not contain CJK; an ASCII summary remains stable.
        safe_text = text if text.isascii() else f"Repair target: {', '.join(task.get('affected_parts', []))}"
        draw.multiline_text((80, 240), safe_text, fill="black", spacing=12)
        draw.line((850, 350, 1110, 350), fill="#d40000", width=18)
        draw.polygon([(1110, 350), (1035, 305), (1035, 395)], fill="#d40000")
        draw.text((80, 600), "OFFLINE MOCK - replace with approved image provider", fill="#666666")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)

    def generate_step_image(
        self,
        source_image_path: str,
        reference_image_path: str,
        prompt: str,
        output_path: str,
        metadata: dict[str, Any] | None = None,
        *,
        execute_api: bool = False,
        mask_path: str | Path | None = None,
    ) -> StepImageResult:
        started = perf_counter()
        target = Path(output_path)
        task = dict(metadata or {})
        task["prompt"] = prompt
        task["test_image_path"] = source_image_path
        task["reference_image_path"] = reference_image_path
        try:
            self.generate(task, target)
            duration = perf_counter() - started
            return StepImageResult(True, str(target.resolve()), self.name, duration, status="success", duration_seconds=duration)
        except Exception as exc:
            duration = perf_counter() - started
            return StepImageResult(False, None, self.name, duration, status="failed", duration_seconds=duration, error=f"{type(exc).__name__}: {exc}")


def generate_step_images(
    tasks: list[dict[str, Any]],
    output_dir: str | Path,
    provider: StepImageProvider | None = None,
    *,
    execute_api: bool = False,
    max_steps_per_run: int | None = None,
    max_requests_per_run: int | None = None,
    sequential: bool = True,
) -> list[dict[str, Any]]:
    """Generate step images sequentially and return a non-throwing manifest."""
    if not isinstance(tasks, list):
        raise TypeError("tasks must be a list")
    active_provider = provider or MockStepImageProvider()
    external = active_provider.name == "openai"
    step_budget = max_steps_per_run if max_steps_per_run is not None else (1 if external else len(tasks))
    request_budget = max_requests_per_run if max_requests_per_run is not None else (1 if external else len(tasks))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    successful_output: str | None = None
    provider_calls = 0
    stop_reason: str | None = None
    for index, task in enumerate(tasks):
        started = perf_counter()
        path = target / str(task.get("output_filename", f"step_{len(manifest)+1:02d}.png"))
        source_path = successful_output if sequential and successful_output else str(task.get("test_image_path") or "")
        reference_path = str(task.get("reference_image_path") or "")
        record = {
            "step_number": task.get("step_number"),
            "provider": active_provider.name,
            "prompt": task.get("prompt"),
            "source_image_path": source_path,
            "reference_image_path": reference_path,
            "input_paths": {
                "test": source_path,
                "reference": reference_path,
                "previous": task.get("previous_output"),
            },
            "output_path": str(path.resolve()),
            "status": "pending",
            "duration_sec": 0.0,
            "warning": None,
            "error": None,
            "metadata": {},
        }
        if stop_reason or index >= max(0, int(step_budget)) or (external and provider_calls >= max(0, int(request_budget))):
            record["status"] = "skipped"
            record["warning"] = stop_reason or "Step image generation budget exhausted."
            record["output_path"] = None
            manifest.append(record)
            continue
        try:
            if hasattr(active_provider, "generate_step_image"):
                provider_result = active_provider.generate_step_image(
                    source_image_path=source_path,
                    reference_image_path=reference_path,
                    prompt=str(task.get("prompt") or ""),
                    output_path=str(path),
                    metadata=dict(task),
                    execute_api=execute_api,
                )
                provider_calls += 1
                if not provider_result.success:
                    record["provider"] = provider_result.provider
                    record["status"] = provider_result.status
                    record["warning"] = provider_result.warning
                    record["error"] = provider_result.error
                    record["metadata"] = dict(provider_result.metadata)
                    record["output_path"] = None
                    manifest.append(record)
                    stop_reason = provider_result.warning or provider_result.error or "Provider stopped image generation."
                    continue
                record["provider"] = provider_result.provider
                record["warning"] = provider_result.warning
                record["metadata"] = dict(provider_result.metadata)
                record["duration_sec"] = round(provider_result.duration_seconds or provider_result.duration, 6)
            else:
                # Backward-compatible bridge for existing local prototypes.
                active_provider.generate(task, path)
            if not path.is_file():
                raise FileNotFoundError("Provider did not create its declared output")
            record["status"] = "success"
            successful_output = str(path.resolve())
        except Exception as exc:  # provider failure is a documented fallback
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["output_path"] = None
            stop_reason = record["error"]
        finally:
            record["duration_sec"] = round(perf_counter() - started, 6)
        if record not in manifest:
            manifest.append(record)
    with (target / "generation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
