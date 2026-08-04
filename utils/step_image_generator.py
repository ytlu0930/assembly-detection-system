"""Provider-neutral SOP step image generation with a deterministic mock."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from PIL import Image, ImageDraw


class StepImageProvider(Protocol):
    """Small interface implemented by real or mock image providers."""

    name: str

    def generate(self, task: dict[str, Any], output_path: Path) -> None:
        """Write exactly one image to ``output_path``."""


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


def generate_step_images(
    tasks: list[dict[str, Any]],
    output_dir: str | Path,
    provider: StepImageProvider | None = None,
) -> list[dict[str, Any]]:
    """Generate all tasks independently and return a non-throwing manifest."""
    if not isinstance(tasks, list):
        raise TypeError("tasks must be a list")
    active_provider = provider or MockStepImageProvider()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for task in tasks:
        started = perf_counter()
        path = target / str(task.get("output_filename", f"step_{len(manifest)+1:02d}.png"))
        record = {
            "step_number": task.get("step_number"),
            "provider": active_provider.name,
            "prompt": task.get("prompt"),
            "input_paths": {
                "test": task.get("test_image_path"),
                "reference": task.get("reference_image_path"),
                "previous": task.get("previous_output"),
            },
            "output_path": str(path.resolve()),
            "status": "pending",
            "duration_sec": 0.0,
            "error": None,
        }
        try:
            active_provider.generate(task, path)
            if not path.is_file():
                raise FileNotFoundError("Provider did not create its declared output")
            record["status"] = "success"
        except Exception as exc:  # provider failure is a documented fallback
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["output_path"] = None
        finally:
            record["duration_sec"] = round(perf_counter() - started, 6)
        manifest.append(record)
    with (target / "generation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
