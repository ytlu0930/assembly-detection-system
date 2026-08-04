"""Build provider-neutral image instructions from structured SOP steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_step_prompts(
    sop: dict[str, Any],
    *,
    test_image_path: str,
    reference_image_path: str,
    model_id: str,
    step_id: str,
    view_angle: str,
    instruction_image_path: str | None = None,
) -> list[dict[str, Any]]:
    """Return one immutable edit task per correction step."""
    if not isinstance(sop, dict) or not isinstance(sop.get("steps", []), list):
        raise ValueError("sop.steps must be a list")
    tasks: list[dict[str, Any]] = []
    previous: str | None = None
    for raw in sop.get("steps", []):
        if not isinstance(raw, dict):
            raise ValueError("Each SOP step must be a dictionary")
        number = int(raw.get("step_number", len(tasks) + 1))
        output_name = f"step_{number:02d}.png"
        instruction = str(raw.get("visual_instruction") or raw.get("instruction") or "")
        affected = [str(value) for value in raw.get("affected_parts", [])]
        prompt = (
            "Create one clear construction-toy repair instruction image. "
            f"Perform only this step: {instruction} "
            f"Target parts: {', '.join(affected) or 'identified local area'}. "
            "Keep the vehicle body, background, camera viewpoint, lighting, colors, "
            "part shapes, part counts, and every non-target part unchanged. "
            "Show the removal, movement, swap, or installation direction with one clear red arrow. "
            "Use a clean numbered assembly-manual style. Do not add nonexistent parts. "
            "Do not show hands or people. Always compare against the correct reference image."
        )
        tasks.append(
            {
                "step_number": number,
                "action": raw.get("action"),
                "instruction": raw.get("instruction"),
                "visual_instruction": instruction,
                "prompt": prompt,
                "test_image_path": str(Path(test_image_path)),
                "reference_image_path": str(Path(reference_image_path)),
                "previous_output": previous,
                "instruction_image_path": instruction_image_path,
                "bbox": raw.get("bbox"),
                "affected_parts": affected,
                "model_id": model_id,
                "step_id": step_id,
                "view_angle": view_angle,
                "output_filename": output_name,
            }
        )
        previous = output_name
    return tasks
