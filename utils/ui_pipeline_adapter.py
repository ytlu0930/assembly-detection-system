"""Stable UI contract over the full integration pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from utils.integration_pipeline import PROJECT_ROOT, run_full_pipeline


def _normalize(value: str, prefix: str) -> str:
    compact = value.replace("_", "").lower()
    return compact if compact.startswith(prefix) else f"{prefix}{compact}"


def run_analysis_for_ui(
    image_path: str,
    model_id: str,
    step_id: str,
    view_angle: str,
    *,
    reference_image_path: str | None = None,
    expected_state_path: str | None = None,
    pipeline: Callable[..., dict[str, Any]] = run_full_pipeline,
    **pipeline_kwargs: Any,
) -> dict[str, Any]:
    """Return a fixed dictionary suitable for Gradio callbacks and Gallery."""
    model = _normalize(model_id, "model")
    step = _normalize(step_id, "step")
    reference = reference_image_path or str(
        PROJECT_ROOT / "input" / "normal" / f"{model}_{step}" / f"{model}_{step}_correct-01_{view_angle}_01.jpg"
    )
    expected = expected_state_path or str(PROJECT_ROOT / "ground_truth" / model / f"{step}.json")
    raw = pipeline(
        test_image_path=image_path,
        reference_image_path=reference,
        expected_state_path=expected,
        model_id=model,
        step_id=step,
        view_angle=view_angle,
        **pipeline_kwargs,
    )
    sop = raw.get("correction_sop") or {"steps": []}
    steps = sop.get("steps", []) if isinstance(sop, dict) else []
    gallery = [
        (step_item.get("generated_image"), f"步驟 {step_item.get('step_number')}：{step_item.get('instruction', '')}")
        for step_item in steps if step_item.get("generated_image")
    ]
    reports = raw.get("error_reports", [])
    confidence = max((float(item.get("confidence", 0.0)) for item in reports), default=1.0 if raw.get("success") else 0.0)
    correction_text = "\n".join(
        f"{item.get('step_number')}. {item.get('instruction', '')}" for item in steps
    ) or (sop.get("message", "") if isinstance(sop, dict) else "")
    return {
        "success": bool(raw.get("success")),
        "analysis_json": raw.get("analysis_result", {}),
        "annotated_image": raw.get("annotated_image"),
        "sop_steps": steps,
        "sop_gallery": gallery,
        "flowchart": raw.get("flowchart_image"),
        "correction_text": correction_text,
        "confidence": confidence,
        "warnings": list(raw.get("warnings", [])),
        "error_message": raw.get("error_message"),
        "raw_result": raw,
    }
