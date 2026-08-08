"""Gradio UI for the canonical ``main.run_pipeline`` contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gradio as gr

from main import run_pipeline
from utils.current_state_analyzer import parse_filename


PROJECT_ROOT = Path(__file__).resolve().parent


def _parsed_json_for_upload(image: str, analyzer: Callable[..., dict[str, Any]] | None = None) -> Path:
    image_path = Path(image).resolve()
    if image_path.suffix.lower() == ".json":
        return image_path
    info = parse_filename(image_path)
    if not all(info.get(key) for key in ("model_id", "step_id", "view_angle")):
        raise ValueError("Uploaded filename must contain model, step, and view angle.")
    normal_dir = PROJECT_ROOT / "input" / "normal" / f"{info['model_id']}_{info['step_id']}"
    candidates = sorted(normal_dir.glob(f"*correct*_{info['view_angle']}_*.jpg"))
    if not candidates:
        raise FileNotFoundError("Correct reference image was not found for the uploaded filename.")
    expected = PROJECT_ROOT / "ground_truth" / info["model_id"] / f"{info['step_id']}.json"
    if analyzer is None:
        from utils.current_state_analyzer import analyze_image
        analyzer = analyze_image
    analysis = analyzer(
        image_path=str(image_path), reference_image_path=str(candidates[0]),
        expected_state_path=str(expected), filename_info=info,
    )
    if not analysis.get("success") or not analysis.get("parsed_json_path"):
        raise RuntimeError(str(analysis.get("error") or "Vision analysis failed"))
    parsed = Path(str(analysis["parsed_json_path"]))
    return parsed if parsed.is_absolute() else PROJECT_ROOT / parsed


def run_analysis(
    image: str | None,
    step: str | None = None,
    *,
    pipeline_runner: Callable[..., Any] = run_pipeline,
    analyzer: Callable[..., dict[str, Any]] | None = None,
):
    if image is None:
        return None, None, "Please upload an image.", {}
    try:
        parsed = _parsed_json_for_upload(image, analyzer=analyzer)
        output = PROJECT_ROOT / "output" / "ui_runs" / parsed.stem
        manifest = pipeline_runner(parsed_json_path=parsed, output_dir=output, image_provider="mock")
        message = "Pipeline completed."
        if manifest.warnings:
            message += "\n\nWarnings:\n" + "\n".join(manifest.warnings)
        if manifest.errors:
            message = "\n".join(manifest.errors)
        return (
            manifest.annotated_image_path or image,
            manifest.final_instruction_path,
            message,
            {"status": manifest.status},
        )
    except Exception as exc:
        return image, None, f"{type(exc).__name__}: {exc}", {}


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Assembly Correction SOP") as demo:
        gr.Markdown("# Assembly Correction SOP")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="Assembly image")
                step = gr.Dropdown(choices=[f"step_{value:02d}" for value in range(1, 6)], value="step_01", label="Step")
                analyze_btn = gr.Button("Analyze", variant="primary")
            annotated_output = gr.Image(label="Localized / annotated image", interactive=False)
            final_output = gr.Image(label="Final assembly instruction book", interactive=False)
        suggestion_output = gr.Textbox(label="Pipeline result", lines=8, interactive=False)
        status_output = gr.Label(label="Status")
        analyze_btn.click(
            fn=run_analysis,
            inputs=[image_input, step],
            outputs=[annotated_output, final_output, suggestion_output, status_output],
        )
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
