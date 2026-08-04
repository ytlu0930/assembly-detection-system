"""Unified Vision -> localization -> SOP -> visual-output pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from utils.correction_sop_generator import generate_correction_sop
from utils.error_report_adapter import adapt_vision_result
from utils.image_annotator import annotate_image
from utils.step_image_generator import MockStepImageProvider, generate_step_images as render_steps
from utils.step_prompt_builder import build_step_prompts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def _localize(localizer: Any, **kwargs: Any) -> dict[str, Any]:
    if hasattr(localizer, "localize"):
        return localizer.localize(**kwargs)
    if callable(localizer):
        return localizer(**kwargs)
    raise TypeError("localizer must be callable or expose localize()")


def run_full_pipeline(
    test_image_path: str,
    reference_image_path: str,
    expected_state_path: str,
    model_id: str,
    step_id: str,
    view_angle: str,
    generate_step_images: bool = True,
    generate_flowchart: bool = True,
    *,
    analysis_result: dict[str, Any] | None = None,
    analyzer: Callable[..., dict[str, Any]] | None = None,
    localizer: Any | None = None,
    image_provider: Any | None = None,
    flowchart_builder: Callable[..., str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete system while isolating optional-stage failures.

    Dependency injection keeps offline tests free of network, API, GPU and model
    downloads.  If ``analysis_result`` is supplied, Vision is replayed from that
    payload instead of making an API request.
    """
    started = perf_counter()
    timing: dict[str, float] = {}
    warnings: list[str] = []
    result: dict[str, Any] = {
        "success": False,
        "analysis_result": {},
        "error_reports": [],
        "annotated_image": None,
        "correction_sop": None,
        "flowchart_image": None,
        "timing": timing,
        "warnings": warnings,
        "error_message": None,
    }
    try:
        test_path = Path(test_image_path).resolve()
        reference_path = Path(reference_image_path).resolve()
        expected_path = Path(expected_state_path).resolve()
        for label, path in (("test image", test_path), ("reference image", reference_path), ("expected state", expected_path)):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        target = Path(output_dir or PROJECT_ROOT / "output" / "integration" / f"{model_id}_{step_id}_{view_angle}")
        target.mkdir(parents=True, exist_ok=True)
        expected_state = _load_json(expected_path)

        stage = perf_counter()
        if analysis_result is None:
            if analyzer is None:
                from utils.current_state_analyzer import analyze_image, parse_filename
                analyzer = analyze_image
                info = parse_filename(test_path)
            else:
                info = {
                    "image_name": test_path.name,
                    "relative_path": str(test_path),
                    "model_id": model_id,
                    "step_id": step_id,
                    "view_angle": view_angle,
                }
            analysis_result = analyzer(
                image_path=str(test_path),
                reference_image_path=str(reference_path),
                expected_state_path=str(expected_path),
                filename_info=info,
            )
        timing["vision_sec"] = round(perf_counter() - stage, 6)
        if not isinstance(analysis_result, dict):
            raise TypeError("Analyzer result must be a dictionary")
        if analysis_result.get("success") is False:
            raise RuntimeError(str(analysis_result.get("error") or "Vision analysis failed"))
        result["analysis_result"] = analysis_result

        reports = adapt_vision_result(analysis_result)
        result["error_reports"] = reports

        stage = perf_counter()
        if reports and localizer is None:
            try:
                from utils.localization_pipeline import LocalizationPipeline
                localizer = LocalizationPipeline(device="auto")
            except Exception as exc:
                warnings.append(f"Localization unavailable: {type(exc).__name__}: {exc}")
        for report in reports:
            if localizer is None:
                break
            source = reference_path if report["error_type"] == "missingpart" else test_path
            try:
                localized = _localize(
                    localizer,
                    image_path=str(source),
                    text_prompt=str(report["part_id"]).replace("_", " ").lower(),
                    target_position="center",
                    output_dir=str(target / "localized"),
                )
                report["localization"] = localized
                bbox = localized.get("selected_bbox") if isinstance(localized, dict) else None
                if isinstance(bbox, list) and len(bbox) == 4:
                    report["bbox"] = [float(value) for value in bbox]
                elif isinstance(localized, dict) and localized.get("error_message"):
                    warnings.append(f"Localization failed for {report['part_id']}: {localized['error_message']}")
            except Exception as exc:
                warnings.append(f"Localization failed for {report['part_id']}: {type(exc).__name__}: {exc}")
        timing["localization_sec"] = round(perf_counter() - stage, 6)

        annotations = [
            {"part_id": report["part_id"], "error_type": report["error_type"], "status": "error", "bbox": report["bbox"]}
            for report in reports if report.get("bbox") is not None
        ]
        if annotations:
            try:
                result["annotated_image"] = annotate_image(str(test_path), annotations, str(target / "annotated"))
            except Exception as exc:
                warnings.append(f"Annotation failed: {type(exc).__name__}: {exc}")
        else:
            result["annotated_image"] = str(test_path)

        stage = perf_counter()
        sop = generate_correction_sop(
            reports,
            expected_state,
            step_id,
            {"path": str(reference_path), "model_id": model_id, "view_angle": view_angle},
        )
        result["correction_sop"] = sop
        timing["sop_sec"] = round(perf_counter() - stage, 6)

        if generate_step_images and sop.get("steps"):
            tasks = build_step_prompts(
                sop,
                test_image_path=str(test_path),
                reference_image_path=str(reference_path),
                model_id=model_id,
                step_id=step_id,
                view_angle=view_angle,
            )
            manifest = render_steps(tasks, target / "step_images", image_provider or MockStepImageProvider())
            for step, record in zip(sop["steps"], manifest):
                step["generated_image"] = record.get("output_path")
                step["image_generation_status"] = record.get("status")
                if record.get("error"):
                    warnings.append(f"Step image {step['step_number']} failed: {record['error']}")

        if generate_flowchart:
            try:
                if flowchart_builder is None:
                    from flowchart_generator import generate_sop_flowchart
                    flowchart_builder = generate_sop_flowchart
                result["flowchart_image"] = flowchart_builder(sop, output_dir=str(target / "flowchart"))
            except Exception as exc:
                warnings.append(f"Flowchart failed: {type(exc).__name__}: {exc}")

        result["success"] = True
        timing["total_sec"] = round(perf_counter() - started, 6)
        with (target / "full_pipeline_result.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        result["error_message"] = f"{type(exc).__name__}: {exc}"
    finally:
        timing["total_sec"] = round(perf_counter() - started, 6)
    return result
