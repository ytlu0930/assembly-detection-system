"""
pipeline_smoke_test.py

批次友善版 Vision JSON -> Error-aware Localization 執行入口。

重點：
- 不重新呼叫 Azure OpenAI Vision。
- 直接讀取 logs/current_parsed_json 中既有的 *_parsed_*.json。
- process_one(parsed_json_path, output_dir) 可由 batch_pipeline.py 呼叫。
- 每張圖片使用自己的 output_dir，因此不互相覆蓋。
- 單獨執行時可指定 --parsed-json；未指定時可用 --image-stem 找最新一份。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from utils.localization_pipeline import LocalizationPipeline
from utils.error_report_adapter import adapt_vision_result


ROOT_DIR = Path(__file__).resolve().parent
PARSED_JSON_DIR = ROOT_DIR / "logs" / "current_parsed_json"
OUTPUT_ROOT = ROOT_DIR / "output"
DEFAULT_SINGLE_RUN_ROOT = OUTPUT_ROOT / "single_runs"

PARSED_TS_RE = re.compile(r"_parsed_(\d{8})_(\d{6})_(\d+)$")


@dataclass
class RunPaths:
    run_dir: Path
    results_json: Path
    run_summary_json: Path
    annotated_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object, got {type(data).__name__}: {path}")
    return data


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)


def image_stem_from_parsed_json(path: str | Path) -> str:
    name = Path(path).stem
    return name.split("_parsed_", 1)[0] if "_parsed_" in name else name


def parsed_timestamp_key(path: Path) -> tuple[int, int, int, float]:
    match = PARSED_TS_RE.search(path.stem)
    if match:
        return (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            path.stat().st_mtime,
        )
    return (0, 0, 0, path.stat().st_mtime)


def find_latest_parsed_json(
    image_stem: Optional[str] = None,
    *,
    parsed_json_dir: Path = PARSED_JSON_DIR,
) -> Path:
    parsed_json_dir = parsed_json_dir.expanduser().resolve()
    if not parsed_json_dir.is_dir():
        raise FileNotFoundError(f"Parsed JSON directory not found: {parsed_json_dir}")

    pattern = f"{image_stem}_parsed_*.json" if image_stem else "*_parsed_*.json"
    files = [path for path in parsed_json_dir.glob(pattern) if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No parsed JSON matched: {parsed_json_dir / pattern}")
    return max(files, key=parsed_timestamp_key)


def resolve_project_path(path_value: str | Path) -> Path:
    raw_path = Path(str(path_value).replace("\\", "/"))
    path = raw_path if raw_path.is_absolute() else ROOT_DIR / raw_path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Project file not found: {path}")
    return path


def resolve_test_image(parsed_result: dict[str, Any]) -> Path:
    relative_path = (
        parsed_result.get("test_image", {}).get("relative_path")
        or parsed_result.get("file_info", {}).get("relative_path")
    )
    if not relative_path:
        raise KeyError(
            "Parsed JSON does not contain test_image.relative_path "
            "or file_info.relative_path."
        )
    return resolve_project_path(relative_path)


def resolve_reference_image(parsed_result: dict[str, Any]) -> Path:
    relative_path = parsed_result.get("reference_image", {}).get("relative_path")
    if not relative_path:
        raise KeyError("Parsed JSON does not contain reference_image.relative_path.")
    return resolve_project_path(relative_path)


def resolve_expected_state(
    parsed_result: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    relative_path = parsed_result.get("expected_state_path")
    if relative_path:
        raw = Path(str(relative_path).replace("\\", "/"))
        expected_path = raw if raw.is_absolute() else ROOT_DIR / raw
    else:
        info = parsed_result.get("file_info", {})
        model_id = info.get("model_id")
        step_id = info.get("step_id")
        if not model_id or not step_id:
            raise KeyError("Cannot infer expected-state path: model_id or step_id missing.")
        expected_path = ROOT_DIR / "ground_truth" / str(model_id) / f"{step_id}.json"

    expected_path = expected_path.resolve()
    return expected_path, load_json(expected_path)


def extract_error_parts(parsed_result: dict[str, Any]) -> list[dict[str, Any]]:
    model_response = parsed_result.get("model_response", {})
    if not isinstance(model_response, dict):
        raise TypeError("model_response must be a JSON object.")

    detected_parts = model_response.get("detected_parts", [])
    if not isinstance(detected_parts, list):
        raise TypeError("model_response.detected_parts must be a list.")

    return [
        part
        for part in detected_parts
        if isinstance(part, dict)
        and str(part.get("error_type", "uncertain")).lower() != "correct"
    ]


def choose_primary_error(
    error_parts: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not error_parts:
        return None

    def score(part: dict[str, Any]) -> float:
        try:
            return float(part.get("confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0

    return max(error_parts, key=score)


def find_expected_part(
    expected_state: dict[str, Any],
    part_id: str,
) -> Optional[dict[str, Any]]:
    for part in expected_state.get("expected_parts", []):
        if isinstance(part, dict) and str(part.get("part_id", "")) == part_id:
            return part
    return None


def convert_target_position(position: Optional[str]) -> str:
    mapping = {
        "LEFT": "left",
        "RIGHT": "right",
        "CENTER": "center",
        "TOP": "top",
        "BOTTOM": "bottom",
        "FRONT": "center",
        "BACK": "center",
    }
    if not position:
        return "center"
    return mapping.get(str(position).upper(), "center")


def load_part_library() -> dict[str, list[str]]:
    path = ROOT_DIR / "config" / "part_library.json"
    if not path.is_file():
        return {}
    payload = load_json(path)
    return {
        str(part_id): [str(alias) for alias in aliases]
        if isinstance(aliases, list)
        else [str(aliases)]
        for part_id, aliases in payload.items()
    }


PART_LIBRARY = load_part_library()


def contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def readable_part_prompt(part_id: str) -> str:
    english_aliases = [
        alias.strip()
        for alias in PART_LIBRARY.get(part_id, [])
        if alias.strip() and not contains_chinese(alias)
    ]
    if english_aliases:
        return max(english_aliases, key=lambda value: (len(value.split()), len(value)))

    fallback = {
        "EYE_BALL": "white ball with black pupil",
        "JOINT_BLUE_Y": "blue Y shaped joint",
        "PLATE_BLUE_TRIANGLE": "blue triangular plate",
        "CONNECTOR_ORANGE": "orange U shaped connector",
        "BLOCK_YELLOW_CUBE": "yellow cube block",
        "JOINT_YELLOW_H": "yellow H shaped joint",
        "PIN_YELLOW": "yellow flat head pin",
        "WHEEL_BLUE_LARGE": "large blue wheel",
        "WHEEL_BLUE_SMALL": "small blue wheel",
        "ROD_GREEN_LONG": "long green rod",
        "PIN_RED_SHORT": "short red cylinder pin",
        "BLOCK_GREEN_4HOLE_2PEG": "green rectangular four hole block with two pegs",
        "LINK_RED_3HOLE": "red three hole link",
        "LINK_GREEN_5HOLE": "green five hole link",
        "LINK_BLUE_5HOLE": "blue five hole link",
    }
    return fallback.get(part_id, part_id.replace("_", " ").lower())


def choose_localization_strategy(
    *,
    error_part: dict[str, Any],
    test_image_path: Path,
    reference_image_path: Path,
    expected_state: dict[str, Any],
) -> dict[str, Any]:
    error_type = str(error_part.get("error_type", "uncertain")).lower()
    part_id = str(error_part.get("part_id", "unknown_part"))
    description = str(error_part.get("description", "")).strip()

    expected_part = find_expected_part(expected_state, part_id)
    target_position = convert_target_position(
        expected_part.get("position") if expected_part else None
    )
    part_prompt = readable_part_prompt(part_id)

    if error_type == "missingpart":
        return {
            "image_path": str(reference_image_path),
            "prompt": part_prompt,
            "target_position": target_position,
            "localization_role": "reference_missing_part_location",
            "expected_part": expected_part,
        }

    if error_type == "extrapart":
        prompt = (
            part_prompt
            if part_id and not part_id.startswith("unknown")
            else description or "extra construction toy part"
        )
        return {
            "image_path": str(test_image_path),
            "prompt": prompt,
            "target_position": target_position,
            "localization_role": "test_extra_part_location",
            "expected_part": expected_part,
        }

    if error_type in {"wrongpart", "positionerror", "criticalerror"}:
        prompt = (
            part_prompt
            if part_id and not part_id.startswith("unknown")
            else description or "incorrect construction toy part"
        )
        return {
            "image_path": str(test_image_path),
            "prompt": prompt,
            "target_position": target_position,
            "localization_role": "test_error_part_location",
            "expected_part": expected_part,
        }

    return {
        "image_path": str(test_image_path),
        "prompt": (
            part_prompt
            if part_id and not part_id.startswith("unknown")
            else description or "incorrect construction toy part"
        ),
        "target_position": target_position,
        "localization_role": "uncertain",
        "expected_part": expected_part,
    }


def create_run_paths(output_dir: str | Path) -> RunPaths:
    run_dir = Path(output_dir).expanduser().resolve()
    annotated_dir = run_dir / "images" / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        results_json=run_dir / "results.json",
        run_summary_json=run_dir / "run_summary.json",
        annotated_dir=annotated_dir,
    )


def write_run_summary(
    *,
    paths: RunPaths,
    parsed_json_path: Path,
    image_stem: str,
    success: bool,
    error_parts: list[dict[str, Any]],
    localization_strategy: dict[str, Any],
    localization_result: dict[str, Any],
    notes: Optional[list[str]] = None,
) -> None:
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "completed" if success else "partial",
        "source": "current_parsed_json",
        "image_stem": image_stem,
        "parsed_json_path": str(parsed_json_path),
        "success": success,
        "error_part_count": len(error_parts),
        "parameters": {
            "localization_role": localization_strategy.get("localization_role"),
            "target_position": localization_strategy.get("target_position"),
            "localization_prompt": localization_strategy.get("prompt"),
        },
        "outputs": {
            "results_json": str(paths.results_json),
            "annotated_image": localization_result.get("annotated_image_path"),
        },
        "notes": notes or [],
    }
    save_json(paths.run_summary_json, payload)


def process_one(
    parsed_json_path: str | Path,
    output_dir: str | Path,
    *,
    box_threshold: float = 0.15,
    text_threshold: float = 0.10,
    max_detections: int = 10,
    device: str = "auto",
    overwrite: bool = False,
    localizer: Any | None = None,
) -> Path:
    """處理一份既有 Vision parsed JSON，回傳 results.json 路徑。"""
    parsed_path = Path(parsed_json_path).expanduser().resolve()
    paths = create_run_paths(output_dir)

    if paths.results_json.is_file() and not overwrite:
        print(f"[INFO] Existing results found; skipped: {paths.results_json}")
        return paths.results_json

    parsed_result = load_json(parsed_path)
    if not bool(parsed_result.get("success", False)):
        raise RuntimeError("The selected parsed JSON is not a successful Vision result.")

    model_response = parsed_result.get("model_response", {})
    if not isinstance(model_response, dict):
        raise TypeError("model_response must be a JSON object.")

    test_image_path = resolve_test_image(parsed_result)
    reference_image_path = resolve_reference_image(parsed_result)
    expected_state_path, expected_state = resolve_expected_state(parsed_result)

    error_parts = extract_error_parts(parsed_result)
    error_reports = adapt_vision_result(parsed_result)
    primary_error = error_reports[0] if error_reports else None
    localization_strategy: dict[str, Any] = {}
    localization_result: dict[str, Any] = {}
    localization_results: list[dict[str, Any]] = []

    if primary_error is None:
        localization_result = {
            "status": "skipped",
            "reason": "no_error_parts",
            "annotated_image_path": None,
            "error_message": None,
        }
    else:
        pipeline_error: str | None = None
        try:
            pipeline = localizer or LocalizationPipeline(device=device)
        except Exception as exc:
            pipeline = None
            pipeline_error = f"{type(exc).__name__}: {exc}"
        for index, report in enumerate(error_reports):
            strategy = choose_localization_strategy(
                error_part=report,
                test_image_path=test_image_path,
                reference_image_path=reference_image_path,
                expected_state=expected_state,
            )
            localization_image = Path(strategy["image_path"])
            try:
                if pipeline is None:
                    raise RuntimeError(pipeline_error or "Localization unavailable")
                current = pipeline.localize(
                    image_path=str(localization_image),
                    text_prompt=str(strategy["prompt"]),
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_position=str(strategy["target_position"]),
                    max_detections=max_detections,
                    output_dir=str(paths.annotated_dir / f"error_{index + 1:02d}"),
                )
                if not isinstance(current, dict):
                    raise TypeError("LocalizationPipeline.localize() must return a dict.")
            except Exception as exc:
                current = {"status": "error", "selected_bbox": None, "annotated_image_path": None, "error_message": f"{type(exc).__name__}: {exc}"}
            current.update({
                "localization_role": strategy.get("localization_role"),
                "localization_source_image": str(localization_image),
                "localization_prompt": strategy.get("prompt"),
                "target_position": strategy.get("target_position"),
            })
            bbox = current.get("selected_bbox")
            report["bbox"] = [float(value) for value in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None
            report["localization_strategy"] = strategy
            report["localization"] = current
            localization_results.append(current)
        localization_strategy = error_reports[0]["localization_strategy"]
        localization_result = localization_results[0]

    payload = {
        "schema_version": "1.1",
        "source_mode": "existing_parsed_json",
        "parsed_json_path": str(parsed_path),
        "image_stem": image_stem_from_parsed_json(parsed_path),
        "test_image_path": str(test_image_path),
        "reference_image_path": str(reference_image_path),
        "expected_state_path": str(expected_state_path),
        "vision_result": parsed_result,
        "model_response": model_response,
        "error_parts": error_parts,
        "error_reports": error_reports,
        "primary_error": primary_error,
        "localization_strategy": localization_strategy,
        "localization": localization_result,
        "localizations": localization_results,
    }
    save_json(paths.results_json, payload)

    localization_status = str(localization_result.get("status", "unknown"))
    success = True

    notes: list[str] = []
    if localization_result.get("error_message"):
        notes.append(str(localization_result["error_message"]))
    if len(error_reports) > 1:
        notes.append(f"Localized {len(error_reports)} independent ErrorReports.")

    write_run_summary(
        paths=paths,
        parsed_json_path=parsed_path,
        image_stem=image_stem_from_parsed_json(parsed_path),
        success=success,
        error_parts=error_parts,
        localization_strategy=localization_strategy,
        localization_result=localization_result,
        notes=notes,
    )

    print("=" * 70)
    print("Pipeline smoke test finished")
    print(f"Results JSON: {paths.results_json}")
    print(f"Run summary:  {paths.run_summary_json}")
    print("=" * 70)
    return paths.results_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run localization from an existing Vision parsed JSON."
    )
    parser.add_argument("--parsed-json", type=Path, default=None)
    parser.add_argument("--image-stem", type=str, default=None)
    parser.add_argument("--parsed-json-dir", type=Path, default=PARSED_JSON_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.10)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.parsed_json is not None:
        parsed_path = args.parsed_json.expanduser().resolve()
    else:
        parsed_path = find_latest_parsed_json(
            image_stem=args.image_stem,
            parsed_json_dir=args.parsed_json_dir,
        )

    image_stem = image_stem_from_parsed_json(parsed_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_SINGLE_RUN_ROOT / image_stem
    )

    result_path = process_one(
        parsed_json_path=parsed_path,
        output_dir=output_dir,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_detections=args.max_detections,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"[SUCCESS] {result_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
