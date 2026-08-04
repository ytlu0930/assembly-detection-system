"""
pipeline_smoke_test.py

用途：
1. 不重新呼叫 Azure OpenAI Vision。
2. 直接讀取組員 B 已產生的 logs/current_parsed_json。
3. 從 Vision JSON 取得主要錯誤。
4. 依錯誤類型決定定位測試圖或正確參考圖。
5. 呼叫 LocalizationPipeline 定位目標區域。
6. 輸出 results.json、annotated image 與 run_summary.json。

目前用途仍是 Smoke Test：
只驗證 Vision JSON -> Error-aware Localization -> Output Manager 是否可串接。
尚未包含 Correction SOP 與 OpenAI 圖片生成。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.localization_pipeline import LocalizationPipeline
from utils.output_manager import create_run_output, write_run_summary


ROOT_DIR = Path(__file__).resolve().parent
PARSED_JSON_DIR = ROOT_DIR / "logs" / "current_parsed_json"
OUTPUT_ROOT = ROOT_DIR / "output"

TEST_IMAGE_STEM = "model03_step03_missingpart-A01_front_01"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object, got {type(data).__name__}: {path}")
    return data


def find_latest_parsed_json(image_stem: str) -> Path:
    pattern = f"{image_stem}_parsed_*.json"
    files = sorted(
        PARSED_JSON_DIR.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(
            "No parsed JSON matched:\n"
            f"{PARSED_JSON_DIR / pattern}"
        )
    return files[0]


def resolve_project_path(relative_path: str) -> Path:
    normalized = str(relative_path).replace("\\", "/")
    path = ROOT_DIR / Path(normalized)
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
    return resolve_project_path(str(relative_path))


def resolve_reference_image(parsed_result: dict[str, Any]) -> Path:
    relative_path = parsed_result.get("reference_image", {}).get("relative_path")
    if not relative_path:
        raise KeyError(
            "Parsed JSON does not contain reference_image.relative_path."
        )
    return resolve_project_path(str(relative_path))


def resolve_expected_state(
    parsed_result: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    relative_path = parsed_result.get("expected_state_path")
    if relative_path:
        expected_path = ROOT_DIR / Path(str(relative_path).replace("\\", "/"))
    else:
        file_info = parsed_result.get("file_info", {})
        model_id = file_info.get("model_id")
        step_id = file_info.get("step_id")
        if not model_id or not step_id:
            raise KeyError(
                "Cannot infer expected-state path because model_id or step_id is missing."
            )
        expected_path = ROOT_DIR / "ground_truth" / model_id / f"{step_id}.json"
    expected_state = load_json(expected_path)
    return expected_path, expected_state


def extract_error_parts(
    parsed_result: dict[str, Any],
) -> list[dict[str, Any]]:
    detected_parts = (
        parsed_result.get("model_response", {}).get("detected_parts", [])
    )
    if not isinstance(detected_parts, list):
        raise TypeError("model_response.detected_parts must be a list.")
    return [
        part
        for part in detected_parts
        if isinstance(part, dict)
        and str(part.get("error_type", "uncertain")).lower() != "correct"
    ]


def find_expected_part(
    expected_state: dict[str, Any],
    part_id: str,
) -> dict[str, Any] | None:
    for part in expected_state.get("expected_parts", []):
        if isinstance(part, dict) and part.get("part_id") == part_id:
            return part
    return None


def convert_target_position(position: str | None) -> str:
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


def readable_part_prompt(part_id: str) -> str:
    aliases = {
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
        "PIN_RED_SHORT": "short red pin",
        "BLOCK_GREEN_4HOLE_2PEG": "green rectangular four hole block",
        "LINK_RED_3HOLE": "red three hole link",
        "LINK_GREEN_5HOLE": "green five hole link",
        "LINK_BLUE_5HOLE": "blue five hole link",
    }
    return aliases.get(part_id, part_id.replace("_", " ").lower())


def choose_localization_strategy(
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
        if part_id and not part_id.startswith("unknown"):
            prompt = part_prompt
        else:
            prompt = "extra construction toy part"
        return {
            "image_path": str(test_image_path),
            "prompt": prompt,
            "target_position": "center",
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
        "target_position": "center",
        "localization_role": "uncertain",
        "expected_part": expected_part,
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    print("RUNNING ERROR-AWARE PIPELINE SMOKE TEST")
    print(f"FILE: {Path(__file__).resolve()}")

    parsed_json_path = find_latest_parsed_json(TEST_IMAGE_STEM)

    print("=" * 70)
    print("1. Load existing Vision JSON")
    print("=" * 70)
    print(f"Parsed JSON: {parsed_json_path}")

    parsed_result = load_json(parsed_json_path)
    if not parsed_result.get("success", False):
        raise RuntimeError(
            "The selected parsed JSON is not a successful Vision result."
        )

    model_response = parsed_result.get("model_response", {})
    print(json.dumps(model_response, ensure_ascii=False, indent=2))

    test_image_path = resolve_test_image(parsed_result)
    reference_image_path = resolve_reference_image(parsed_result)
    expected_state_path, expected_state = resolve_expected_state(parsed_result)

    output_paths = create_run_output(
        category="pipeline",
        experiment="error_aware_localization_smoke_test",
        output_root=OUTPUT_ROOT,
        image_subdirs=["annotated"],
    )

    error_parts = extract_error_parts(parsed_result)
    localization_result: dict[str, Any] = {}
    localization_strategy: dict[str, Any] = {}

    if not error_parts:
        print("=" * 70)
        print("No error detected; localization skipped.")
        print("=" * 70)
    else:
        primary_error = error_parts[0]

        localization_strategy = choose_localization_strategy(
            error_part=primary_error,
            test_image_path=test_image_path,
            reference_image_path=reference_image_path,
            expected_state=expected_state,
        )

        localization_image = Path(localization_strategy["image_path"])
        localization_prompt = str(localization_strategy["prompt"])
        target_position = str(localization_strategy["target_position"])

        print("=" * 70)
        print("2. Error-aware Localization")
        print("=" * 70)
        print(f"Error type: {primary_error.get('error_type')}")
        print("Localization role:", localization_strategy["localization_role"])
        print(f"Test image: {test_image_path}")
        print(f"Reference image: {reference_image_path}")
        print(f"Localization image: {localization_image}")
        print(f"Prompt: {localization_prompt}")
        print(f"Target position: {target_position}")

        localization_pipeline = LocalizationPipeline(device="auto")

        localization_result = localization_pipeline.localize(
            image_path=str(localization_image),
            text_prompt=localization_prompt,
            box_threshold=0.15,
            text_threshold=0.10,
            target_position=target_position,
            max_detections=10,
            output_dir=str(output_paths.image_subdirs["annotated"]),
        )

        localization_result["localization_role"] = localization_strategy[
            "localization_role"
        ]
        localization_result["localization_source_image"] = str(localization_image)
        localization_result["localization_prompt"] = localization_prompt
        localization_result["target_position"] = target_position

        print(json.dumps(localization_result, ensure_ascii=False, indent=2))

    payload = {
        "parsed_json_path": str(parsed_json_path),
        "test_image_path": str(test_image_path),
        "reference_image_path": str(reference_image_path),
        "expected_state_path": str(expected_state_path),
        "vision_result": parsed_result,
        "error_parts": error_parts,
        "localization_strategy": localization_strategy,
        "localization": localization_result,
    }

    save_json(output_paths.json_path, payload)

    localization_status = (
        localization_result.get("status") if error_parts else "skipped"
    )
    success = (
        not error_parts
        or localization_status in {"success", "no_detection"}
    )

    write_run_summary(
        output_paths,
        status="completed" if success else "partial",
        input_count=1,
        success_count=1 if success else 0,
        failure_count=0 if success else 1,
        parameters={
            "source": "current_parsed_json",
            "test_image_stem": TEST_IMAGE_STEM,
            "box_threshold": 0.15,
            "text_threshold": 0.10,
            "localization_role": localization_strategy.get("localization_role"),
            "target_position": localization_strategy.get("target_position"),
        },
        output_paths={
            "results_json": str(output_paths.json_path),
            "annotated_image": localization_result.get("annotated_image_path"),
        },
        notes=(
            [localization_result["error_message"]]
            if localization_result.get("error_message")
            else []
        ),
    )

    print("=" * 70)
    print("Smoke test finished")
    print(f"Output: {output_paths.run_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()