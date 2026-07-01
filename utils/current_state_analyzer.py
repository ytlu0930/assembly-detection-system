"""
utils/current_state_analyzer.py

Core GPT-4o Vision analyzer for reference-guided image comparison.
This module is the only place that loads the vision prompt, calls the GPT
Vision API, validates the schema, and writes raw/parsed/failed model logs.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from jsonschema import ValidationError, validate
from openai import AzureOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "vision_v2.txt"
SCHEMA_PATH = PROJECT_ROOT / "schema" / "vision_output_schema.json"
GROUND_TRUTH_DIR = PROJECT_ROOT / "ground_truth"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

load_dotenv(dotenv_path=ENV_PATH)


def safe_relative_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def parse_filename(image_path: str | Path) -> dict[str, str]:
    """Parse names such as model03_step03_missingpart-A01_front_01.jpg."""
    path = Path(image_path)
    parts = path.stem.split("_")
    info = {
        "image_name": path.name,
        "relative_path": safe_relative_path(path),
        "model_id": "",
        "step_id": "",
        "ground_truth": "",
        "target_part": "",
        "view_angle": "",
        "image_index": "",
    }

    if len(parts) < 5:
        return info

    error_part = parts[2]
    if "-" in error_part:
        ground_truth, target_part = error_part.split("-", 1)
    else:
        ground_truth, target_part = error_part, ""

    info.update({
        "model_id": parts[0],
        "step_id": parts[1],
        "ground_truth": ground_truth,
        "target_part": target_part,
        "view_angle": parts[3],
        "image_index": parts[4],
    })
    return info


def load_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_mime_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    raise ValueError(f"Unsupported image extension: {path}")


def image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{get_mime_type(path)};base64,{encoded}"


def build_prompt(
    prompt_template: str,
    expected_state: dict[str, Any],
    filename_info: dict[str, Any],
    image_path: Path,
    reference_image_path: Path,
) -> str:
    replacements = {
        "model_id": filename_info.get("model_id", ""),
        "step_id": filename_info.get("step_id", ""),
        "step_name": expected_state.get("step_name", ""),
        "view_angle": filename_info.get("view_angle", ""),
        "reference_image_name": reference_image_path.name,
        "test_image_name": image_path.name,
        "expected_state_json": json.dumps(expected_state, ensure_ascii=False, indent=2),
    }

    prompt = prompt_template
    for key, value in replacements.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt


def extract_json(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    fence = chr(96) * 3
    if cleaned.lower().startswith(fence + "json"):
        cleaned = cleaned[len(fence + "json"):].strip()
    elif cleaned.startswith(fence):
        cleaned = cleaned[len(fence):].strip()
    if cleaned.endswith(fence):
        cleaned = cleaned[:-len(fence)].strip()
    return json.loads(cleaned)


def max_confidence(model_response: dict[str, Any]) -> float:
    values = [
        float(part["confidence"])
        for part in model_response.get("detected_parts", [])
        if isinstance(part.get("confidence"), (int, float))
    ]
    return max(values) if values else 0.0


def calculate_decision_level(ground_truth: str, is_error: bool | str) -> str:
    expected_error = ground_truth != "correct"
    predicted_error = is_error.lower() == "true" if isinstance(is_error, str) else bool(is_error)

    if expected_error and predicted_error:
        return "TP"
    if not expected_error and not predicted_error:
        return "TN"
    if not expected_error and predicted_error:
        return "FP"
    return "FN"


def get_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is missing in .env")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY is missing in .env")

    return AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=endpoint,
        api_key=api_key,
    )


def make_log_paths(image_path: Path, attempt: int) -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = image_path.stem
    logs_dir = PROJECT_ROOT / "logs"
    return {
        "raw_text": logs_dir / "current_raw_responses" / f"{safe_name}_attempt{attempt}_raw_{timestamp}.txt",
        "raw_json": logs_dir / "current_raw_responses" / f"{safe_name}_attempt{attempt}_response_{timestamp}.json",
        "parsed": logs_dir / "current_parsed_json" / f"{safe_name}_parsed_{timestamp}.json",
        "failed": logs_dir / "current_parse_failed" / f"{safe_name}_attempt{attempt}_failed_{timestamp}.json",
    }


def save_failure(
    path: Path,
    error: str,
    detail: str,
    raw_text: str,
    filename_info: dict[str, Any],
    raw_response_path: Path | None = None,
    parsed_candidate: dict[str, Any] | None = None,
) -> None:
    payload = {
        "success": False,
        "error": error,
        "detail": detail,
        "file_info": filename_info,
        "raw_response_path": safe_relative_path(raw_response_path) if raw_response_path else "",
        "raw_response": raw_text,
        "parsed_candidate": parsed_candidate,
    }
    save_json(payload, path)


def analyze_image(
    image_path: str,
    reference_image_path: str,
    expected_state_path: str,
    filename_info: dict,
    max_retry: int = 3,
) -> dict[str, Any]:
    """
    Analyze one test image against one correct reference image.

    This function owns prompt loading, expected-state loading, GPT-4o Vision API
    calls, JSON parsing, schema validation, and current_* log writing.
    """
    image_path_obj = Path(image_path)
    reference_image_path_obj = Path(reference_image_path)
    expected_state_path_obj = Path(expected_state_path)
    file_info = dict(filename_info or {})

    result_base: dict[str, Any] = {
        "success": False,
        "model_response": None,
        "parsed_json_path": "",
        "raw_response_path": "",
        "error": "",
        "file_info": file_info,
    }

    try:
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Test image not found: {image_path_obj}")
        if not reference_image_path_obj.exists():
            raise FileNotFoundError(f"Reference image not found: {reference_image_path_obj}")

        prompt_template = load_text(PROMPT_PATH)
        schema = load_json(SCHEMA_PATH)
        expected_state = load_json(expected_state_path_obj)
        prompt_text = build_prompt(prompt_template, expected_state, file_info, image_path_obj, reference_image_path_obj)
        reference_data_url = image_to_data_url(reference_image_path_obj)
        test_data_url = image_to_data_url(image_path_obj)
        client = get_client()
        deployment = os.getenv("GPT4O_DEPLOYMENT", "gpt-4o")
    except Exception as exc:
        result_base["error"] = "Initialization failed"
        result_base["detail"] = str(exc)
        return result_base

    last_error = ""
    last_detail = ""

    for attempt in range(1, max_retry + 1):
        paths = make_log_paths(image_path_obj, attempt)
        raw_text = ""
        parsed_json: dict[str, Any] | None = None

        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=deployment,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON-only reference-guided vision inspector. "
                            "Return only the requested JSON object. Do not include markdown or extra keys."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "text", "text": "Correct Reference Image:"},
                            {"type": "image_url", "image_url": {"url": reference_data_url, "detail": "high"}},
                            {"type": "text", "text": "Test Image:"},
                            {"type": "image_url", "image_url": {"url": test_data_url, "detail": "high"}},
                        ],
                    },
                ],
            )
            elapsed = round(time.time() - start_time, 3)
            raw_text = response.choices[0].message.content or ""

            paths["raw_text"].parent.mkdir(parents=True, exist_ok=True)
            paths["raw_text"].write_text(raw_text, encoding="utf-8")
            save_json(response.model_dump(), paths["raw_json"])

            try:
                parsed_json = extract_json(raw_text)
            except json.JSONDecodeError as exc:
                last_error = "JSON parse failed"
                last_detail = str(exc)
                save_failure(paths["failed"], last_error, last_detail, raw_text, file_info, paths["raw_text"])
                if attempt < max_retry:
                    time.sleep(1)
                    continue
                return {
                    **result_base,
                    "error": last_error,
                    "detail": last_detail,
                    "raw_response_path": safe_relative_path(paths["raw_text"]),
                    "failed_path": safe_relative_path(paths["failed"]),
                    "attempt": attempt,
                }

            try:
                validate(instance=parsed_json, schema=schema)
            except ValidationError as exc:
                last_error = "Schema validation failed"
                last_detail = exc.message
                save_failure(paths["failed"], last_error, last_detail, raw_text, file_info, paths["raw_text"], parsed_json)
                if attempt < max_retry:
                    time.sleep(1)
                    continue
                return {
                    **result_base,
                    "error": last_error,
                    "detail": last_detail,
                    "model_response": parsed_json,
                    "raw_response_path": safe_relative_path(paths["raw_text"]),
                    "failed_path": safe_relative_path(paths["failed"]),
                    "attempt": attempt,
                }

            parsed_payload = {
                "success": True,
                "file_info": file_info,
                "reference_image": {
                    "image_name": reference_image_path_obj.name,
                    "relative_path": safe_relative_path(reference_image_path_obj),
                },
                "test_image": {
                    "image_name": image_path_obj.name,
                    "relative_path": safe_relative_path(image_path_obj),
                },
                "expected_state_path": safe_relative_path(expected_state_path_obj),
                "model_response": parsed_json,
                "runtime": {
                    "attempt": attempt,
                    "response_time_sec": elapsed,
                    "prompt_path": safe_relative_path(PROMPT_PATH),
                    "schema_path": safe_relative_path(SCHEMA_PATH),
                },
            }
            save_json(parsed_payload, paths["parsed"])

            return {
                "success": True,
                "model_response": parsed_json,
                "parsed_json_path": safe_relative_path(paths["parsed"]),
                "raw_response_path": safe_relative_path(paths["raw_text"]),
                "error": None,
                "detail": "",
                "attempt": attempt,
                "response_time_sec": elapsed,
                "file_info": file_info,
                "reference_image": parsed_payload["reference_image"],
                "expected_state_path": safe_relative_path(expected_state_path_obj),
                "confidence": max_confidence(parsed_json),
            }

        except Exception as exc:
            last_error = "API request failed"
            last_detail = str(exc)
            save_failure(paths["failed"], last_error, last_detail, raw_text, file_info, None, parsed_json)
            if attempt < max_retry:
                time.sleep(2)
                continue
            return {
                **result_base,
                "error": last_error,
                "detail": last_detail,
                "failed_path": safe_relative_path(paths["failed"]),
                "attempt": attempt,
            }

    return {**result_base, "error": last_error or "Unknown failure", "detail": last_detail}


if __name__ == "__main__":
    sample_image = PROJECT_ROOT / "input" / "normal" / "model03_step03" / "model03_step03_correct-01_front_01.jpg"
    info = parse_filename(sample_image)
    sample_expected = GROUND_TRUTH_DIR / info["model_id"] / f"{info['step_id']}.json"
    result = analyze_image(
        image_path=str(sample_image),
        reference_image_path=str(sample_image),
        expected_state_path=str(sample_expected),
        filename_info=info,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
