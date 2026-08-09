"""Build safe dry-run packages for affected-part Prompt A/B/C."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.affected_part_candidate_builder import build_affected_part_candidates
from utils.experiment_request_guard import ExperimentLockedError, ExperimentRequestGuard

CURRENT_SCHEMA = PROJECT_ROOT / "schema/vision_output_schema.json"
PART_LIBRARY = PROJECT_ROOT / "config/part_library.json"
VARIANTS = {
    "baseline": PROJECT_ROOT / "experiments/prompts/vision_affected_parts_baseline.txt",
    "reference": PROJECT_ROOT / "experiments/prompts/vision_affected_parts_reference_guided.txt",
    "reference_candidate": PROJECT_ROOT / "experiments/prompts/vision_affected_parts_reference_candidate.txt",
}
CASES = (
    ("missingpart-A01", "input/missingpart/model03_step03/model03_step03_missingpart-A01_front_01.jpg", "missingpart"),
    ("missingpart-B01", "input/missingpart/model03_step03/model03_step03_missingpart-B01_front_01.jpg", "missingpart"),
    ("extrapart-A01", "input/extrapart/model03_step03/model03_step03_extrapart-A01_front_01.jpg", "extrapart"),
    ("wrongpart-A01", "input/wrongpart/model03_step03/model03_step03_wrongpart-A01_front_01.jpg", "wrongpart"),
    ("wrongpart-B01", "input/wrongpart/model03_step03/model03_step03_wrongpart-B01_front_01.jpg", "wrongpart"),
    ("correct-control", "input/normal/model03_step01/model03_step01_correct-01_front_01.jpg", "correct"),
)
MAX_LOGICAL_REQUESTS = 18
DEFAULT_API_VERSION = "2024-12-01-preview"
EXPERIMENT_ID = "affected-part-prompt-ab-20260809"
ALLOWED_UNKNOWN_PART_IDS = {"UNKNOWN", "UNRESOLVED", "UNKNOWN_PART", "UNRESOLVED_PART", "UNKNOWN_EXTRA_PART"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    value = json.loads(cleaned.strip())
    if not isinstance(value, dict):
        raise ValueError("Vision response must be a JSON object")
    return value


def candidate_leakage_audit() -> dict[str, Any]:
    source = (PROJECT_ROOT / "utils/affected_part_candidate_builder.py").read_text(encoding="utf-8")
    forbidden = ("affected_part_eval_ground_truth", "affected_parts_review_template", "missingpart-A01", "PIN_RED_SHORT")
    hits = [value for value in forbidden if value in source]
    return {"status": "PASS" if not hits else "FAIL", "forbidden_hits": hits}


def safe_environment_preflight(*, load_environment: bool) -> dict[str, Any]:
    if load_environment:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("GPT4O_DEPLOYMENT", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION).strip()
    return {
        "provider": "azure_openai_chat_completions",
        "model": deployment or None,
        "api_version": api_version,
        "endpoint_configured": bool(endpoint),
        "key_configured": bool(key),
        "deployment_configured": bool(deployment),
        "ready": bool(endpoint and key and deployment and api_version),
    }


def _case_context(case: tuple[str, str, str]) -> dict[str, Any]:
    case_id, test_image, error_type = case
    name = Path(test_image).name
    pieces = name.removesuffix(".jpg").split("_")
    model_id, step_id, view_angle = pieces[0], pieces[1], pieces[-2]
    reference = f"input/normal/{model_id}_{step_id}/{model_id}_{step_id}_correct-01_{view_angle}_01.jpg"
    expected = PROJECT_ROOT / f"ground_truth/{model_id}/{step_id}.json"
    return {
        "case_id": case_id, "model_id": model_id, "step_id": step_id,
        "view_angle": view_angle, "error_type_hint": error_type,
        "test_image": test_image, "reference_image": reference,
        "expected_state": expected.relative_to(PROJECT_ROOT).as_posix(),
    }


def _case_preflight(selected_cases: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    confirmed_path = PROJECT_ROOT / "analysis/affected_part_eval_ground_truth.csv"
    review_path = PROJECT_ROOT / "analysis/affected_parts_review_template.csv"
    with confirmed_path.open(encoding="utf-8-sig", newline="") as handle:
        confirmed = {row["image_id"] for row in csv.DictReader(handle) if row.get("review_status") == "confirmed"}
    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        review = {row["image_id"]: row.get("annotation_status", "unresolved") for row in csv.DictReader(handle)}
    rows = []
    for case in selected_cases:
        context = _case_context(case)
        image_id = Path(context["test_image"]).name
        rows.append({
            "case_id": context["case_id"], "image": context["test_image"],
            "reference": context["reference_image"],
            "gt_status": "confirmed" if image_id in confirmed else review.get(image_id, "not_in_frozen_evaluation"),
            "variant_count": 3,
        })
    return rows


def build_packages(*, variants: list[str], case_limit: int | None, output_dir: Path) -> dict[str, Any]:
    selected_cases = list(CASES[:case_limit] if case_limit is not None else CASES)
    package_paths = []
    schema_sha = _sha256(CURRENT_SCHEMA)
    prompts = {variant: VARIANTS[variant].read_text(encoding="utf-8") for variant in variants}
    for case in selected_cases:
        for variant in variants:
            prompt_template = prompts[variant]
            context = _case_context(case)
            expected_path = PROJECT_ROOT / context["expected_state"]
            expected_payload = json.loads(expected_path.read_text(encoding="utf-8"))
            candidate_payload = None
            if variant == "reference_candidate":
                candidate_payload = build_affected_part_candidates(
                    model_id=context["model_id"], step_id=context["step_id"],
                    expected_state=expected_payload, part_library=PART_LIBRARY,
                    error_type=context["error_type_hint"],
                )
            prompt_sections = [
                prompt_template.rstrip(),
                "\nCase context:\n" + json.dumps(context, ensure_ascii=False, indent=2),
                "\nexpected_state JSON:\n" + json.dumps(expected_payload, ensure_ascii=False, indent=2),
            ]
            if candidate_payload is not None:
                prompt_sections.append("\nCandidate affected parts:\n" + json.dumps(candidate_payload, ensure_ascii=False, indent=2))
            package_dir = output_dir / "packages" / variant / context["case_id"]
            package_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = package_dir / "prompt.txt"
            metadata_path = package_dir / "request_metadata.json"
            prompt_path.write_text("\n".join(prompt_sections) + "\n", encoding="utf-8")
            metadata = {
                **context,
                "variant": variant,
                "prompt_source": VARIANTS[variant].relative_to(PROJECT_ROOT).as_posix(),
                "prompt_sha256": _sha256(prompt_path),
                "schema": CURRENT_SCHEMA.relative_to(PROJECT_ROOT).as_posix(),
                "schema_sha256": schema_sha,
                "schema_condition": "current",
                "candidate_part_ids": candidate_payload["candidate_part_ids"] if candidate_payload else [],
                "mode": "dry-run",
                "contains_api_key": False,
                "future_result_contract": {
                    "variant": variant, "case_id": context["case_id"], "view_angle": context["view_angle"],
                    "request_id": None, "parsed_output": None, "latency": None,
                    "token_usage": None, "predicted_part_ids": [], "confidence": None,
                    "error_type": None,
                },
            }
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            package_paths.append(metadata_path.resolve().as_posix())
    result = {
        "mode": "dry-run", "number_of_cases": len(selected_cases),
        "number_of_variants": len(variants), "estimated_requests": len(selected_cases) * len(variants),
        "case_ids": [case[0] for case in selected_cases], "variants": variants,
        "current_schema": CURRENT_SCHEMA.relative_to(PROJECT_ROOT).as_posix(),
        "case_preflight": _case_preflight(selected_cases),
        "package_paths": package_paths, "api_calls_performed": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dry_run_plan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _prediction(parsed: dict[str, Any]) -> tuple[list[str], list[float], str, bool]:
    error_type = str(parsed.get("overall_error_type") or "uncertain")
    if error_type == "correct":
        return [], [], error_type, False
    scores: dict[str, float] = {}
    for part in parsed.get("detected_parts") or []:
        if not isinstance(part, dict) or str(part.get("error_type") or "") == "correct":
            continue
        part_id = str(part.get("part_id") or "").strip().upper()
        if part_id:
            scores[part_id] = max(scores.get(part_id, 0.0), float(part.get("confidence") or 0.0))
    part_ids = sorted(scores)
    unknown = any(item.startswith(("UNKNOWN", "UNRESOLVED")) for item in part_ids)
    return part_ids, [scores[item] for item in part_ids], error_type, unknown


def enforce_candidate_constraint(variant: str, candidate_part_ids: list[str], predicted_part_ids: list[str]) -> dict[str, Any]:
    if variant != "reference_candidate":
        return {"candidate_constraint_status": "not_applicable", "candidate_constraint_violations": []}
    allowed = {str(value).upper() for value in candidate_part_ids} | ALLOWED_UNKNOWN_PART_IDS
    violations = sorted({str(value).upper() for value in predicted_part_ids if str(value).upper() not in allowed})
    return {
        "candidate_constraint_status": "violation" if violations else "valid",
        "candidate_constraint_violations": violations,
    }


def execute_packages(*, plan: dict[str, Any], output_dir: Path, guard: ExperimentRequestGuard, retry_failed: bool = False, client: Any | None = None) -> dict[str, Any]:
    if plan["number_of_cases"] != 6 or plan["number_of_variants"] != 3 or plan["estimated_requests"] != MAX_LOGICAL_REQUESTS:
        raise RuntimeError("Execution gate failed: expected exactly 6 cases, 3 variants, and 18 logical requests")
    leakage = candidate_leakage_audit()
    if leakage["status"] != "PASS":
        raise RuntimeError(f"GT leakage audit failed: {leakage['forbidden_hits']}")
    environment = safe_environment_preflight(load_environment=True)
    if not environment["ready"]:
        raise RuntimeError("Azure Vision configuration is incomplete")
    if client is None:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_version=environment["api_version"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
        )
    from jsonschema import validate

    results_dir = output_dir / "results"
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads(CURRENT_SCHEMA.read_text(encoding="utf-8"))
    started_at = _timestamp()
    existing_results = [json.loads(path.read_text(encoding="utf-8")) for path in raw_dir.glob("*.json")]
    skipped = len(existing_results)
    package_paths = [Path(value) for value in plan["package_paths"]]
    for sequence, metadata_path in enumerate(package_paths, start=1):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        case_id, variant = metadata["case_id"], metadata["variant"]
        result_path = raw_dir / f"{sequence:02d}_{case_id}_{variant}.json"
        if result_path.exists():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            matching = prior.get("prompt_sha256") == metadata["prompt_sha256"] and prior.get("schema_sha256") == metadata["schema_sha256"]
            if matching and (prior.get("status") == "success" or not retry_failed):
                continue
        package_id = f"{sequence:02d}:{case_id}:{variant}"
        reservation_id = guard.reserve(package_id, explicit_retry=retry_failed)
        if reservation_id is None:
            continue
        prompt_path = metadata_path.with_name("prompt.txt")
        test_path = PROJECT_ROOT / metadata["test_image"]
        reference_path = PROJECT_ROOT / metadata["reference_image"]
        request_started = _timestamp()
        tick = time.perf_counter()
        base_result = {
            "experiment_id": EXPERIMENT_ID, "sequence": sequence, "variant": variant,
            "case_id": case_id, "image_id": test_path.name, "view_angle": metadata["view_angle"],
            "error_type": metadata["error_type_hint"], "prompt_path": metadata["prompt_source"],
            "prompt_sha256": metadata["prompt_sha256"], "schema_path": metadata["schema"],
            "schema_sha256": metadata["schema_sha256"], "candidate_part_ids": metadata["candidate_part_ids"],
            "provider": environment["provider"], "model": environment["model"],
            "started_at": request_started, "request_attempts": 1,
            "experiment_run_uuid": guard.run_uuid, "request_reservation_id": reservation_id,
        }
        response = None
        parsed = None
        raw_text = ""
        try:
            response = client.chat.completions.create(
                model=environment["model"], temperature=0, response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a strict JSON-only reference-guided vision inspector. Return a JSON object conforming exactly to this current schema: " + json.dumps(schema, ensure_ascii=False)},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_path.read_text(encoding="utf-8")},
                        {"type": "text", "text": "Correct Reference Image:"},
                        {"type": "image_url", "image_url": {"url": _data_url(reference_path), "detail": "high"}},
                        {"type": "text", "text": "Test Image:"},
                        {"type": "image_url", "image_url": {"url": _data_url(test_path), "detail": "high"}},
                    ]},
                ],
            )
            raw_text = response.choices[0].message.content or ""
            parsed = _extract_json(raw_text)
            validate(instance=parsed, schema=schema)
            part_ids, confidences, predicted_error, unknown = _prediction(parsed)
            constraint = enforce_candidate_constraint(variant, metadata["candidate_part_ids"], part_ids)
            payload = {
                **base_result, "finished_at": _timestamp(), "latency_seconds": round(time.perf_counter() - tick, 3),
                "status": "success", "request_id": getattr(response, "id", None),
                "raw_response": response.model_dump() if hasattr(response, "model_dump") else {"content": raw_text},
                "parsed_output": parsed, "predicted_part_ids": part_ids, "predicted_confidence": confidences,
                "predicted_error_type": predicted_error, "unknown_flag": unknown, "usage": _usage(response),
                **constraint,
            }
        except Exception as exc:
            message = str(exc)
            secret = os.getenv("AZURE_OPENAI_API_KEY", "")
            if secret:
                message = message.replace(secret, "[REDACTED]")
            failed_part_ids = _prediction(parsed)[0] if isinstance(parsed, dict) else []
            failed_constraint = enforce_candidate_constraint(variant, metadata["candidate_part_ids"], failed_part_ids)
            payload = {
                **base_result, "finished_at": _timestamp(), "latency_seconds": round(time.perf_counter() - tick, 3),
                "status": "failed", "error": f"{type(exc).__name__}: {message}",
                "raw_response": (
                    response.model_dump() if response is not None and hasattr(response, "model_dump")
                    else {"content": raw_text} if response is not None
                    else None
                ),
                "parsed_output": parsed, "predicted_part_ids": [],
                "predicted_confidence": [], "predicted_error_type": None, "unknown_flag": False,
                "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
                **failed_constraint,
            }
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        guard.finish(reservation_id, "completed" if payload["status"] == "success" else "failed")
    final_results = [json.loads(path.read_text(encoding="utf-8")) for path in raw_dir.glob("*.json")]
    successful = sum(item.get("status") == "success" for item in final_results)
    failed = sum(item.get("status") == "failed" for item in final_results)
    ledger = guard.read_ledger()
    physical_requests = int(ledger["physical_request_counter"])
    retry_requests = sum(bool(item.get("explicit_retry")) for item in ledger["reservations"])
    audit = {
        "experiment_id": EXPERIMENT_ID, "logical_packages": MAX_LOGICAL_REQUESTS,
        "successful_packages": successful, "failed_packages": failed,
        "skipped_packages": skipped, "physical_requests": physical_requests,
        "retry_requests": retry_requests, "provider": environment["provider"], "model": environment["model"],
        "api_version": environment["api_version"], "started_at": started_at, "finished_at": _timestamp(),
        "budget_exceeded": physical_requests > MAX_LOGICAL_REQUESTS, "gt_leakage_check": leakage["status"],
        "experiment_run_uuid": guard.run_uuid, "request_ledger": guard.ledger_path.resolve().as_posix(),
    }
    (results_dir / "api_request_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dry-run", action="store_true", help="Build packages only (the default)")
    result.add_argument("--variant", action="append", choices=tuple(VARIANTS))
    result.add_argument("--case-limit", type=int)
    result.add_argument("--execute-api", action="store_true")
    result.add_argument("--confirm-cost", action="store_true")
    result.add_argument("--retry-failed", action="store_true", help="Explicit retry only; persistent physical budget still applies")
    result.add_argument("--recover-stale-lock", action="store_true", help="Recover only after the recorded PID is confirmed dead")
    result.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "analysis/vision_prompt_ab")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.execute_api and not args.confirm_cost:
        raise SystemExit("--execute-api requires --confirm-cost")
    variants = args.variant or list(VARIANTS)
    plan = build_packages(variants=variants, case_limit=args.case_limit, output_dir=args.output_dir)
    safe_preflight = safe_environment_preflight(load_environment=args.execute_api)
    print(json.dumps({**plan, "gt_leakage_check": candidate_leakage_audit()["status"], "api_preflight": safe_preflight}, ensure_ascii=False, indent=2))
    if args.execute_api:
        results_dir = args.output_dir / "results"
        audit_path = results_dir / "api_request_audit.json"
        prior_physical = int(json.loads(audit_path.read_text(encoding="utf-8")).get("physical_requests") or 0) if audit_path.exists() else 0
        guard = ExperimentRequestGuard(
            experiment_id=EXPERIMENT_ID, lock_path=results_dir / ".execution.lock",
            ledger_path=results_dir / "request_ledger.json", max_physical_requests=MAX_LOGICAL_REQUESTS,
            initial_physical_requests=prior_physical,
        )
        try:
            try:
                guard.acquire(recover_stale=args.recover_stale_lock)
            except ExperimentLockedError as exc:
                raise SystemExit(str(exc)) from exc
            audit = execute_packages(plan=plan, output_dir=args.output_dir, guard=guard, retry_failed=args.retry_failed)
            print(json.dumps(audit, ensure_ascii=False, indent=2))
        finally:
            guard.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
