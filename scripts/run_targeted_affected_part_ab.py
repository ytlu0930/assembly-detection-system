"""Fixed six-request targeted affected-part A/B runner; defaults to offline pre-flight."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_affected_part_prompt_ab import (
    CURRENT_SCHEMA,
    PART_LIBRARY,
    VARIANTS,
    _case_context,
    _data_url,
    _extract_json,
    _prediction,
    _sha256,
    _timestamp,
    _usage,
    candidate_leakage_audit,
    enforce_candidate_constraint,
    safe_environment_preflight,
)
from utils.affected_part_candidate_builder import build_affected_part_candidates
from utils.experiment_request_guard import ExperimentLockedError, ExperimentRequestGuard

TARGET_CASES = (
    ("missingpart-A01", "input/missingpart/model03_step03/model03_step03_missingpart-A01_front_01.jpg", "missingpart"),
    ("missingpart-B01", "input/missingpart/model03_step03/model03_step03_missingpart-B01_front_01.jpg", "missingpart"),
    ("wrongpart-B01", "input/wrongpart/model03_step03/model03_step03_wrongpart-B01_front_01.jpg", "wrongpart"),
)
TARGET_VARIANTS = ("reference", "reference_candidate")
LOGICAL_REQUEST_LIMIT = 6
PHYSICAL_REQUEST_HARD_CEILING = 6
AUTOMATIC_RETRY = 0


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_targeted_preflight(run_dir: Path, *, run_uuid: str | None = None) -> dict[str, Any]:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Targeted run directory is not empty: {run_dir}")
    run_uuid = str(run_uuid or uuid.uuid4())
    for name in ("requests", "responses", "evaluation"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    schema_sha = _sha256(CURRENT_SCHEMA)
    planned = []
    sequence = 0
    for case in TARGET_CASES:
        context = _case_context(case)
        expected_path = PROJECT_ROOT / context["expected_state"]
        expected_payload = json.loads(expected_path.read_text(encoding="utf-8"))
        for variant in TARGET_VARIANTS:
            sequence += 1
            logical_id = f"TGT-{sequence:03d}:{context['case_id']}:{variant}"
            candidate_payload = None
            if variant == "reference_candidate":
                candidate_payload = build_affected_part_candidates(
                    model_id=context["model_id"], step_id=context["step_id"],
                    expected_state=expected_payload, part_library=PART_LIBRARY,
                    error_type=context["error_type_hint"],
                )
            sections = [
                VARIANTS[variant].read_text(encoding="utf-8").rstrip(),
                "\nCase context:\n" + json.dumps(context, ensure_ascii=False, indent=2),
                "\nexpected_state JSON:\n" + json.dumps(expected_payload, ensure_ascii=False, indent=2),
            ]
            if candidate_payload is not None:
                sections.append("\nCandidate affected parts:\n" + json.dumps(candidate_payload, ensure_ascii=False, indent=2))
            package_dir = run_dir / "requests" / f"{sequence:02d}_{context['case_id']}_{variant}"
            package_dir.mkdir(parents=True, exist_ok=False)
            prompt_path = package_dir / "prompt.txt"
            prompt_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
            metadata = {
                **context,
                "logical_request_id": logical_id,
                "sequence": sequence,
                "variant": variant,
                "prompt_source": VARIANTS[variant].relative_to(PROJECT_ROOT).as_posix(),
                "prompt_sha256": _sha256(prompt_path),
                "schema": CURRENT_SCHEMA.relative_to(PROJECT_ROOT).as_posix(),
                "schema_sha256": schema_sha,
                "schema_condition": "current",
                "candidate_part_ids": candidate_payload["candidate_part_ids"] if candidate_payload else [],
                "candidate_metadata": candidate_payload["candidate_metadata"] if candidate_payload else None,
                "contains_api_key": False,
                "automatic_retry": AUTOMATIC_RETRY,
            }
            metadata_path = package_dir / "request_metadata.json"
            _write_json(metadata_path, metadata)
            planned.append({**metadata, "package_path": metadata_path.relative_to(run_dir).as_posix()})
    if len(planned) != LOGICAL_REQUEST_LIMIT:
        raise RuntimeError("Targeted package count is not exactly six")
    leakage = candidate_leakage_audit()
    if leakage["status"] != "PASS":
        raise RuntimeError(f"Candidate leakage audit failed: {leakage['forbidden_hits']}")
    manifest = {
        "experiment_type": "targeted_affected_part_ab",
        "run_uuid": run_uuid,
        "created_at": _timestamp(),
        "mode": "preflight_only",
        "logical_request_limit": LOGICAL_REQUEST_LIMIT,
        "physical_request_hard_ceiling": PHYSICAL_REQUEST_HARD_CEILING,
        "automatic_retry": AUTOMATIC_RETRY,
        "schema": CURRENT_SCHEMA.relative_to(PROJECT_ROOT).as_posix(),
        "schema_sha256": schema_sha,
        "candidate_source": "runtime expected_state+part_library",
        "human_review_source_used": False,
        "planned_requests": planned,
        "api_requests_made": 0,
        "execution_gates": ["--execute-api", "--confirm-six-requests", "--run-uuid exact-manifest-value"],
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    guard = ExperimentRequestGuard(
        experiment_id="targeted-affected-part-ab", run_uuid=run_uuid,
        lock_path=run_dir / ".execution.lock", ledger_path=run_dir / "request_ledger.json",
        max_physical_requests=PHYSICAL_REQUEST_HARD_CEILING,
    )
    guard.acquire()
    guard.release()
    return manifest


def validate_preflight(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    ledger = json.loads((run_dir / "request_ledger.json").read_text(encoding="utf-8"))
    planned = manifest.get("planned_requests") or []
    failures = []
    if len(planned) != 6 or manifest.get("logical_request_limit") != 6:
        failures.append("logical request count/limit is not six")
    if manifest.get("physical_request_hard_ceiling") != 6 or ledger.get("max_physical_requests") != 6:
        failures.append("physical hard ceiling is not six")
    if manifest.get("automatic_retry") != 0:
        failures.append("automatic retry is not zero")
    if ledger.get("physical_request_counter") != 0 or ledger.get("reservations") != []:
        failures.append("new ledger is not empty")
    if ledger.get("experiment_id") != "targeted-affected-part-ab" or ledger.get("run_uuid") != manifest.get("run_uuid"):
        failures.append("ledger experiment identity/run UUID is invalid")
    expected_pairs = {(case[0], variant) for case in TARGET_CASES for variant in TARGET_VARIANTS}
    expected_contexts = {case[0]: _case_context(case) for case in TARGET_CASES}
    actual_pairs = {(item.get("case_id"), item.get("variant")) for item in planned}
    if actual_pairs != expected_pairs:
        failures.append("case/variant matrix mismatch")
    for item in planned:
        metadata_path = run_dir / item["package_path"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prompt_path = metadata_path.with_name("prompt.txt")
        expected_context = expected_contexts.get(metadata.get("case_id"), {})
        for context_key in ("test_image", "reference_image", "expected_state", "error_type_hint"):
            if metadata.get(context_key) != expected_context.get(context_key):
                failures.append(f"{metadata['logical_request_id']} {context_key} mismatch")
        if metadata.get("prompt_source") != VARIANTS[metadata["variant"]].relative_to(PROJECT_ROOT).as_posix():
            failures.append(f"{metadata['logical_request_id']} prompt source mismatch")
        for asset_key in ("test_image", "reference_image", "expected_state"):
            if not (PROJECT_ROOT / metadata[asset_key]).is_file():
                failures.append(f"{metadata['logical_request_id']} missing {asset_key}")
        if metadata["schema"] != CURRENT_SCHEMA.relative_to(PROJECT_ROOT).as_posix() or metadata["schema_sha256"] != _sha256(CURRENT_SCHEMA):
            failures.append(f"{metadata['logical_request_id']} schema mismatch")
        if metadata["prompt_sha256"] != _sha256(prompt_path):
            failures.append(f"{metadata['logical_request_id']} prompt hash mismatch")
        if metadata["variant"] == "reference_candidate":
            candidate = build_affected_part_candidates(
                model_id=metadata["model_id"], step_id=metadata["step_id"],
                expected_state=PROJECT_ROOT / metadata["expected_state"], part_library=PART_LIBRARY,
                error_type=metadata["error_type_hint"],
            )
            if metadata["candidate_part_ids"] != candidate["candidate_part_ids"]:
                failures.append(f"{metadata['logical_request_id']} candidate set is not runtime reproducible")
            if metadata.get("candidate_metadata", {}).get("human_review_source_used") is not False:
                failures.append(f"{metadata['logical_request_id']} candidate provenance is unsafe")
        elif metadata["candidate_part_ids"]:
            failures.append(f"{metadata['logical_request_id']} Reference unexpectedly has candidates")
    result = {"status": "PASS" if not failures else "FAIL", "failures": failures, "manifest": manifest}
    _write_json(run_dir / "evaluation" / "preflight_validation.json", {
        "status": result["status"], "failures": failures, "validated_at": _timestamp(),
        "run_uuid": manifest.get("run_uuid"), "api_requests_made": 0,
    })
    return result


def verify_exclusive_lock(run_dir: Path, run_uuid: str) -> dict[str, Any]:
    first = ExperimentRequestGuard(
        experiment_id="targeted-affected-part-ab", run_uuid=run_uuid,
        lock_path=run_dir / ".execution.lock", ledger_path=run_dir / "request_ledger.json",
        max_physical_requests=PHYSICAL_REQUEST_HARD_CEILING,
    )
    second = ExperimentRequestGuard(
        experiment_id="targeted-affected-part-ab", run_uuid=run_uuid,
        lock_path=run_dir / ".execution.lock", ledger_path=run_dir / "request_ledger.json",
        max_physical_requests=PHYSICAL_REQUEST_HARD_CEILING,
    )
    blocked = False
    first.acquire()
    try:
        try:
            second.acquire()
        except ExperimentLockedError:
            blocked = True
    finally:
        first.release()
        second.release()
    return {"status": "PASS" if blocked else "FAIL", "second_process_blocked": blocked}


def execute_targeted(
    run_dir: Path, *, confirmed_run_uuid: str,
    client: Any | None = None, offline_verifier: Any | None = None,
) -> dict[str, Any]:
    validation = validate_preflight(run_dir)
    if validation["status"] != "PASS":
        raise RuntimeError(f"Pre-flight validation failed: {validation['failures']}")
    manifest = validation["manifest"]
    if confirmed_run_uuid != manifest["run_uuid"]:
        raise RuntimeError("Run UUID execution gate mismatch")
    environment = safe_environment_preflight(load_environment=True)
    if not environment["ready"]:
        raise RuntimeError("Azure Vision configuration is incomplete")
    from jsonschema import validate
    if client is None:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_version=environment["api_version"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
        )
    schema = json.loads(CURRENT_SCHEMA.read_text(encoding="utf-8"))
    from scripts.finalize_affected_part_prompt_ab import OfflineProductionVerifier, quarantine_candidate_violation
    if offline_verifier is None:
        offline_verifier = OfflineProductionVerifier(run_dir / "evaluation" / "verifier_localization")
    guard = ExperimentRequestGuard(
        experiment_id="targeted-affected-part-ab", run_uuid=manifest["run_uuid"],
        lock_path=run_dir / ".execution.lock", ledger_path=run_dir / "request_ledger.json",
        max_physical_requests=PHYSICAL_REQUEST_HARD_CEILING,
    )
    try:
        guard.acquire()
        for metadata in manifest["planned_requests"]:
            logical_id = metadata["logical_request_id"]
            response_path = run_dir / "responses" / f"{metadata['sequence']:02d}_{metadata['case_id']}_{metadata['variant']}.json"
            if response_path.exists():
                continue
            internal_request_id = guard.reserve(logical_id, explicit_retry=False)
            if internal_request_id is None:
                continue
            tick = time.perf_counter()
            response = None
            raw_text = ""
            parsed = None
            schema_result: dict[str, Any] = {"status": "not_run", "error_type": None, "message": None}
            membership = {"candidate_constraint_status": "not_run", "candidate_constraint_violations": []}
            verifier_result: dict[str, Any] = {"status": "not_run", "reason": "schema_not_valid"}
            api_error_type = None
            try:
                package = run_dir / metadata["package_path"]
                prompt = package.with_name("prompt.txt").read_text(encoding="utf-8")
                response = client.chat.completions.create(
                    model=environment["model"], temperature=0, response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are a strict JSON-only reference-guided vision inspector. Return a JSON object conforming exactly to this current schema: " + json.dumps(schema, ensure_ascii=False)},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "Correct Reference Image:"},
                            {"type": "image_url", "image_url": {"url": _data_url(PROJECT_ROOT / metadata["reference_image"]), "detail": "high"}},
                            {"type": "text", "text": "Test Image:"},
                            {"type": "image_url", "image_url": {"url": _data_url(PROJECT_ROOT / metadata["test_image"]), "detail": "high"}},
                        ]},
                    ],
                )
                raw_text = response.choices[0].message.content or ""
                parsed = _extract_json(raw_text)
                try:
                    validate(instance=parsed, schema=schema)
                    schema_result = {"status": "valid", "error_type": None, "message": None}
                except Exception as exc:
                    schema_result = {"status": "invalid", "error_type": type(exc).__name__, "message": str(exc)}
                part_ids = _prediction(parsed)[0]
                membership = enforce_candidate_constraint(metadata["variant"], metadata["candidate_part_ids"], part_ids)
                if schema_result["status"] == "valid":
                    verifier_input = {
                        **metadata,
                        "image_id": Path(metadata["test_image"]).name,
                        "parsed_output": parsed,
                    }
                    try:
                        verifier_result = offline_verifier.verify(verifier_input)
                        verifier_result = quarantine_candidate_violation(
                            verifier_result, membership["candidate_constraint_status"]
                        )
                    except Exception as exc:
                        verifier_result = {
                            "status": "error", "error_type": type(exc).__name__,
                            "requires_manual_review": True,
                        }
            except Exception as exc:
                api_error_type = type(exc).__name__
                secret = os.getenv("AZURE_OPENAI_API_KEY", "")
                error_message = str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
                if parsed is None:
                    schema_result = {"status": "not_run", "error_type": type(exc).__name__, "message": error_message}
            payload = {
                "run_uuid": manifest["run_uuid"], "logical_request_id": logical_id,
                "request_id": internal_request_id, "api_request_id": getattr(response, "id", None),
                "raw_response": response.model_dump() if response is not None and hasattr(response, "model_dump") else ({"content": raw_text} if response is not None else None),
                "parsed_response": parsed, "schema_validation_result": schema_result,
                "candidate_membership_result": membership, "verifier_result": verifier_result,
                "request_duration_seconds": round(time.perf_counter() - tick, 3),
                "http_api_error_type": api_error_type, "usage": _usage(response) if response is not None else None,
            }
            _write_json(response_path, payload)
            terminal = "completed" if response is not None else "failed"
            guard.finish(internal_request_id, terminal)
    finally:
        guard.release()
    return {"status": "completed", "api_requests_made": guard.read_ledger()["physical_request_counter"]}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--initialize-preflight", action="store_true")
    result.add_argument("--execute-api", action="store_true")
    result.add_argument("--confirm-six-requests", action="store_true")
    result.add_argument("--run-uuid")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.initialize_preflight and args.execute_api:
        raise SystemExit("Initialization and API execution must be separate invocations")
    if args.initialize_preflight:
        manifest = build_targeted_preflight(args.run_dir, run_uuid=args.run_uuid)
        print(json.dumps({"status": "PRE_FLIGHT_CREATED", "run_uuid": manifest["run_uuid"], "api_requests_made": 0}, indent=2))
        return 0
    validation = validate_preflight(args.run_dir)
    if not args.execute_api:
        environment = safe_environment_preflight(load_environment=True)
        lock_check = verify_exclusive_lock(args.run_dir, validation["manifest"]["run_uuid"])
        _write_json(args.run_dir / "evaluation" / "environment_preflight.json", {
            **environment,
            "execution_gates": validation["manifest"]["execution_gates"],
            "api_requests_made": 0,
        })
        _write_json(args.run_dir / "evaluation" / "request_safety_preflight.json", {
            "logical_request_limit": LOGICAL_REQUEST_LIMIT,
            "physical_request_hard_ceiling": PHYSICAL_REQUEST_HARD_CEILING,
            "automatic_retry": AUTOMATIC_RETRY,
            "pid_lock": lock_check,
            "reservation_before_transport": True,
            "reserved_or_completed_resume_policy": "fail_closed_no_resend",
            "seventh_request_policy": "hard_ceiling_fail_before_transport",
            "schema_validation_retry": False,
            "api_requests_made": 0,
        })
        status = "PASS" if validation["status"] == "PASS" and lock_check["status"] == "PASS" else "FAIL"
        print(json.dumps({
            "status": status, "failures": validation["failures"],
            "pid_lock": lock_check["status"], "environment_ready": environment["ready"],
            "api_requests_made": 0,
        }, indent=2))
        return 0 if status == "PASS" else 1
    if not args.confirm_six_requests or not args.run_uuid:
        raise SystemExit("API execution requires --confirm-six-requests and --run-uuid")
    try:
        result = execute_targeted(args.run_dir, confirmed_run_uuid=args.run_uuid)
    except ExperimentLockedError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
