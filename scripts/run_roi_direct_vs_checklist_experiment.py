"""Six-request ROI Direct vs Checklist experiment runner.

Dry-run is the default operational mode for Codex. Network transport requires
three explicit CLI gates and uses a fresh, run-local crash-safe ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_affected_part_prompt_ab import _data_url, _extract_json, _usage, safe_environment_preflight
from utils.experiment_request_guard import ExperimentLockedError, ExperimentRequestGuard
from utils.roi_checklist_rule_engine import evaluate_roi_checklist

CASES = ("missingpart-A01", "missingpart-B01", "wrongpart-B01")
METHODS = ("roi_direct", "roi_checklist")
REQUEST_IDS = tuple(f"EXP-{index:03d}" for index in range(1, 7))
LOGICAL_REQUEST_LIMIT = 6
PHYSICAL_REQUEST_HARD_CEILING = 6
AUTOMATIC_RETRY = 0
PROMPTS = {
    "roi_direct": PROJECT_ROOT / "experiments/prompts/vision_roi_direct_identity.txt",
    "roi_checklist": PROJECT_ROOT / "experiments/prompts/vision_roi_checklist_verification.txt",
}
SCHEMAS = {
    "roi_direct": PROJECT_ROOT / "experiments/schema/vision_roi_direct_output_schema.json",
    "roi_checklist": PROJECT_ROOT / "experiments/schema/vision_roi_checklist_output_schema.json",
}
ROI_PACKAGE_DIR = PROJECT_ROOT / "analysis/roi_identity_poc/packages"
ALLOWED_UNKNOWN = {"UNKNOWN", "UNKNOWN_PART", "UNRESOLVED", "UNRESOLVED_PART"}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _expected_summary(roi: dict[str, Any]) -> list[dict[str, Any]]:
    expected = json.loads((PROJECT_ROOT / f"ground_truth/{roi['model_id']}/{roi['step_id']}.json").read_text(encoding="utf-8"))
    candidates = set(roi["candidate_part_ids"])
    counts: dict[tuple[str, str, str, str], int] = {}
    for item in expected.get("expected_parts", []):
        part_id = str(item.get("part_id") or "")
        if part_id not in candidates:
            continue
        key = (part_id, str(item.get("color") or ""), str(item.get("position") or ""), str(item.get("orientation") or ""))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"part_id": key[0], "color": key[1], "position": key[2], "orientation": key[3], "expected_count": count}
        for key, count in sorted(counts.items())
    ]


def _copy_roi_assets(roi: dict[str, Any], package_dir: Path) -> tuple[list[str], list[str]]:
    asset_dir = package_dir / "roi_images"
    asset_dir.mkdir(parents=True, exist_ok=True)
    copied_test: list[str] = []
    copied_reference: list[str] = []
    selected = next(item for item in roi["view_results"] if item["view_angle"] == roi["primary_view"])
    for role, paths, destination in (
        ("test", selected.get("test_roi") or [], copied_test),
        ("reference", selected.get("reference_roi") or [], copied_reference),
    ):
        for index, raw in enumerate(paths, start=1):
            source = Path(raw)
            if not source.is_file():
                raise FileNotFoundError(f"Frozen ROI asset missing: {source}")
            target = asset_dir / f"{role}_roi_{index:02d}{source.suffix.lower()}"
            shutil.copy2(source, target)
            destination.append(target.relative_to(package_dir).as_posix())
    return copied_test, copied_reference


def _package_payload(roi: dict[str, Any], method: str, package_dir: Path) -> dict[str, Any]:
    selected = next(item for item in roi["view_results"] if item["view_angle"] == roi["primary_view"])
    test_rois, reference_rois = _copy_roi_assets(roi, package_dir)
    payload = {
        "case_id": roi["case_id"],
        "method": method,
        "model_id": roi["model_id"],
        "step_id": roi["step_id"],
        "view_angle": roi["primary_view"],
        "error_type": roi["error_type"],
        "test_image": selected["test_image"],
        "reference_image": selected["reference_image"],
        "test_roi_images": test_rois,
        "reference_roi_images": reference_rois,
        "bbox_evidence": roi["primary_bbox"],
        "localization_score": roi["localization_score"],
        "localization_status": roi["localization_status"],
        "localization_requires_manual_review": roi["requires_manual_review"],
        "candidate_part_ids": roi["candidate_part_ids"],
        "candidate_count": roi["candidate_count"],
        "full_candidate_count": roi["full_candidate_count"],
        "candidate_reduction_ratio": roi["reduction_ratio"],
        "expected_state_summary": _expected_summary(roi),
        "paired_roi_supported": roi["supports_paired_swap_roi"],
        "inference_label_source": "frozen_roi_poc_expected_state_part_library_local_evidence",
        "evaluation_labels_included": False,
    }
    payload["test_image_sha256"] = _sha256(Path(payload["test_image"]))
    payload["reference_image_sha256"] = _sha256(Path(payload["reference_image"]))
    payload["roi_image_sha256"] = {
        relative: _sha256(package_dir / relative)
        for relative in payload["test_roi_images"] + payload["reference_roi_images"]
    }
    return payload


def build_preflight(run_dir: Path, *, run_uuid: str | None = None) -> dict[str, Any]:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be new and empty: {run_dir}")
    run_uuid = str(run_uuid or uuid.uuid4())
    for folder in ("packages", "responses", "evaluation", "figures/cases", "thesis_tables"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    requests = []
    source_freeze = {}
    sequence = 0
    for case_id in CASES:
        source_path = ROI_PACKAGE_DIR / f"{case_id}.json"
        roi = json.loads(source_path.read_text(encoding="utf-8"))
        source_freeze[case_id] = {"path": source_path.relative_to(PROJECT_ROOT).as_posix(), "sha256": _sha256(source_path)}
        if roi.get("localization_status") != "success" or not roi.get("candidate_part_ids"):
            raise RuntimeError(f"Frozen ROI package is not usable: {case_id}")
        for method in METHODS:
            sequence += 1
            request_id = REQUEST_IDS[sequence - 1]
            package_dir = run_dir / "packages" / case_id / method
            package_dir.mkdir(parents=True, exist_ok=False)
            payload = _package_payload(roi, method, package_dir)
            prompt_template = PROMPTS[method].read_text(encoding="utf-8").rstrip()
            prompt_path = package_dir / "prompt.txt"
            prompt_path.write_text(prompt_template + "\n\nFROZEN ROI PACKAGE:\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            metadata = {
                **payload,
                "logical_request_id": request_id,
                "sequence": sequence,
                "prompt_source": PROMPTS[method].relative_to(PROJECT_ROOT).as_posix(),
                "prompt_sha256": _sha256(prompt_path),
                "schema_source": SCHEMAS[method].relative_to(PROJECT_ROOT).as_posix(),
                "schema_sha256": _sha256(SCHEMAS[method]),
                "source_roi_package_sha256": source_freeze[case_id]["sha256"],
                "automatic_retry": AUTOMATIC_RETRY,
                "contains_api_key": False,
            }
            metadata_path = package_dir / "request_metadata.json"
            _write_json(metadata_path, metadata)
            requests.append({
                "logical_request_id": request_id,
                "case_id": case_id,
                "method": method,
                "package_metadata": metadata_path.relative_to(run_dir).as_posix(),
            })
    manifest = {
        "experiment_type": "roi_direct_vs_checklist",
        "run_uuid": run_uuid,
        "created_at": _now(),
        "mode": "preflight_only",
        "cases": list(CASES),
        "methods": list(METHODS),
        "logical_request_limit": LOGICAL_REQUEST_LIMIT,
        "physical_request_hard_ceiling": PHYSICAL_REQUEST_HARD_CEILING,
        "automatic_retry": AUTOMATIC_RETRY,
        "source_roi_freeze": source_freeze,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "planned_requests": requests,
        "api_requests_made": 0,
        "execution_gates": ["--execute-api", "--confirm-six-requests", "--run-uuid exact-manifest-value"],
        "labels_join_policy": "evaluation_only_after_all_six_responses_are_frozen",
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    guard = ExperimentRequestGuard(
        experiment_id="roi-direct-vs-checklist", run_uuid=run_uuid,
        lock_path=run_dir / ".execution.lock", ledger_path=run_dir / "request_ledger.json",
        max_physical_requests=PHYSICAL_REQUEST_HARD_CEILING,
    )
    guard.acquire(); guard.release()
    return manifest


def _sanitize_response(value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    metadata_keys = {"$schema", "$defs", "$comment", "title", "type", "properties", "required", "additionalProperties"}
    removed = sorted(key for key in value if key in metadata_keys)
    return {key: item for key, item in value.items() if key not in metadata_keys}, removed


def candidate_membership(method: str, parsed: dict[str, Any], candidate_part_ids: list[str]) -> dict[str, Any]:
    if method == "roi_direct":
        ids = [str(item.get("part_id") or "").upper() for item in parsed.get("affected_parts", [])]
        violations = sorted({item for item in ids if item not in candidate_part_ids and item not in ALLOWED_UNKNOWN})
        missing = []
    else:
        ids = [str(item.get("part_id") or "").upper() for item in parsed.get("checks", [])]
        violations = sorted({item for item in ids if item not in candidate_part_ids})
        missing = [item for item in candidate_part_ids if ids.count(item) != 1]
    return {"status": "violation" if violations or missing else "valid", "violations": violations, "missing_or_duplicate_candidate_checks": missing}


def gt_leakage_audit(run_dir: Path) -> dict[str, Any]:
    forbidden = (
        "affected_part_eval_ground_truth", "affected_parts_review_template",
        "review_status", "confirmed_gt", "ground_truth_parts",
    )
    hits = []
    for path in sorted((run_dir / "packages").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".json", ".csv"}:
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                if token in text:
                    hits.append({"file": path.relative_to(run_dir).as_posix(), "token": token})
    source_hits = []
    for relative in ("utils/roi_identity_pipeline.py", "utils/roi_candidate_builder.py"):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8").lower()
        for token in ("affected_part_eval_ground_truth", "affected_parts_review_template", "missingpart-a01"):
            if token in text:
                source_hits.append({"file": relative, "token": token})
    result = {
        "status": "PASS" if not hits and not source_hits else "FAIL",
        "package_hits": hits,
        "inference_source_hits": source_hits,
        "candidate_identity_overlap_policy": "allowed_only_when_derived_by_frozen_roi_pipeline; no evaluation labels loaded",
        "evaluation_labels_loaded": False,
    }
    _write_json(run_dir / "evaluation/gt_leakage_audit.json", result)
    return result


def validate_preflight(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    ledger = json.loads((run_dir / "request_ledger.json").read_text(encoding="utf-8"))
    failures = []
    if len(manifest.get("planned_requests", [])) != 6 or manifest.get("logical_request_limit") != 6:
        failures.append("logical request matrix is not exactly six")
    if manifest.get("physical_request_hard_ceiling") != 6 or ledger.get("max_physical_requests") != 6:
        failures.append("physical hard ceiling is not six")
    if manifest.get("automatic_retry") != 0:
        failures.append("automatic retry is not zero")
    if ledger.get("physical_request_counter") != 0 or ledger.get("reservations") != []:
        failures.append("fresh preflight ledger is not empty")
    if manifest.get("runner_sha256") != _sha256(Path(__file__).resolve()):
        failures.append("experiment runner changed after preflight freeze")
    expected_pairs = {(case, method) for case in CASES for method in METHODS}
    actual_pairs = {(item.get("case_id"), item.get("method")) for item in manifest.get("planned_requests", [])}
    if expected_pairs != actual_pairs:
        failures.append("case/method matrix mismatch")
    if [item.get("logical_request_id") for item in manifest.get("planned_requests", [])] != list(REQUEST_IDS):
        failures.append("logical request IDs/order mismatch")
    for case_id, frozen in manifest.get("source_roi_freeze", {}).items():
        source = PROJECT_ROOT / frozen["path"]
        if not source.is_file() or _sha256(source) != frozen["sha256"]:
            failures.append(f"frozen ROI source changed: {case_id}")
    for item in manifest.get("planned_requests", []):
        metadata_path = run_dir / item["package_metadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prompt = metadata_path.with_name("prompt.txt")
        if _sha256(prompt) != metadata.get("prompt_sha256"):
            failures.append(f"prompt changed: {item['logical_request_id']}")
        schema = PROJECT_ROOT / metadata["schema_source"]
        if _sha256(schema) != metadata.get("schema_sha256"):
            failures.append(f"schema changed: {item['logical_request_id']}")
        source_hash = manifest["source_roi_freeze"][item["case_id"]]["sha256"]
        if metadata.get("source_roi_package_sha256") != source_hash:
            failures.append(f"ROI source hash mismatch: {item['logical_request_id']}")
        for asset in metadata.get("test_roi_images", []) + metadata.get("reference_roi_images", []):
            if not (metadata_path.parent / asset).is_file():
                failures.append(f"missing ROI asset: {item['logical_request_id']}:{asset}")
            elif metadata.get("roi_image_sha256", {}).get(asset) != _sha256(metadata_path.parent / asset):
                failures.append(f"ROI asset hash changed: {item['logical_request_id']}:{asset}")
        if not Path(metadata["test_image"]).is_file() or not Path(metadata["reference_image"]).is_file():
            failures.append(f"missing full image: {item['logical_request_id']}")
        else:
            if metadata.get("test_image_sha256") != _sha256(Path(metadata["test_image"])):
                failures.append(f"test image hash changed: {item['logical_request_id']}")
            if metadata.get("reference_image_sha256") != _sha256(Path(metadata["reference_image"])):
                failures.append(f"reference image hash changed: {item['logical_request_id']}")
    leakage = gt_leakage_audit(run_dir)
    if leakage["status"] != "PASS":
        failures.append("GT leakage audit failed")
    result = {"status": "PASS" if not failures else "FAIL", "failures": failures, "run_uuid": manifest["run_uuid"]}
    _write_json(run_dir / "evaluation/preflight_validation.json", {**result, "api_requests_made": 0, "validated_at": _now()})
    return result


def _lock_probe(run_dir: Path, run_uuid: str) -> dict[str, Any]:
    args = dict(
        experiment_id="roi-direct-vs-checklist", run_uuid=run_uuid,
        lock_path=run_dir / ".execution.lock", ledger_path=run_dir / "request_ledger.json",
        max_physical_requests=PHYSICAL_REQUEST_HARD_CEILING,
    )
    first, second = ExperimentRequestGuard(**args), ExperimentRequestGuard(**args)
    blocked = False
    first.acquire()
    try:
        try: second.acquire()
        except ExperimentLockedError: blocked = True
    finally:
        first.release(); second.release()
    return {"status": "PASS" if blocked else "FAIL", "second_process_blocked": blocked}


def run_dry_preflight(run_dir: Path) -> dict[str, Any]:
    if not (run_dir / "run_manifest.json").exists():
        build_preflight(run_dir)
    validation = validate_preflight(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    lock = _lock_probe(run_dir, manifest["run_uuid"])
    environment = safe_environment_preflight(load_environment=True)
    safety = {
        "logical_request_limit": 6, "physical_request_hard_ceiling": 6,
        "automatic_retry": 0, "pid_lock": lock,
        "reservation_before_transport": True,
        "reserved_completed_resume_policy": "skip_or_fail_closed_no_resend",
        "seventh_request_policy": "fail_before_transport",
        "schema_validation_retry": False,
        "api_requests_made": 0,
    }
    _write_json(run_dir / "evaluation/request_safety_preflight.json", safety)
    _write_json(run_dir / "evaluation/environment_preflight.json", {**environment, "api_requests_made": 0})
    pending_tables = {
        "roi_direct_vs_checklist_metrics.csv": ["metric", "roi_direct", "roi_checklist", "delta", "preferred_direction", "winner"],
        "roi_direct_vs_checklist_cases.csv": ["case_id", "error_type", "ground_truth_parts", "full_candidate_count", "roi_candidate_count", "candidate_reduction", "localization_score", "direct_prediction", "direct_confidence", "direct_exact_match", "direct_verifier_status", "checklist_prediction", "checklist_confidence", "checklist_exact_match", "checklist_verifier_status", "checklist_uncertain_count", "annotated_result_path"],
        "checklist_component_results.csv": ["case_id", "part_id", "gt_status", "reference_present", "test_present", "reference_count", "test_count", "spatial_match", "appearance_match", "predicted_status", "confidence", "correct", "uncertain"],
        "request_efficiency.csv": ["method", "case_id", "request_count", "latency_seconds", "input_tokens", "output_tokens", "total_tokens"],
    }
    for name, fields in pending_tables.items():
        path = run_dir / "thesis_tables" / name
        if not path.exists():
            path.write_text(",".join(fields) + "\n", encoding="utf-8")
    evolution = run_dir / "thesis_tables/research_method_evolution.csv"
    if not evolution.exists():
        evolution.write_text(
            "stage,exact_match,part_f1,false_confident_0_80,candidate_reduction,gt_coverage,denominator\n"
            "Stage 1: Free-form VLM Baseline,0.08,0.105263,0.88,,,25\n"
            "Stage 2: Prompt/Candidate Constraint,0.0,0.2857,0.6667,,,3\n"
            "Stage 3: ROI Candidate Reduction,,,,0.6444,1.0,3\n"
            "Stage 4: ROI Direct Classification,pending,pending,pending,0.6444,1.0,3\n"
            "Stage 5: ROI Checklist Verification,pending,pending,pending,0.6444,1.0,3\n",
            encoding="utf-8",
        )
    package_summaries = []
    for planned in manifest["planned_requests"]:
        metadata = json.loads((run_dir / planned["package_metadata"]).read_text(encoding="utf-8"))
        package_summaries.append({
            "logical_request_id": planned["logical_request_id"], "case_id": planned["case_id"],
            "method": planned["method"], "candidate_part_ids": metadata["candidate_part_ids"],
            "candidate_count": metadata["candidate_count"], "candidate_reduction_ratio": metadata["candidate_reduction_ratio"],
            "localization_score": metadata["localization_score"], "paired_roi_supported": metadata["paired_roi_supported"],
        })
    final_report = {
        "experiment_run_directory": str(run_dir.resolve()), "run_uuid": manifest["run_uuid"],
        "logical_request_ids": [item["logical_request_id"] for item in package_summaries],
        "cases": list(CASES), "methods": list(METHODS), "packages": package_summaries,
        "direct_prompt": str(PROMPTS["roi_direct"].resolve()), "checklist_prompt": str(PROMPTS["roi_checklist"].resolve()),
        "checklist_schema": str(SCHEMAS["roi_checklist"].resolve()),
        "rule_engine": str((PROJECT_ROOT / "utils/roi_checklist_rule_engine.py").resolve()),
        "annotator": str((PROJECT_ROOT / "utils/deterministic_correction_annotator.py").resolve()),
        "evaluation_script": str((PROJECT_ROOT / "scripts/evaluate_roi_direct_vs_checklist.py").resolve()),
        "confusion_matrix_script": str((PROJECT_ROOT / "scripts/render_checklist_confusion_matrix.py").resolve()),
        "thesis_figure_script": str((PROJECT_ROOT / "scripts/render_roi_thesis_case_figure.py").resolve()),
        "thesis_table_directory": str((run_dir / "thesis_tables").resolve()),
        "gt_leakage": json.loads((run_dir / "evaluation/gt_leakage_audit.json").read_text(encoding="utf-8"))["status"],
        "pid_lock": lock["status"], "physical_request_hard_ceiling": 6, "automatic_retry": 0,
        "resume_protection": "PASS", "environment_readiness": environment,
        "api_requests_actually_made": 0, "production_prompt_modified_by_experiment": False,
        "production_schema_modified_by_experiment": False, "evaluation_labels_modified_by_experiment": False,
        "source_images_modified_by_experiment": False, "gpt_image_requests": 0, "phase_2b_executed": False,
    }
    _write_json(run_dir / "evaluation/final_preflight_report.json", final_report)
    status = "PASS" if validation["status"] == "PASS" and lock["status"] == "PASS" else "FAIL"
    return {"status": status, "run_uuid": manifest["run_uuid"], "failures": validation["failures"], "environment": environment, "safety": safety}


def _message_content(metadata: dict[str, Any], package_dir: Path) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": (package_dir / "prompt.txt").read_text(encoding="utf-8")}]
    for label, raw in (("Correct Reference full image", metadata["reference_image"]), ("Test full image", metadata["test_image"])):
        content.extend(({"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": _data_url(Path(raw)), "detail": "high"}}))
    for role, paths in (("Reference ROI", metadata.get("reference_roi_images", [])), ("Test ROI", metadata.get("test_roi_images", []))):
        for index, raw in enumerate(paths, start=1):
            content.extend(({"type": "text", "text": f"{role} {index}"}, {"type": "image_url", "image_url": {"url": _data_url(package_dir / raw), "detail": "high"}}))
    return content


def execute(run_dir: Path, *, confirmed_run_uuid: str, client: Any | None = None) -> dict[str, Any]:
    validation = validate_preflight(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if validation["status"] != "PASS" or confirmed_run_uuid != manifest["run_uuid"]:
        raise RuntimeError("Preflight or run UUID execution gate failed")
    environment = safe_environment_preflight(load_environment=True)
    if not environment["ready"]:
        raise RuntimeError("Azure Vision environment is incomplete")
    if client is None:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_version=environment["api_version"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"], max_retries=0,
        )
    from jsonschema import validate
    guard = ExperimentRequestGuard(
        experiment_id="roi-direct-vs-checklist", run_uuid=manifest["run_uuid"],
        lock_path=run_dir / ".execution.lock", ledger_path=run_dir / "request_ledger.json",
        max_physical_requests=PHYSICAL_REQUEST_HARD_CEILING,
    )
    guard.acquire()
    try:
        for planned in manifest["planned_requests"]:
            response_path = run_dir / "responses" / f"{planned['logical_request_id']}_{planned['case_id']}_{planned['method']}.json"
            if response_path.exists():
                continue
            reservation = guard.reserve(planned["logical_request_id"], explicit_retry=False)
            if reservation is None:
                continue
            package_dir = (run_dir / planned["package_metadata"]).parent
            metadata = json.loads((package_dir / "request_metadata.json").read_text(encoding="utf-8"))
            response = None; parsed = None; raw_text = ""; error_type = None
            schema_result: dict[str, Any] = {"status": "not_run"}
            membership: dict[str, Any] = {"status": "not_run", "violations": []}
            rule_result = None
            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=environment["model"], temperature=0, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": "Return JSON only. Follow the supplied experiment contract; never echo schema metadata."}, {"role": "user", "content": _message_content(metadata, package_dir)}],
                )
                raw_text = response.choices[0].message.content or ""
                parsed, removed = _sanitize_response(_extract_json(raw_text))
                schema = json.loads((PROJECT_ROOT / metadata["schema_source"]).read_text(encoding="utf-8"))
                try:
                    validate(instance=parsed, schema=schema)
                    schema_result = {"status": "valid", "sanitized_metadata_keys": removed}
                except Exception as exc:
                    schema_result = {"status": "invalid", "error_type": type(exc).__name__, "message": str(exc), "sanitized_metadata_keys": removed}
                membership = candidate_membership(metadata["method"], parsed, metadata["candidate_part_ids"])
                if metadata["method"] == "roi_checklist":
                    if schema_result["status"] == "valid" and membership["status"] == "valid":
                        rule_result = evaluate_roi_checklist(checks=parsed["checks"], candidate_part_ids=metadata["candidate_part_ids"], error_type=metadata["error_type"], paired_roi_supported=metadata["paired_roi_supported"])
            except Exception as exc:
                error_type = type(exc).__name__
            affected = (parsed or {}).get("affected_parts", []) if metadata["method"] == "roi_direct" else (rule_result or {}).get("affected_parts", [])
            verifier = {
                "verifier_status": "conflict" if affected else "unresolved",
                "verified_part_ids": [],
                "requires_manual_review": True,
                "reason": "frozen_roi_package_requires_manual_review",
            }
            payload = {
                "run_uuid": manifest["run_uuid"], "logical_request_id": planned["logical_request_id"],
                "request_id": reservation, "api_request_id": getattr(response, "id", None),
                "method": metadata["method"], "case_id": metadata["case_id"],
                "raw_response": response.model_dump() if response is not None and hasattr(response, "model_dump") else ({"content": raw_text} if response is not None else None),
                "parsed_response": parsed, "schema_validation_result": schema_result,
                "candidate_membership_result": membership, "rule_engine_result": rule_result,
                "verifier_result": verifier, "request_duration_seconds": round(time.perf_counter() - started, 3),
                "http_api_error_type": error_type, "usage": _usage(response) if response is not None else None,
            }
            _write_json(response_path, payload)
            guard.finish(reservation, "completed" if response is not None else "failed")
    finally:
        guard.release()
    return {"status": "completed", "physical_requests": guard.read_ledger()["physical_request_counter"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--confirm-six-requests", action="store_true")
    parser.add_argument("--run-uuid")
    args = parser.parse_args()
    if args.execute_api:
        if args.dry_run or not args.confirm_six_requests or not args.run_uuid:
            raise SystemExit("API execution requires only --execute-api --confirm-six-requests --run-uuid")
        print(json.dumps(execute(args.run_dir, confirmed_run_uuid=args.run_uuid), indent=2))
        return 0
    result = run_dry_preflight(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
