"""Freeze and audit the six ROI experiment responses without loading labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def freeze_and_audit(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    ledger = json.loads((run_dir / "request_ledger.json").read_text(encoding="utf-8"))
    planned = {item["logical_request_id"]: item for item in manifest["planned_requests"]}
    reservations = {item["logical_request_id"]: item for item in ledger["reservations"]}
    response_paths = sorted((run_dir / "responses").glob("*.json"))
    failures: list[str] = []
    if len(planned) != 6: failures.append("planned logical request count is not six")
    if ledger.get("physical_request_counter") != 6: failures.append("physical request counter is not six")
    if len(reservations) != 6: failures.append("reservation count is not six")
    if any(item.get("explicit_retry") for item in ledger.get("reservations", [])): failures.append("retry reservation found")
    if any(item.get("status") != "completed" for item in ledger.get("reservations", [])): failures.append("non-completed reservation found")
    frozen_dir = run_dir / "evaluation/frozen_responses"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for source in response_paths:
        response = json.loads(source.read_text(encoding="utf-8"))
        logical_id = response.get("logical_request_id")
        if logical_id not in planned: failures.append(f"unplanned response: {source.name}"); continue
        if logical_id in seen: failures.append(f"duplicate response: {logical_id}"); continue
        seen.add(logical_id)
        plan, reservation = planned[logical_id], reservations.get(logical_id, {})
        metadata_path = run_dir / plan["package_metadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frozen_path = frozen_dir / source.name
        if frozen_path.exists() and _sha256(frozen_path) != _sha256(source):
            failures.append(f"existing frozen response differs: {logical_id}")
        elif not frozen_path.exists():
            shutil.copy2(source, frozen_path)
        row = {
            "logical_request_id": logical_id, "case_id": response.get("case_id"), "method": response.get("method"),
            "reserved": bool(reservation), "sent": bool(response.get("api_request_id") or response.get("raw_response")),
            "response_received": response.get("raw_response") is not None,
            "raw_response_saved": response.get("raw_response") is not None,
            "parsed_response_saved": response.get("parsed_response") is not None,
            "schema_validation": (response.get("schema_validation_result") or {}).get("status"),
            "candidate_membership_validation": (response.get("candidate_membership_result") or {}).get("status"),
            "rule_engine_result_saved": response.get("rule_engine_result") is not None,
            "verifier_result_saved": response.get("verifier_result") is not None,
            "ledger_status": reservation.get("status"), "explicit_retry": bool(reservation.get("explicit_retry")),
            "request_id": response.get("request_id"), "api_response_id": response.get("api_request_id"),
            "http_api_error_type": response.get("http_api_error_type"), "latency_seconds": response.get("request_duration_seconds"),
            "candidate_ids": metadata.get("candidate_part_ids"), "localization_score": metadata.get("localization_score"),
            "raw_response_path": source.resolve().as_posix(), "frozen_response_path": frozen_path.resolve().as_posix(),
            "response_sha256": _sha256(source), "frozen_sha256": _sha256(frozen_path),
            "parsed_response": response.get("parsed_response"),
        }
        rows.append(row)
    if seen != set(planned): failures.append("response logical IDs do not equal planned IDs")
    if any(row["response_sha256"] != row["frozen_sha256"] for row in rows): failures.append("frozen hash mismatch")
    audit = {
        "status": "PASS" if not failures else "FAIL", "failures": failures,
        "run_uuid": manifest["run_uuid"], "logical_requests": len(planned),
        "physical_requests": ledger.get("physical_request_counter"),
        "retry_requests": sum(bool(item.get("explicit_retry")) for item in ledger.get("reservations", [])),
        "successful_requests": sum(row["response_received"] and not row["http_api_error_type"] for row in rows),
        "schema_valid_requests": sum(row["schema_validation"] == "valid" for row in rows),
        "requests": rows, "api_requests_during_audit": 0,
        "labels_loaded": False,
    }
    _write_json(run_dir / "evaluation/request_audit_summary.json", audit)
    _write_json(frozen_dir / "frozen_manifest.json", {
        "run_uuid": manifest["run_uuid"], "snapshot_status": audit["status"],
        "response_count": len(rows), "responses": rows,
        "labels_loaded": False, "api_requests_during_freeze": 0,
    })
    if failures:
        raise RuntimeError("; ".join(failures))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = freeze_and_audit(args.run_dir)
    print(json.dumps({key: audit[key] for key in ("status", "logical_requests", "physical_requests", "retry_requests", "successful_requests", "schema_valid_requests", "api_requests_during_audit")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
