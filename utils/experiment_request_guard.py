"""Crash-safe process lock and persistent physical-request ledger."""

from __future__ import annotations

import atexit
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class ExperimentLockedError(RuntimeError): pass
class PhysicalBudgetExhaustedError(RuntimeError): pass
class ExplicitRetryRequiredError(RuntimeError): pass


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class ExperimentRequestGuard:
    def __init__(
        self, *, experiment_id: str, lock_path: Path, ledger_path: Path,
        max_physical_requests: int, initial_physical_requests: int = 0,
        process_id: int | None = None, pid_checker: Callable[[int], bool] = pid_is_alive,
        run_uuid: str | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.lock_path = Path(lock_path)
        self.ledger_path = Path(ledger_path)
        self.max_physical_requests = int(max_physical_requests)
        self.initial_physical_requests = int(initial_physical_requests)
        self.process_id = int(process_id or os.getpid())
        self.pid_checker = pid_checker
        self.run_uuid = str(run_uuid or uuid.uuid4())
        self._lock_fd: int | None = None

    def acquire(self, *, recover_stale: bool = False) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            try:
                existing = json.loads(self.lock_path.read_text(encoding="utf-8"))
                existing_pid = int(existing.get("pid") or 0)
            except Exception:
                existing_pid = 0
            if self.pid_checker(existing_pid):
                raise ExperimentLockedError(f"Experiment lock is owned by live PID {existing_pid}")
            if not recover_stale:
                raise ExperimentLockedError("Stale experiment lock requires explicit recovery after PID validation")
            self.lock_path.unlink()
        payload = {"experiment_id": self.experiment_id, "run_uuid": self.run_uuid, "pid": self.process_id, "created_at": _now()}
        try:
            self._lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ExperimentLockedError("Experiment lock was acquired concurrently") from exc
        os.write(self._lock_fd, json.dumps(payload).encode("utf-8"))
        os.fsync(self._lock_fd)
        atexit.register(self.release)
        self._ensure_ledger()

    def _ensure_ledger(self) -> dict[str, Any]:
        if self.ledger_path.exists():
            return self.read_ledger()
        payload = {
            "experiment_id": self.experiment_id, "run_uuid": self.run_uuid,
            "max_physical_requests": self.max_physical_requests,
            "physical_request_counter": self.initial_physical_requests, "reservations": [],
            "created_at": _now(), "updated_at": _now(),
        }
        self._write_ledger(payload)
        return payload

    def read_ledger(self) -> dict[str, Any]:
        return json.loads(self.ledger_path.read_text(encoding="utf-8"))

    def reserve(self, package_id: str, *, explicit_retry: bool = False) -> str | None:
        ledger = self._ensure_ledger()
        package_entries = [item for item in ledger["reservations"] if item.get("package_id") == package_id]
        if any(item.get("status") == "completed" for item in package_entries):
            return None
        if package_entries and not explicit_retry:
            raise ExplicitRetryRequiredError(f"Package {package_id} was previously attempted; explicit retry is required")
        if int(ledger["physical_request_counter"]) >= int(ledger["max_physical_requests"]):
            raise PhysicalBudgetExhaustedError("MAX_PHYSICAL_REQUESTS hard ceiling reached")
        reservation_id = str(uuid.uuid4())
        ledger["physical_request_counter"] = int(ledger["physical_request_counter"]) + 1
        ledger["reservations"].append({
            "reservation_id": reservation_id, "request_id": reservation_id,
            "package_id": package_id, "logical_request_id": package_id, "run_uuid": self.run_uuid,
            "pid": self.process_id, "status": "reserved", "reserved_at": _now(),
            "explicit_retry": bool(explicit_retry),
        })
        ledger["updated_at"] = _now()
        self._write_ledger(ledger)
        return reservation_id

    def finish(self, reservation_id: str, status: str) -> None:
        ledger = self.read_ledger()
        entry = next(item for item in ledger["reservations"] if item["reservation_id"] == reservation_id)
        entry["status"] = status
        entry["finished_at"] = _now()
        ledger["updated_at"] = _now()
        self._write_ledger(ledger)

    def _write_ledger(self, payload: dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.ledger_path)

    def release(self) -> None:
        if self._lock_fd is None:
            return
        try:
            os.close(self._lock_fd)
        finally:
            self._lock_fd = None
        try:
            current = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if current.get("run_uuid") == self.run_uuid and int(current.get("pid") or 0) == self.process_id:
                self.lock_path.unlink(missing_ok=True)
        finally:
            try: atexit.unregister(self.release)
            except Exception: pass

    def __enter__(self) -> "ExperimentRequestGuard":
        self.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()
