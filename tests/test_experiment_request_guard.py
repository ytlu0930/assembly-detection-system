import json

import pytest

from utils.experiment_request_guard import (
    ExperimentLockedError,
    ExperimentRequestGuard,
    ExplicitRetryRequiredError,
    PhysicalBudgetExhaustedError,
)


def guard(tmp_path, **kwargs):
    return ExperimentRequestGuard(
        experiment_id="exp", lock_path=tmp_path / "run.lock",
        ledger_path=tmp_path / "ledger.json", max_physical_requests=kwargs.pop("max_requests", 2),
        process_id=kwargs.pop("pid", 100), pid_checker=kwargs.pop("pid_checker", lambda pid: pid == 100),
        **kwargs,
    )


def test_second_live_process_is_blocked(tmp_path):
    first = guard(tmp_path); first.acquire()
    second = guard(tmp_path, pid=200, pid_checker=lambda pid: pid == 100)
    with pytest.raises(ExperimentLockedError, match="live PID"):
        second.acquire()
    first.release()


def test_completed_package_is_skipped_after_restart(tmp_path):
    first = guard(tmp_path); first.acquire()
    reservation = first.reserve("package-1"); first.finish(reservation, "completed"); first.release()
    second = guard(tmp_path); second.acquire()
    assert second.reserve("package-1") is None
    assert second.read_ledger()["physical_request_counter"] == 1
    second.release()


def test_reserved_package_after_crash_is_not_resent_on_restart(tmp_path):
    first = guard(tmp_path); first.acquire()
    first.reserve("package-1")
    first.release()
    second = guard(tmp_path); second.acquire()
    with pytest.raises(ExplicitRetryRequiredError):
        second.reserve("package-1")
    assert second.read_ledger()["physical_request_counter"] == 1
    second.release()


def test_reservation_persists_before_request_and_budget_is_hard_ceiling(tmp_path):
    item = guard(tmp_path, max_requests=1); item.acquire()
    reservation = item.reserve("package-1")
    ledger = item.read_ledger()
    assert ledger["physical_request_counter"] == 1
    assert ledger["reservations"][0]["status"] == "reserved"
    with pytest.raises(PhysicalBudgetExhaustedError):
        item.reserve("package-2")
    item.finish(reservation, "failed"); item.release()


def test_failed_package_requires_explicit_retry(tmp_path):
    item = guard(tmp_path); item.acquire()
    reservation = item.reserve("package-1"); item.finish(reservation, "failed")
    with pytest.raises(ExplicitRetryRequiredError):
        item.reserve("package-1")
    retry = item.reserve("package-1", explicit_retry=True)
    assert retry and item.read_ledger()["physical_request_counter"] == 2
    item.release()


def test_stale_lock_requires_explicit_pid_validated_recovery(tmp_path):
    lock = tmp_path / "run.lock"
    lock.write_text(json.dumps({"pid": 999, "run_uuid": "old"}), encoding="utf-8")
    item = guard(tmp_path, pid_checker=lambda pid: False)
    with pytest.raises(ExperimentLockedError, match="explicit recovery"):
        item.acquire()
    item.acquire(recover_stale=True)
    assert item.run_uuid in lock.read_text(encoding="utf-8")
    item.release()
    assert not lock.exists()


def test_initial_counter_survives_process_restart_and_blocks_over_budget(tmp_path):
    item = guard(tmp_path, max_requests=18, initial_physical_requests=31)
    item.acquire()
    with pytest.raises(PhysicalBudgetExhaustedError):
        item.reserve("new-package")
    item.release()
