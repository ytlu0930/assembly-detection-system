import json
import socket
from unittest.mock import patch

from scripts.run_openai_image_smoke_test import main


def test_smoke_cli_defaults_to_dry_run_without_network(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch.object(socket, "create_connection") as connect:
        code = main(["--dry-run", "--output-dir", str(tmp_path / "run")])
    connect.assert_not_called()
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload["status"] == "dry-run"
    assert payload["estimated_requests"] == 1
    assert payload["flags"]["api_key_configured"] is False
    assert (tmp_path / "run" / "prompts" / "step_01.txt").is_file()
    assert (tmp_path / "run" / "inputs_manifest.json").is_file()
    assert (tmp_path / "run" / "results.json").is_file()
    assert (tmp_path / "run" / "run_summary.json").is_file()


def test_execute_without_cost_confirmation_is_blocked_before_network(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-print")
    monkeypatch.setenv("ENABLE_OPENAI_IMAGE_API", "true")
    monkeypatch.setenv("CONFIRM_OPENAI_IMAGE_API_EXECUTION", "true")
    with patch.object(socket, "create_connection") as connect:
        code = main(["--execute-api", "--output-dir", str(tmp_path / "run")])
    connect.assert_not_called()
    output = capsys.readouterr().out
    assert code == 2 and "sk-do-not-print" not in output
    assert json.loads(output)["status"] == "disabled"


def test_dry_run_overrides_all_execution_flags(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-print")
    monkeypatch.setenv("ENABLE_OPENAI_IMAGE_API", "true")
    monkeypatch.setenv("CONFIRM_OPENAI_IMAGE_API_EXECUTION", "true")
    with patch.object(socket, "create_connection") as connect:
        code = main(["--dry-run", "--execute-api", "--confirm-cost", "--output-dir", str(tmp_path / "run")])
    connect.assert_not_called()
    output = capsys.readouterr().out
    assert code == 0 and "sk-do-not-print" not in output
    assert json.loads(output)["status"] == "dry-run"
