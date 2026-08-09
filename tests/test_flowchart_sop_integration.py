from pathlib import Path

from flowchart_generator import generate_sop_flowchart


def test_flowchart_reads_multiple_sop_steps(tmp_path):
    sop = {"source_step_id": "step03", "steps": [
        {"step_number": 1, "action": "remove", "instruction": "remove extra part"},
        {"step_number": 2, "action": "verify", "instruction": "verify result"},
    ]}
    assert Path(generate_sop_flowchart(sop, str(tmp_path))).is_file()


def test_flowchart_supports_correct_case(tmp_path):
    assert Path(generate_sop_flowchart({"source_step_id": "step03", "steps": []}, str(tmp_path))).is_file()
