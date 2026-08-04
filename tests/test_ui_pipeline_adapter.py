from utils.ui_pipeline_adapter import run_analysis_for_ui


def test_ui_contract_is_fixed():
    def pipeline(**kwargs):
        return {
            "success": True, "analysis_result": {"model_response": {}},
            "error_reports": [{"confidence": .8}], "annotated_image": "a.png",
            "correction_sop": {"steps": [{"step_number": 1, "instruction": "fix", "generated_image": "s.png"}]},
            "flowchart_image": "f.png", "warnings": [], "error_message": None,
        }
    result = run_analysis_for_ui("bad.jpg", "model03", "step03", "front", pipeline=pipeline)
    assert set(result) == {"success", "analysis_json", "annotated_image", "sop_steps", "sop_gallery", "flowchart", "correction_text", "confidence", "warnings", "error_message", "raw_result"}
    assert result["sop_gallery"] == [("s.png", "步驟 1：fix")]
