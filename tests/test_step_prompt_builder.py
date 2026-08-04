from utils.step_prompt_builder import build_step_prompts


def test_prompt_preserves_non_target_state_and_chains_steps():
    sop = {"steps": [
        {"step_number": 1, "action": "locate", "instruction": "locate", "visual_instruction": "locate", "affected_parts": ["P1"]},
        {"step_number": 2, "action": "insert", "instruction": "insert", "visual_instruction": "insert", "affected_parts": ["P1"]},
    ]}
    tasks = build_step_prompts(sop, test_image_path="bad.jpg", reference_image_path="good.jpg", model_id="model03", step_id="step03", view_angle="front")
    assert "every non-target part unchanged" in tasks[0]["prompt"]
    assert "Do not add nonexistent parts" in tasks[0]["prompt"]
    assert tasks[1]["previous_output"] == "step_01.png"
