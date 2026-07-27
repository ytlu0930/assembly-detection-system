import json
from pathlib import Path

import gradio as gr

from utils.current_state_analyzer import (
    analyze_image,
    parse_filename,
    PROJECT_ROOT,
    GROUND_TRUTH_DIR,
)

# 如果 flowchart_generator.py 在專案根目錄
from flowchart_generator import generate_flowchart


def run_analysis(image, step):

    if image is None:
        flowchart = generate_flowchart(
            step_id=step,
            error_reports=[]
        )

        return (
            None,
            flowchart,
            "請先上傳圖片",
            {}
        )

    try:

        # ------------------------
        # 解析圖片名稱
        # ------------------------

        info = parse_filename(image)

        # ------------------------
        # Ground Truth JSON
        # ------------------------

        expected_state = (
            GROUND_TRUTH_DIR
            / info["model_id"]
            / f"{info['step_id']}.json"
        )

        # ------------------------
        # Reference Image
        # ------------------------

        reference_image = (
            PROJECT_ROOT
            / "input"
            / "normal"
            / f"{info['model_id']}_{info['step_id']}"
            / (
                f"{info['model_id']}_{info['step_id']}"
                f"_correct-01_{info['view_angle']}_01.jpg"
            )
        )

        # ------------------------
        # Reference 不存在
        # ------------------------

        if not reference_image.exists():

            flowchart = generate_flowchart(
                step_id=step,
                error_reports=[]
            )

            return (
                image,
                flowchart,
                f"找不到 Reference Image：\n{reference_image}",
                {}
            )

        # ------------------------
        # Ground Truth 不存在
        # ------------------------

        if not expected_state.exists():

            flowchart = generate_flowchart(
                step_id=step,
                error_reports=[]
            )

            return (
                image,
                flowchart,
                f"找不到 Ground Truth：\n{expected_state}",
                {}
            )

        # ------------------------
        # 呼叫 Vision API
        # ------------------------

        result = analyze_image(
            image_path=image,
            reference_image_path=str(reference_image),
            expected_state_path=str(expected_state),
            filename_info=info,
        )

        # ------------------------
        # Vision API 失敗
        # ------------------------

        if not result["success"]:

            flowchart = generate_flowchart(
                step_id=step,
                error_reports=[]
            )

            return (
                image,
                flowchart,
                json.dumps(result, ensure_ascii=False, indent=2),
                {}
            )

        # ------------------------
        # Vision 回傳結果
        # ------------------------

        model = result["model_response"]

        summary = model["summary"]

        confidence = {}

        for part in model["detected_parts"]:

            confidence[part["part_id"]] = part["confidence"]

        # ------------------------
        # 找出真正錯誤
        # ------------------------

        error_reports = []

        for part in model["detected_parts"]:

            if part["error_type"] != "correct":

                error_reports.append(
                    {
                        "part_id": part["part_id"],
                        "error_type": part["error_type"],
                        "confidence": part["confidence"],
                    }
                )

        # ------------------------
        # 產生流程圖
        # ------------------------

        flowchart = generate_flowchart(
            step_id=model["step_id"],
            error_reports=error_reports,
        )

        # ------------------------
        # 目前先顯示原圖
        # (之後再接 image_annotator)
        # ------------------------

        annotated_image = image

        return (
            annotated_image,
            flowchart,
            summary,
            confidence,
        )

    except Exception as e:

        flowchart = generate_flowchart(
            step_id=step,
            error_reports=[]
        )

        return (
            image,
            flowchart,
            str(e),
            {}
        )
with gr.Blocks(title="積木組裝引導系統") as demo:

    gr.Markdown("# 🧩 積木組裝引導系統")

    with gr.Row():

        # ==========================
        # 左側：圖片上傳
        # ==========================

        with gr.Column(scale=1):

            image_input = gr.Image(
                type="filepath",
                label="上傳積木圖片",
            )

            step = gr.Dropdown(
                choices=[
                    "step_01",
                    "step_02",
                    "step_03",
                    "step_04",
                    "step_05",
                ],
                value="step_01",
                label="目前步驟",
            )

            analyze_btn = gr.Button(
                "開始分析",
                variant="primary",
            )

        # ==========================
        # 中間：標記圖片
        # ==========================

        with gr.Column(scale=1):

            annotated_output = gr.Image(
                label="標記後圖片",
                interactive=False,
                height=450,
            )

        # ==========================
        # 右側：流程圖
        # ==========================

        with gr.Column(scale=1):

            flowchart_output = gr.Image(
                label="流程圖",
                interactive=False,
                height=450,
            )

    gr.Markdown("---")

    with gr.Row():

        with gr.Column(scale=2):

            suggestion_output = gr.Textbox(
                label="修正建議",
                lines=8,
                interactive=False,
            )

        with gr.Column(scale=1):

            confidence_output = gr.Label(
                label="信心分數",
            )

    analyze_btn.click(
        fn=run_analysis,
        inputs=[
            image_input,
            step,
        ],
        outputs=[
            annotated_output,
            flowchart_output,
            suggestion_output,
            confidence_output,
        ],
    )

demo.launch()