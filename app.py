import gradio as gr

from utils.current_state_analyzer import parse_filename
from utils.ui_pipeline_adapter import run_analysis_for_ui


def run_analysis(image, step):
    if image is None:
        return None, None, "請先上傳圖片", {}

    try:
        info = parse_filename(image)
        if not all(info.get(key) for key in ("model_id", "step_id", "view_angle")):
            raise ValueError("圖片檔名無法解析 model、step 或 view angle")
        result = run_analysis_for_ui(
            image_path=image,
            model_id=info["model_id"],
            step_id=info["step_id"],
            view_angle=info["view_angle"],
        )
        message = result["correction_text"]
        if result["warnings"]:
            message += "\n\n警告：\n" + "\n".join(result["warnings"])
        if result["error_message"]:
            message = result["error_message"]
        return result["annotated_image"], result["flowchart"], message, {"confidence": result["confidence"]}
    except Exception as exc:
        return image, None, str(exc), {}
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
