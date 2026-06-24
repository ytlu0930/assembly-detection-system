import gradio as gr
import json


def analyze_image(image_path, step):
    result = {
        "image": image_path,
        "step": step,
        "status": "success"
    }

    return json.dumps(result, indent=4)


with gr.Blocks() as demo:

    gr.Markdown("# 電容辨識系統")

    image = gr.Image(
        label="上傳圖片",
        type="filepath"
    )

    step = gr.Dropdown(
        choices=[
            "step_01",
            "step_02",
            "step_03",
            "step_04"
        ],
        label="選擇步驟"
    )

    output = gr.Textbox(
        label="JSON結果",
        lines=10
    )

    btn = gr.Button("開始分析")

    btn.click(
        fn=analyze_image,
        inputs=[image, step],
        outputs=output
    )

demo.launch()
