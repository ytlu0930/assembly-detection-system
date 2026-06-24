import gradio as gr


def analyze_image(image, step):
    return f"""
圖片位置：
{image}

目前步驟：
{step}

等待 Vision API...
"""


demo = gr.Interface(
    fn=analyze_image,

    inputs=[
        gr.Image(type="filepath", label="上傳積木圖片"),
        gr.Dropdown(
            choices=[
                "step_01",
                "step_02",
                "step_03",
                "step_04",
                "step_05"
            ],
            label="選擇步驟"
        )
    ],

    outputs=gr.Textbox(
        lines=10,
        label="分析結果"
    ),

    title="積木組裝引導系統",

    description="上傳圖片後，選擇目前步驟，系統將分析積木狀態。"
)

demo.launch()
