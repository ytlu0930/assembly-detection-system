import json
from pathlib import Path

import gradio as gr

from utils.current_state_analyzer import (
    analyze_image as vision_analyze_image,
    parse_filename,
    GROUND_TRUTH_DIR,
    PROJECT_ROOT,
)


def find_reference_image(info):
    """
    根據 model_id、step_id 自動找到 correct 參考圖
    """

    folder = (
        PROJECT_ROOT
        / "input"
        / "normal"
        / f"{info['model_id']}_{info['step_id']}"
    )

    if not folder.exists():
        raise FileNotFoundError(f"找不到資料夾：{folder}")

    extensions = ("*.jpg", "*.jpeg", "*.png")

    for ext in extensions:
        for img in folder.glob(ext):
            if "correct" in img.name.lower():
                return str(img)

    raise FileNotFoundError("找不到 Reference Image")


def run_analysis(image_path, step):
    if image_path is None:
        return "請先上傳圖片"

    try:
        # 顯示提示
        gr.Info("分析中，請稍候...")

        # 解析圖片名稱
        info = parse_filename(image_path)

        # 找 Reference Image
        reference_image = find_reference_image(info)

        # 找 Expected State JSON
        expected_state = (
            GROUND_TRUTH_DIR
            / info["model_id"]
            / f"{info['step_id']}.json"
        )

        # 呼叫 Vision API
        result = vision_analyze_image(
            image_path=image_path,
            reference_image_path=reference_image,
            expected_state_path=str(expected_state),
            filename_info=info,
        )

        if result["success"]:
            return json.dumps(
                result["model_response"],
                ensure_ascii=False,
                indent=2,
            )

        return json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return f"❌ 發生錯誤\n\n{e}"


demo = gr.Interface(
    fn=run_analysis,

    inputs=[
        gr.Image(
            type="filepath",
            label="上傳積木圖片"
        ),

        gr.Dropdown(
            choices=[
                "step_01",
                "step_02",
                "step_03",
                "step_04",
                "step_05",
            ],
            value="step_01",
            label="目前步驟（暫時保留）"
        ),
    ],

    outputs=gr.Textbox(
        label="Vision API JSON",
        lines=20,
    ),

    title="積木組裝引導系統",

    description="""
上傳待測圖片後，
系統會自動：

1. 解析圖片名稱
2. 尋找 Reference Image
3. 載入 Ground Truth JSON
4. 呼叫 GPT-4o Vision
5. 顯示 JSON 分析結果
""",
)

demo.launch()
