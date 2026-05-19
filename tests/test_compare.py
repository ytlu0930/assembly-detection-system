from dotenv import load_dotenv
from openai import AzureOpenAI
from pathlib import Path
from datetime import datetime
import os
import base64
import json
import re
import time

# ============================================================
# 1. 專案路徑設定
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_PATH = PROJECT_ROOT / ".env"

INPUT_DIR = PROJECT_ROOT / "input"
GROUND_TRUTH_DIR = PROJECT_ROOT / "ground_truth"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "vision_v1_1.txt"

RAW_DIR = PROJECT_ROOT / "logs" / "compare_raw_responses"
PARSED_DIR = PROJECT_ROOT / "logs" / "compare_parsed_json"
FAILED_DIR = PROJECT_ROOT / "logs" / "compare_parse_failed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)
FAILED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. 讀取 .env
# ============================================================

load_dotenv(dotenv_path=ENV_PATH)

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
gpt_deployment = os.getenv("GPT4O_DEPLOYMENT")

print("ENV path:", ENV_PATH)
print("Endpoint:", endpoint)
print("GPT4O deployment:", gpt_deployment)
print("API key loaded:", api_key is not None)

# ============================================================
# 3. 建立 Azure OpenAI Client
# ============================================================

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=api_key
)

# ============================================================
# 4. 檔案與 JSON 工具
# ============================================================

def parse_filename(image_path: Path) -> dict:
    """
    解析圖片檔名。

    建議格式：
    model03_step03_correct-01_front_01.jpg
    model03_step03_extrapart-A01_front_01.jpg
    """

    stem = image_path.stem
    parts = stem.split("_")

    info = {
        "image_name": image_path.name,
        "relative_path": str(image_path.relative_to(PROJECT_ROOT)),
        "model_id": "",
        "step_id": "",
        "ground_truth": "",
        "target_part": "",
        "view_angle": "",
        "image_index": ""
    }

    if len(parts) >= 5:
        info["model_id"] = parts[0]
        info["step_id"] = parts[1]

        error_part = parts[2]

        if "-" in error_part:
            error_type, target_part = error_part.split("-", 1)
            info["ground_truth"] = error_type
            info["target_part"] = target_part
        else:
            info["ground_truth"] = error_part

        info["view_angle"] = parts[3]
        info["image_index"] = parts[4]

    else:
        print(f"[警告] 檔名格式不完整：{image_path.name}")

    return info


def load_expected_state(model_id: str, step_id: str) -> dict:
    """
    根據 model_id 與 step_id 讀取 expected_state JSON。

    例如：
    ground_truth/model03/step01.json
    """

    expected_path = GROUND_TRUTH_DIR / model_id / f"{step_id}.json"

    if not expected_path.exists():
        raise FileNotFoundError(f"找不到 expected_state：{expected_path}")

    with open(expected_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_template() -> str:
    """
    讀取 Prompt v1.1。
    """

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"找不到 Prompt：{PROMPT_PATH}")

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(filename_info: dict, expected_state: dict) -> str:
    """
    將圖片資訊與 expected_state 塞進 Prompt v1.1。

    注意：
    不使用 template.format()，因為 Prompt 裡有 JSON 範例，
    JSON 的大括號 { } 會被 Python 誤判成 format 變數。
    因此改用 replace() 逐一替換指定 placeholder。
    """

    template = load_prompt_template()

    expected_state_json = json.dumps(
        expected_state,
        ensure_ascii=False,
        indent=2
    )

    prompt = template
    prompt = prompt.replace("{model_id}", filename_info["model_id"])
    prompt = prompt.replace("{step_id}", filename_info["step_id"])
    prompt = prompt.replace("{step_name}", expected_state.get("step_name", ""))
    prompt = prompt.replace("{view_angle}", filename_info["view_angle"])
    prompt = prompt.replace("{expected_state_json}", expected_state_json)

    return prompt


def encode_image_to_base64(path: Path) -> str:
    """
    圖片轉 base64。
    """

    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime_type(path: Path) -> str:
    """
    判斷圖片 MIME type。
    """

    suffix = path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        return "image/jpeg"

    if suffix == ".png":
        return "image/png"

    raise ValueError(f"不支援的圖片格式：{path}")


def extract_json(text: str) -> dict:
    """
    清理 Markdown code block 後解析 JSON。
    """

    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    return json.loads(text)


def collect_images(input_dir: Path) -> list[Path]:
    """
    批次收集 input/ 底下所有圖片。
    """

    image_files = []
    image_files.extend(input_dir.rglob("*.jpg"))
    image_files.extend(input_dir.rglob("*.jpeg"))
    image_files.extend(input_dir.rglob("*.png"))

    return sorted(image_files)

# ============================================================
# 5. 單張圖片比對分析
# ============================================================

def analyze_single_image(image_path: Path) -> dict:
    """
    單張圖片：
    1. 解析檔名
    2. 讀取 expected_state
    3. 建立 Prompt
    4. 呼叫 GPT-4o Vision
    5. 儲存 raw 與 parsed 結果
    """

    filename_info = parse_filename(image_path)

    if not filename_info["model_id"] or not filename_info["step_id"]:
        raise ValueError(f"無法從檔名解析 model_id 或 step_id：{image_path.name}")

    expected_state = load_expected_state(
        filename_info["model_id"],
        filename_info["step_id"]
    )

    prompt = build_prompt(filename_info, expected_state)

    base64_image = encode_image_to_base64(image_path)
    mime_type = get_mime_type(image_path)
    data_url = f"data:{mime_type};base64,{base64_image}"

    response = client.chat.completions.create(
        model=gpt_deployment,
        messages=[
            {
                "role": "system",
                "content": "你是精準的積木組裝錯誤偵測與 expected_state 比對助手。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        temperature=0
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = image_path.stem

    raw_path = RAW_DIR / f"{safe_name}_compare_raw_{timestamp}.json"
    parsed_path = PARSED_DIR / f"{safe_name}_compare_parsed_{timestamp}.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)

    content = response.choices[0].message.content

    try:
        parsed_json = extract_json(content)

        output_data = {
            "file_info": filename_info,
            "expected_state": expected_state,
            "model_response": parsed_json
        }

        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return {
            "image_name": image_path.name,
            "status": "success",
            "ground_truth": filename_info["ground_truth"],
            "gpt_result": parsed_json.get("overall_error_type", ""),
            "is_error": parsed_json.get("is_error", ""),
            "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
            "parsed_path": str(parsed_path.relative_to(PROJECT_ROOT))
        }

    except json.JSONDecodeError:
        failed_path = FAILED_DIR / f"{safe_name}_compare_parse_failed_{timestamp}.txt"

        with open(failed_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "image_name": image_path.name,
            "status": "parse_failed",
            "ground_truth": filename_info["ground_truth"],
            "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
            "failed_path": str(failed_path.relative_to(PROJECT_ROOT))
        }

# ============================================================
# 6. 主程式：批次執行
# ============================================================

if __name__ == "__main__":
    image_files = collect_images(INPUT_DIR)

    print(f"\n找到 {len(image_files)} 張圖片。")

    if not image_files:
        raise FileNotFoundError(f"找不到圖片，請確認 input/ 是否有圖片：{INPUT_DIR}")

    summary = []

    for idx, image_path in enumerate(image_files, start=1):
        print(f"\n[{idx}/{len(image_files)}] 正在比對：{image_path.relative_to(PROJECT_ROOT)}")

        try:
            result = analyze_single_image(image_path)
            summary.append(result)

            print(f"狀態：{result['status']}")
            print(f"Ground Truth：{result.get('ground_truth')}")
            print(f"GPT Result：{result.get('gpt_result')}")
            print(f"Raw：{result.get('raw_path')}")

            if result["status"] == "success":
                print(f"Parsed：{result.get('parsed_path')}")
            else:
                print(f"Failed：{result.get('failed_path')}")

        except Exception as e:
            error_result = {
                "image_name": image_path.name,
                "status": "api_error",
                "error": str(e),
                "file_info": parse_filename(image_path)
            }

            summary.append(error_result)
            print(f"錯誤：{e}")

        time.sleep(1)

    summary_path = PROJECT_ROOT / "logs" / f"compare_batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nExpected-state comparison 批次測試完成。")
    print(f"總結檔案：{summary_path}")