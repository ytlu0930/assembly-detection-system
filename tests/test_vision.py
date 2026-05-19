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

# 零件命名規格書
PART_LIBRARY_PATH = PROJECT_ROOT / "config" / "part_library.json"

RAW_DIR = PROJECT_ROOT / "logs" / "raw_responses"
PARSED_DIR = PROJECT_ROOT / "logs" / "parsed_json"
FAILED_DIR = PROJECT_ROOT / "logs" / "parse_failed"

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
# 4. 讀取 part_library.json
# ============================================================

def load_part_library() -> dict:
    """
    讀取 config/part_library.json。
    這份檔案是正式零件命名規格書。

    目標：
    讓 GPT 輸出 BLOCK_YELLOW_CUBE、LINK_GREEN_5HOLE 等正式 part_id，
    而不是 B1、B2 這種臨時名稱。
    """

    if not PART_LIBRARY_PATH.exists():
        raise FileNotFoundError(f"找不到 part_library.json：{PART_LIBRARY_PATH}")

    with open(PART_LIBRARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def format_part_library_for_prompt(part_library: dict) -> str:
    """
    將 part_library.json 轉成適合放進 Prompt 的文字格式。
    """

    lines = []

    for part_id, aliases in part_library.items():
        alias_text = " / ".join(aliases)
        lines.append(f"- {part_id}: {alias_text}")

    return "\n".join(lines)


part_library = load_part_library()
part_library_text = format_part_library_for_prompt(part_library)


# ============================================================
# 5. 建立 Vision 測試 Prompt
# ============================================================

def build_prompt() -> str:
    """
    建立每次送給 GPT-4o Vision 的 Prompt。

    這裡會把 part_library.json 的零件名稱放進 Prompt，
    要求模型直接使用正式 part_id。
    """

    return f"""
你是一個積木組裝狀態分析助手。請仔細觀察圖片中的積木，特別注意：
1. 積木的位置 position
2. 積木方向 orientation
3. 積木是否互相遮擋
4. 是否有缺件、多件、方向錯誤或位置錯誤的可能
5. 是否有細微邊緣、接合方向、凸點排列差異

目前任務是 Vision API 能力測試，不是正式 Ground Truth 比對。
請根據圖片本身進行觀察與描述，並盡量客觀指出不確定的地方。

請務必遵守以下零件命名規則：
- detected_parts 裡的 part_id 必須優先使用下方 part_library 中的正式名稱。
- 不要使用 B1、B2、block1 這種臨時名稱作為 part_id。
- 如果無法確定是哪一個零件，part_id 請填 "UNKNOWN_PART"。
- 可以根據零件顏色、形狀、孔洞數、輪胎大小、接頭形狀判斷 part_id。
- 不要自行創造 part_library 以外的新 part_id。

以下是可用的 part_id 與描述：

{part_library_text}

請只回傳 JSON，不要使用 Markdown，不要加上 ```json。

JSON 格式如下：

{{
  "image_quality": {{
    "is_clear": true,
    "lighting": "good / normal / poor",
    "background": "clean / messy",
    "occlusion": "none / slight / serious",
    "notes": ""
  }},
  "detected_parts": [
    {{
      "part_id": "BLOCK_YELLOW_CUBE",
      "color": "",
      "shape": "",
      "approx_position": "",
      "orientation": "",
      "visible_features": "",
      "confidence": 0.0
    }}
  ],
  "assembly_state_summary": "",
  "possible_errors": [
    {{
      "error_type": "normal / position_error / orientation_error / missing_part / extra_part / uncertain",
      "target_part_id": "BLOCK_YELLOW_CUBE / UNKNOWN_PART",
      "description": "",
      "severity": "none / low / medium / high",
      "confidence": 0.0
    }}
  ],
  "orientation_analysis": "",
  "position_analysis": "",
  "uncertain_points": [],
  "recommendation_for_next_test": ""
}}
"""


# ============================================================
# 6. 圖片轉 base64
# ============================================================

def encode_image_to_base64(path: Path) -> str:
    """
    將圖片轉成 base64 字串。
    Azure OpenAI Vision API 會用 base64 data URL 接收圖片。
    """

    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime_type(path: Path) -> str:
    """
    根據圖片副檔名判斷 MIME type。
    """

    suffix = path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        return "image/jpeg"

    if suffix == ".png":
        return "image/png"

    raise ValueError(f"不支援的圖片格式：{path}")


# ============================================================
# 7. 從檔名解析測試資訊
# ============================================================

def parse_filename(image_path: Path) -> dict:
    """
    從圖片檔名解析資料集資訊。

    建議檔名格式：
    model03_step01_correct_front_01.jpg
    model03_step03_extrapart-A01_bottom_01.jpg
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
        print(f"[警告] 檔名格式不完整，無法完整解析：{image_path.name}")

    return info


# ============================================================
# 8. 清理並解析 GPT 回傳 JSON
# ============================================================

def extract_json(text: str) -> dict:
    """
    將 GPT 回傳文字轉成 JSON。
    如果模型多包了 ```json ... ```，會先移除。
    """

    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    return json.loads(text)


# ============================================================
# 9. 檢查 part_id 是否符合 part_library
# ============================================================

def validate_part_ids(parsed_json: dict, valid_part_ids: set) -> list:
    """
    檢查模型輸出的 part_id 是否存在於 part_library.json。

    若出現 UNKNOWN_PART 是允許的。
    若出現 B1、block1、或其他不存在的名稱，會記錄到 invalid_part_ids。
    """

    invalid_part_ids = []

    detected_parts = parsed_json.get("detected_parts", [])

    for part in detected_parts:
        part_id = part.get("part_id", "")

        if part_id == "UNKNOWN_PART":
            continue

        if part_id not in valid_part_ids:
            invalid_part_ids.append(part_id)

    return invalid_part_ids


# ============================================================
# 10. 自動收集 input/ 底下所有圖片
# ============================================================

def collect_images(input_dir: Path) -> list[Path]:
    """
    自動掃描 input/ 底下所有圖片。
    包含子資料夾。
    """

    image_files = []
    image_files.extend(input_dir.rglob("*.jpg"))
    image_files.extend(input_dir.rglob("*.jpeg"))
    image_files.extend(input_dir.rglob("*.png"))

    return sorted(image_files)


# ============================================================
# 11. 分析單張圖片
# ============================================================

def analyze_single_image(image_path: Path) -> dict:
    """
    對單張圖片呼叫 GPT-4o Vision API，
    並儲存 raw response 與 parsed JSON。
    """

    filename_info = parse_filename(image_path)

    base64_image = encode_image_to_base64(image_path)
    mime_type = get_mime_type(image_path)
    data_url = f"data:{mime_type};base64,{base64_image}"

    prompt = build_prompt()

    response = client.chat.completions.create(
        model=gpt_deployment,
        messages=[
            {
                "role": "system",
                "content": "你是精準的工業組裝與積木視覺檢查助手。"
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

    raw_path = RAW_DIR / f"{safe_name}_raw_{timestamp}.json"
    parsed_path = PARSED_DIR / f"{safe_name}_parsed_{timestamp}.json"

    raw_data = response.model_dump()

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    content = response.choices[0].message.content

    try:
        parsed_json = extract_json(content)

        valid_part_ids = set(part_library.keys())
        invalid_part_ids = validate_part_ids(parsed_json, valid_part_ids)

        output_data = {
            "file_info": filename_info,
            "part_id_validation": {
                "valid_part_id_count": len(valid_part_ids),
                "invalid_part_ids": invalid_part_ids
            },
            "model_response": parsed_json
        }

        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return {
            "image_name": image_path.name,
            "status": "success",
            "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
            "parsed_path": str(parsed_path.relative_to(PROJECT_ROOT)),
            "invalid_part_ids": invalid_part_ids,
            "file_info": filename_info
        }

    except json.JSONDecodeError:
        failed_path = FAILED_DIR / f"{safe_name}_parse_failed_{timestamp}.txt"

        with open(failed_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "image_name": image_path.name,
            "status": "parse_failed",
            "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
            "failed_path": str(failed_path.relative_to(PROJECT_ROOT)),
            "file_info": filename_info
        }


# ============================================================
# 12. 主程式：批次分析 input/ 所有圖片
# ============================================================

if __name__ == "__main__":
    image_files = collect_images(INPUT_DIR)

    print(f"\n找到 {len(image_files)} 張圖片。")

    if not image_files:
        raise FileNotFoundError(f"找不到圖片，請確認圖片是否放在：{INPUT_DIR}")

    summary = []

    for idx, image_path in enumerate(image_files, start=1):
        print(f"\n[{idx}/{len(image_files)}] 正在分析：{image_path.relative_to(PROJECT_ROOT)}")

        try:
            result = analyze_single_image(image_path)
            summary.append(result)

            print(f"狀態：{result['status']}")
            print(f"Raw：{result['raw_path']}")

            if result["status"] == "success":
                print(f"Parsed：{result['parsed_path']}")

                if result.get("invalid_part_ids"):
                    print(f"[提醒] 有不在 part_library.json 的 part_id：{result['invalid_part_ids']}")
                else:
                    print("part_id 檢查：全部符合 part_library.json")

            else:
                print(f"Failed：{result['failed_path']}")

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

    summary_path = PROJECT_ROOT / "logs" / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n批次測試完成。")
    print(f"總結檔案：{summary_path}")