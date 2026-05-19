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
PROMPT_PATH = PROJECT_ROOT / "prompts" / "vision_v1_2.txt"

RAW_DIR = PROJECT_ROOT / "logs" / "compare_raw_responses"
PARSED_DIR = PROJECT_ROOT / "logs" / "compare_parsed_json"
FAILED_DIR = PROJECT_ROOT / "logs" / "compare_parse_failed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)
FAILED_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


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

if not endpoint:
    raise ValueError("找不到 AZURE_OPENAI_ENDPOINT，請檢查 .env")
if not api_key:
    raise ValueError("找不到 AZURE_OPENAI_API_KEY，請檢查 .env")
if not gpt_deployment:
    raise ValueError("找不到 GPT4O_DEPLOYMENT，請檢查 .env")


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
    model03_step03_missingpart-A01_front_01.jpg
    model03_step03_positionerror-A01_front_01.jpg
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
    讀取 Prompt v1.2。
    """

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"找不到 Prompt：{PROMPT_PATH}")

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(filename_info: dict, expected_state: dict, reference_image_path: Path, test_image_path: Path) -> str:
    """
    將圖片資訊、reference image 資訊與 expected_state 塞進 Prompt v1.2。

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
    prompt = prompt.replace("{reference_image_name}", reference_image_path.name)
    prompt = prompt.replace("{test_image_name}", test_image_path.name)

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


def image_to_data_url(path: Path) -> str:
    """
    圖片轉成 Chat Completions API 可用的 data URL。
    """

    mime_type = get_mime_type(path)
    base64_image = encode_image_to_base64(path)
    return f"data:{mime_type};base64,{base64_image}"


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

    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_dir.rglob(f"*{ext}"))

    return sorted(image_files)


def find_reference_image(test_image_path: Path, filename_info: dict) -> Path:
    """
    自動尋找 Correct Reference Image。

    原則：
    1. 優先找 input/normal/{model_id}_{step_id}/ 底下同 model、同 step、同 view_angle 的 correct 圖。
    2. 若找不到，再擴大到 input/normal/ 底下搜尋。
    3. 若測試圖本身就是 correct，且找到自己，允許使用自己作為 reference。
       這在測 correct baseline 時可以確認 prompt 是否能判定一致。
    """

    model_id = filename_info["model_id"]
    step_id = filename_info["step_id"]
    view_angle = filename_info["view_angle"]

    pattern = f"{model_id}_{step_id}_correct-*_{view_angle}_*"

    candidate_dirs = [
        INPUT_DIR / "normal" / f"{model_id}_{step_id}",
        INPUT_DIR / "normal"
    ]

    candidates = []

    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue

        for ext in IMAGE_EXTENSIONS:
            candidates.extend(candidate_dir.rglob(f"{pattern}{ext}"))

        if candidates:
            break

    # 排序，讓每次找到的 reference 穩定一致
    candidates = sorted(set(candidates))

    if not candidates:
        raise FileNotFoundError(
            "找不到 Correct Reference Image。\n"
            f"搜尋條件：{pattern}\n"
            f"建議放置位置：{INPUT_DIR / 'normal' / f'{model_id}_{step_id}'}"
        )

    # 若有多張，優先選 correct-01，其次選排序第一張
    for candidate in candidates:
        if "correct-01" in candidate.stem:
            return candidate

    return candidates[0]


def calculate_decision_level(ground_truth: str, is_error) -> str:
    """
    依 Ground Truth 與 GPT is_error 自動計算：
    TP / TN / FP / FN

    correct + is_error false => TN
    correct + is_error true  => FP
    error   + is_error true  => TP
    error   + is_error false => FN
    """

    gt_is_correct = ground_truth == "correct"

    if isinstance(is_error, str):
        is_error_bool = is_error.lower() == "true"
    else:
        is_error_bool = bool(is_error)

    if gt_is_correct and not is_error_bool:
        return "TN"

    if gt_is_correct and is_error_bool:
        return "FP"

    if not gt_is_correct and is_error_bool:
        return "TP"

    if not gt_is_correct and not is_error_bool:
        return "FN"

    return ""


def max_confidence(parsed_json: dict) -> float:
    """
    從 detected_parts 中取最高 confidence。
    若沒有 detected_parts，回傳 0。
    """

    confidences = []

    for part in parsed_json.get("detected_parts", []):
        value = part.get("confidence")
        if isinstance(value, (int, float)):
            confidences.append(float(value))

    if not confidences:
        return 0.0

    return max(confidences)


# ============================================================
# 5. 單張圖片比對分析
# ============================================================

def analyze_single_image(image_path: Path) -> dict:
    """
    單張圖片：
    1. 解析檔名
    2. 讀取 expected_state
    3. 自動尋找 Correct Reference Image
    4. 建立 Reference-Guided Prompt
    5. 同時送出 reference image 與 test image
    6. 呼叫 GPT-4o Vision
    7. 儲存 raw 與 parsed 結果
    """

    filename_info = parse_filename(image_path)

    if not filename_info["model_id"] or not filename_info["step_id"]:
        raise ValueError(f"無法從檔名解析 model_id 或 step_id：{image_path.name}")

    expected_state = load_expected_state(
        filename_info["model_id"],
        filename_info["step_id"]
    )

    reference_image_path = find_reference_image(image_path, filename_info)

    prompt = build_prompt(
        filename_info=filename_info,
        expected_state=expected_state,
        reference_image_path=reference_image_path,
        test_image_path=image_path
    )

    reference_data_url = image_to_data_url(reference_image_path)
    test_data_url = image_to_data_url(image_path)

    response = client.chat.completions.create(
        model=gpt_deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是精準的 Reference-Guided 積木組裝錯誤偵測助手。"
                    "你必須先比較 Correct Reference Image 與 Test Image 的視覺差異，"
                    "再輔助參考 expected_state JSON。"
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "text",
                        "text": "Image A：Correct Reference Image。這張是正確答案，請作為主要視覺比對基準。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": reference_data_url,
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Image B：Test Image。請判斷這張圖相對於 Image A 是否有組裝錯誤。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": test_data_url,
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

        decision_level = calculate_decision_level(
            ground_truth=filename_info["ground_truth"],
            is_error=parsed_json.get("is_error", "")
        )

        output_data = {
            "file_info": filename_info,
            "reference_image": {
                "image_name": reference_image_path.name,
                "relative_path": str(reference_image_path.relative_to(PROJECT_ROOT))
            },
            "expected_state": expected_state,
            "model_response": parsed_json,
            "evaluation": {
                "ground_truth": filename_info["ground_truth"],
                "gpt_result": parsed_json.get("overall_error_type", ""),
                "is_error": parsed_json.get("is_error", ""),
                "decision_level": decision_level,
                "max_confidence": max_confidence(parsed_json)
            }
        }

        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return {
            "image_name": image_path.name,
            "status": "success",
            "ground_truth": filename_info["ground_truth"],
            "gpt_result": parsed_json.get("overall_error_type", ""),
            "is_error": parsed_json.get("is_error", ""),
            "decision_level": decision_level,
            "confidence": max_confidence(parsed_json),
            "reference_image": reference_image_path.name,
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
            "reference_image": reference_image_path.name,
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
            print(f"Reference：{result.get('reference_image')}")
            print(f"Ground Truth：{result.get('ground_truth')}")
            print(f"GPT Result：{result.get('gpt_result')}")
            print(f"判定等級：{result.get('decision_level')}")
            print(f"Confidence：{result.get('confidence')}")
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

    print("\nReference-guided comparison 批次測試完成。")
    print(f"總結檔案：{summary_path}")
