import os
import json
import base64
from openai import OpenAI, LengthFinishReasonError

# 初始化 OpenAI Client
# 提醒：請確保環境變數 OPENAI_API_KEY 已設定，或在 .env 檔案中定義
client = OpenAI()

def encode_image(image_path):
    """將圖片轉換為 base64 格式以供 Vision API 讀取"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"[錯誤] 找不到圖片檔案: {image_path}")
        return None

def load_schema(schema_path="schema/schema.json"):
    """讀取成員 C 定義的 JSON Schema"""
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[錯誤] 讀取 Schema 失敗: {e}")
        return None

def load_prompt(step_id, prompt_path="prompts/vision_v1.txt"):
    """讀取成員 A 撰寫的 Prompt 範本，並動態帶入目前的步驟 ID"""
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()
        return template.format(step_id=step_id)
    except Exception as e:
        print(f"[錯誤] 讀取 Prompt 失敗: {e}")
        return "請分析這張積木圖片的組裝狀態。"

def analyze_assembly(image_path, step_id):
    """
    實作 Structured Outputs API 呼叫 (取代人工 JSON 驗證重試)
    使用 client.beta.chat.completions.parse() 確保 100% 符合 Schema
    """
    prompt_content = load_prompt(step_id)
    schema = load_schema()
    base64_image = encode_image(image_path)
    
    if not base64_image or not schema:
        return None

    print(f"\n[Agent] 正在啟動 GPT-4o 分析...")
    print(f"👉 步驟: {step_id}")
    print(f"👉 視角: 高解析度模式 (detail: high)")

    try:
        # 核心：使用 parse 確保輸出結構完全符合 schema.json
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            response_format=schema # 這裡是關鍵！實作 Structured Outputs
        )
        
        # 取得解析後的結構化物件
        return response.choices[0].message.parsed

    except LengthFinishReasonError:
        # 指南要求：捕捉原生 API 例外狀況 (當輸出長度超過 Token 限制時)
        print("[錯誤] API 輸出長度超過限制，JSON 結構可能不完整。")
        return None
    except Exception as e:
        print(f"[錯誤] 呼叫 API 時發生未知錯誤: {e}")
        return None

def generate_correction_feedback(analysis_result):
    """
    成員 A 主責：錯誤偵測與修正建議生成 (PHASE 03)
    根據 API 回傳結果比對零件狀態並生成中文指示。
    """
    if not analysis_result:
        return "無法取得分析結果，請檢查網路或 API 金鑰。"

    detected_parts = getattr(analysis_result, 'detected_parts', [])
    view_angle = getattr(analysis_result, 'view_angle', '未知')
    
    errors = [p for p in detected_parts if p.error_type != "correct"]
    
    print(f"\n--- 分析診斷報告 (視角: {view_angle}) ---")
    
    if not errors:
        return "✅ 檢查完成：所有零件組裝正確！請繼續下一個步驟。"
    
    feedback = f"❌ 偵測到 {len(errors)} 個組裝問題，請根據以下建議修正：\n"
    
    # 對應成員 C 定義的 error_type 進行中文解釋
    error_mapping = {
        "missingpart": "缺件",
        "extrapart": "多件",
        "positionerror": "位置錯誤",
        "wrongpart": "零件選用錯誤",
        "criticalerror": "嚴重錯誤"
    }

    for idx, err in enumerate(errors, 1):
        type_zh = error_mapping.get(err.error_type, err.error_type)
        detail = f"   {idx}. 零件 [{err.part_id}] ({err.color}): {type_zh}"
        
        # 針對特定錯誤類型提供更詳細的指示
        if err.error_type == "positionerror":
            detail += f" -> 請確認座標位置 {err.position} 是否正確。"
        elif err.error_type == "missingpart":
            detail += " -> 請補上對應的零件。"
            
        feedback += detail + "\n"
        
    return feedback

# --- 測試執行 ---
if __name__ == "__main__":
    # 這裡可以修改成成員 B 拍好的測試圖路徑
    TEST_IMAGE_PATH = "dataset/test_sample.jpg" 
    CURRENT_STEP = "step_01"

    if os.path.exists(TEST_IMAGE_PATH):
        # 執行辨識
        parsed_data = analyze_assembly(TEST_IMAGE_PATH, CURRENT_STEP)
        
        # 執行錯誤診斷與修正建議
        final_feedback = generate_correction_feedback(parsed_data)
        
        print("\n[最終回饋給使用者]:")
        print(final_feedback)
    else:
        print(f"\n[提示] 核心邏輯已準備就緒！")
        print(f"請成員 B 將測試照片放入資料夾：{TEST_IMAGE_PATH}")
