import os
import json
import base64
from openai import OpenAI

# 1. 初始化 OpenAI Client (會自動讀取環境變數中的 OPENAI_API_KEY)
client = OpenAI()

def encode_image(image_path):
    """將圖片轉換為 base64 格式以供 Vision API 讀取"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_schema(schema_path="schema/schema.json"):
    """讀取成員 C 定義的 JSON Schema"""
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_prompt(step_id, prompt_path="prompts/vision_v1.txt"):
    """讀取你寫的 Prompt 範本，並動態帶入目前的步驟 ID"""
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(step_id=step_id)

def analyze_lego_assembly(image_path, step_id):
    """呼叫 OpenAI GPT-4o Vision API 並強制執行結構化輸出"""
    # 讀取 Prompt 與 Schema
    prompt_content = load_prompt(step_id)
    schema = load_schema()
    base64_image = encode_image(image_path)
    
    print(f"[Agent] 開始分析步驟: {step_id}, 圖片: {image_path}...")
    
    try:
        # 使用 beta.chat.completions.parse 強制結構化輸出
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
                                "detail": "high" # 依據指南，必須使用高解析度模式
                            }
                        }
                    ]
                }
            ],
            response_format=schema # 直接餵入 schema.json 的結構
        )
        
        # 解析出完全符合 Schema 的結構化 JSON 結果
        analysis_result = response.choices[0].message.content
        return json.loads(analysis_result)
        
    except Exception as e:
        print(f"[錯誤] API 呼叫或解析失敗: {e}")
        return None

def check_errors(analysis_result):
    """
    成員 A 主責的錯誤偵測邏輯。
    分析 AI 回傳的零件列表中，是否存在任何組裝錯誤。
    """
    if not analysis_result:
        return "無法取得分析結果"
    
    parts = analysis_result.get("detected_parts", [])
    errors_found = []
    
    for part in parts:
        part_id = part.get("part_id")
        error_type = part.get("error_type", "correct")
        
        if error_type != "correct":
            errors_found.append({
                "part_id": part_id,
                "error_type": error_type,
                "details": f"零件 {part_id} 偵測到錯誤類型: {error_type} (位置: {part.get('position', '未提供')})"
            })
            
    # 輸出診斷報告
    if errors_found:
        report = f"❌ 偵測到 {len(errors_found)} 個組裝錯誤：\n"
        for err in errors_found:
            report += f"- {err['details']}\n"
    else:
        report = "✅ 恭喜！該步驟組裝完全正確，無任何錯誤。"
        
    return report

# --- 測試主程式 ---
if __name__ == "__main__":
    # 測試用的圖片路徑（可請成員 B 提供一張拍好的測試圖放到 dataset 資料夾）
    test_image = "dataset/test_step_01.jpg" 
    test_step = "step_01"
    
    # 確保測試目錄存在（如果只是先寫好程式，可以先建立空資料夾）
    os.makedirs("dataset", exist_ok=True)
    
    if os.path.exists(test_image):
        # 執行辨識
        result = analyze_lego_assembly(test_image, test_step)
        print("\n[AI 原始結構化回傳]:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 執行成員 A 的錯誤偵測診斷
        diagnostic_report = check_errors(result)
        print("\n[成員 A 錯誤診斷報告]:")
        print(diagnostic_report)
    else:
        print(f"\n[提示] 主程式架構已就緒！請成員 B 拍攝一張照片並命名為 '{test_image}' 後，即可執行此主程式進行完整測試。")
