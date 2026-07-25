import os
import platform
import graphviz

def get_system_chinese_font():
    """自動判斷作業系統並回傳對應的中文字型名稱"""
    sys_name = platform.system()
    if sys_name == "Windows":
        return "Microsoft JhengHei" # 微軟正黑體
    elif sys_name == "Darwin":
        return "PingFang TC"        # Mac 蘋方體
    else:
        return "WenQuanYi Zen Hei"  # Linux 文泉驛正黑

def generate_flowchart(step_id: str, error_reports: list, output_dir: str = "output/flowcharts/") -> str:
    """
    生成積木組裝分析狀態的動態流程圖 (具備中文字型防亂碼機制)
    :param step_id: 當前步驟 ID
    :param error_reports: 偵測到的錯誤清單 (空陣列代表完全正確)
    :param output_dir: 圖片輸出目錄
    :return: 儲存的圖片路徑
    """
    os.makedirs(output_dir, exist_ok=True)
    font_name = get_system_chinese_font()
    
    # 建立有向圖並統一設定中文字型
    dot = graphviz.Digraph(comment=f'Lego Assembly Pipeline - {step_id}')
    dot.attr(
        rankdir='TB', 
        size='8,8', 
        fontname=font_name, 
        node_attr={'fontname': font_name},
        edge_attr={'fontname': font_name}
    )
    
    # --- 定義節點 (Nodes) ---
    dot.node('A', '開始分析\n(Image Input)', shape='ellipse', style='filled', fillcolor='lightblue')
    dot.node('B', 'GPT-4o Vision 分析\n(Structured Output)', shape='box')
    dot.node('C', 'JSON Schema 驗證\n(Format Check)', shape='box')
    dot.node('D', '邏輯比對與錯誤偵測\n(Error Detection)', shape='box')
    
    # 根據是否有錯誤，決定最後節點的顏色與文字
    if not error_reports:
        dot.node('E', '✅ 狀態：完全正確\n(All Correct)', shape='ellipse', style='filled', fillcolor='lightgreen')
        edge_color = 'green'
        detect_label = '無錯誤'
    else:
        error_text = f'❌ 狀態：偵測到錯誤\n(發現 {len(error_reports)} 個問題)'
        dot.node('E', error_text, shape='ellipse', style='filled', fillcolor='lightpink')
        edge_color = 'red'
        detect_label = '發現錯誤'

    # --- 定義連線 (Edges) ---
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E', label=detect_label, color=edge_color, fontcolor=edge_color)

    # --- 儲存與渲染 ---
    file_name = f"{step_id}_flowchart"
    output_path_base = os.path.join(output_dir, file_name)
    
    try:
        dot.render(output_path_base, format='png', cleanup=True)
        final_path = f"{output_path_base}.png"
        print(f"[Agent] ✅ 流程圖已成功生成 (字型: {font_name})：{final_path}")
        return final_path
    except graphviz.backend.execute.ExecutableNotFound:
        print("[錯誤] 系統找不到 Graphviz 執行檔，請確認已安裝 Graphviz 軟體。")
        return ""

# --- 測試執行 ---
if __name__ == "__main__":
    generate_flowchart("step_01", [])
    generate_flowchart("step_02", [{"part_id": "P01", "error_type": "positionerror"}])
```
eof

### 💡 建議說明：
更新這個檔案並重新執行後，`graphviz_flowchart.png` 裡面的亂碼方框就會變成清晰的「開始分析」、「GPT-4o Vision 分析」等繁體中文字了！
