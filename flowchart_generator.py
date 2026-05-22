import os
import graphviz

def generate_flowchart(step_id: str, error_reports: list, output_dir: str = "output/flowcharts/") -> str:
    """
    生成積木組裝分析狀態的動態流程圖
    :param step_id: 當前步驟 ID
    :param error_reports: 偵測到的錯誤清單 (空陣列代表完全正確)
    :param output_dir: 圖片輸出目錄
    :return: 儲存的圖片路徑
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 建立一個有向圖 (Digraph)
    dot = graphviz.Digraph(comment=f'Lego Assembly Pipeline - {step_id}')
    dot.attr(rankdir='TB', size='8,8', fontname='Microsoft JhengHei') # 由上到下排列，支援中文
    
    # --- 定義節點 (Nodes) ---
    dot.node('A', '開始分析\n(Image Input)', shape='ellipse', style='filled', fillcolor='lightblue')
    dot.node('B', 'GPT-4o Vision 分析\n(Structured Output)', shape='box')
    dot.node('C', 'JSON Schema 驗證\n(Format Check)', shape='box')
    dot.node('D', '邏輯比對與錯誤偵測\n(Error Detection)', shape='box')
    
    # 根據是否有錯誤，決定最後節點的顏色與文字
    if not error_reports:
        # 完全正確：綠色節點
        dot.node('E', '✅ 狀態：完全正確\n(All Correct)', shape='ellipse', style='filled', fillcolor='lightgreen')
        edge_color = 'green'
        detect_label = '無錯誤'
    else:
        # 有錯誤：紅色節點，並列出錯誤數量
        error_text = f'❌ 狀態：偵測到錯誤\n(Found {len(error_reports)} Errors)'
        dot.node('E', error_text, shape='ellipse', style='filled', fillcolor='lightpink')
        edge_color = 'red'
        detect_label = '發現錯誤'

    # --- 定義連線 (Edges) ---
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E', label=detect_label, color=edge_color, fontcolor=edge_color)

    # --- 儲存與渲染 ---
    # 設定輸出的檔名 (例如: step_01_flowchart)
    file_name = f"{step_id}_flowchart"
    output_path_base = os.path.join(output_dir, file_name)
    
    try:
        # 渲染並輸出為 PNG 格式 (會自動加上 .png 副檔名)
        dot.render(output_path_base, format='png', cleanup=True)
        final_path = f"{output_path_base}.png"
        print(f"[Agent] 流程圖已成功生成：{final_path}")
        return final_path
    except graphviz.backend.execute.ExecutableNotFound:
        print("[錯誤] 系統找不到 Graphviz 執行檔。")
        print("請確保你已安裝系統層級的 Graphviz (Windows 需下載安裝檔並加入 PATH，Mac 可用 brew install graphviz)")
        return ""

# --- 測試主程式 ---
if __name__ == "__main__":
    # 測試情境 1：無錯誤
    generate_flowchart("step_01", [])
    
    # 測試情境 2：有錯誤 (模擬有兩個零件放錯)
    mock_errors = [
        {"part_id": "P01", "error_type": "positionerror"},
        {"part_id": "P02", "error_type": "missingpart"}
    ]
    generate_flowchart("step_02", mock_errors)
