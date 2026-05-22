import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

def generate_evaluation_report(csv_path: str, output_dir: str = "output/metrics/"):
    """
    讀取批次測試的 CSV 結果，計算 Accuracy/Precision/Recall/F1 Score，
    並繪製混淆矩陣 (Confusion Matrix) 儲存為圖片。
    """
    print(f"[Agent] 正在讀取測試數據：{csv_path}")
    
    try:
        # 讀取 CSV 檔案
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[錯誤] 找不到檔案: {csv_path}")
        return

    # 確保必要的欄位存在 (依據成員 B 的 Structured Comparison CSV 格式)
    if 'Ground Truth' not in df.columns or 'GPT Result' not in df.columns:
        print("[錯誤] CSV 檔案缺少 'Ground Truth' 或 'GPT Result' 欄位。")
        return

    # 提取實際標籤與預測標籤
    y_true = df['Ground Truth'].astype(str).tolist()
    y_pred = df['GPT Result'].astype(str).tolist()

    # 定義所有可能的錯誤類型標籤，確保矩陣完整性
    labels = ['correct', 'missingpart', 'extrapart', 'positionerror', 'wrongpart', 'criticalerror']
    
    # 過濾掉不在預期標籤內的髒資料 (若有的話)
    valid_indices = [i for i in range(len(y_true)) if y_true[i] in labels and y_pred[i] in labels]
    y_true = [y_true[i] for i in valid_indices]
    y_pred = [y_pred[i] for i in valid_indices]

    # --- 1. 計算文字報告 (Accuracy, Precision, Recall, F1) ---
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*40)
    print(f"📊 系統整體準確率 (Accuracy): {acc:.2%}")
    print("="*40)
    print("詳細分類指標 (Classification Report):")
    # zero_division=0 避免某個類別沒出現時報錯
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)

    # --- 2. 繪製混淆矩陣 (Confusion Matrix) ---
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 設定畫布大小與字體 (若無中文字體，改用英文顯示)
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # 繪製熱力圖
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=labels, yticklabels=labels,
                     annot_kws={"size": 14})
    
    plt.title('Lego Assembly Agent - Confusion Matrix', fontsize=18, pad=20)
    plt.xlabel('Predicted Label (GPT Result)', fontsize=14, labelpad=10)
    plt.ylabel('True Label (Ground Truth)', fontsize=14, labelpad=10)
    
    # 旋轉 X 軸標籤以防重疊
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 儲存圖片
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[Agent] ✅ 混淆矩陣圖表已儲存至：{output_path}")
    plt.close()

# --- 測試主程式 ---
if __name__ == "__main__":
    # 請將這裡的檔名替換為成員 B 放在目錄下的實際 CSV 路徑
    # 例如："tests/AI影像辨識測試記錄.xlsx - Day2測試-Structured Comparison Te.csv"
    TEST_CSV_FILE = "Day2_Structured_Comparison.csv" 
    
    # 建立一個模擬的 CSV 檔案以供立即測試 (如果還沒把檔案拉下來的話)
    if not os.path.exists(TEST_CSV_FILE):
        print(f"找不到 {TEST_CSV_FILE}，建立模擬測試資料庫...")
        mock_data = {
            "Ground Truth": ["correct", "correct", "wrongpart", "extrapart", "missingpart", "positionerror", "extrapart", "missingpart"],
            "GPT Result":   ["correct", "correct", "wrongpart", "extrapart", "correct", "correct", "correct", "correct"] # 模擬 bottom 視角造成的 FN
        }
        pd.DataFrame(mock_data).to_csv(TEST_CSV_FILE, index=False)

    generate_evaluation_report(TEST_CSV_FILE)
