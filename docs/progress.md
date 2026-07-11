# Day 2 Progress — Structured Comparison Test

日期：2026/05/19

---

# 今日目標

今日主要目標為改善先前 JSON-only comparison 所產生的大量 False Positive 問題，並提升 GPT-4o Vision 對積木組裝狀態的穩定辨識能力。

---

# 今日主要修改

## 一、Comparison Pipeline 重構

原本流程：

Test Image
→ GPT-4o Vision
→ expected_state JSON comparison

修改後流程：

Correct Reference Image
+ Test Image
+ expected_state JSON
→ GPT-4o Vision Structured Comparison

系統不再只依賴 expected_state JSON 進行空間推理，而是直接比較：

- 正確參考圖
- 測試圖
- expected_state JSON

以降低 spatial hallucination。

---

## 二、更新 Prompt

使用檔案：

prompts/vision_v1_2.txt

主要新增：

- Reference-guided comparison
- 視角容忍機制
- 遮擋容忍機制
- extrapart 優先判定
- uncertain handling
- structured comparison output

---

## 三、更新 test_compare.py

使用檔案：

tests/test_compare.py

新增功能：

- 自動讀取 correct reference image
- 雙圖片輸入 GPT-4o Vision
- 自動比對同 step、同視角 reference
- structured JSON output
- decision level 判定（TP / TN / FP / FN）

---

# Structured Comparison Test

Google Sheet：

Day2測試 - Structured Comparison Test

紀錄欄位：

- 測試日期
- Step ID
- View Angle
- 圖片檔名
- Ground Truth
- GPT Result
- 判定等級
- Confidence
- 備註

---

## Correct Case 測試結果

## STEP 1

Accuracy：100%

改善：

- 消除 LEFT / RIGHT hallucination
- 消除 false missingpart
- 消除 orientation hallucination

---

## STEP 2

Accuracy：100%

改善：

- wheel counting 穩定
- side-view hallucination 大幅降低
- occlusion robustness 提升

---

## STEP 3

Accuracy：75%

發現問題：

- camera alignment sensitivity
- pseudo-positionerror

特別是在：

- back view
- 結構密集區域

仍可能因拍攝角度差異而產生誤判。

---

## Extrapart 測試結果

測試：

- extrapart-A01

結果：

Detection Rate：83.3%

發現：

系統已能穩定察覺異常，但目前仍容易：

extrapart → positionerror

代表 GPT 傾向將：

「新增零件」

理解為：

「既有零件位置偏移」。

---

## MissingPart 測試結果

測試：

- missingpart-A01
- missingpart-B01

結果：

大部分視角皆可穩定辨識 missingpart。

最佳視角：

- front
- top
- side

最差視角：

- bottom

主要問題：

occlusion-induced false negative

---

## WrongPart 測試結果

測試：

- wrongpart-A01
- wrongpart-B01

結果：

大部分視角皆可成功辨識 wrongpart。

較穩定視角：

- front
- top
- left/right

較弱視角：

- bottom

---

# 今日研究發現

## 1. Reference-guided comparison 有效降低 hallucination

相較 JSON-only baseline：

- False Positive 大幅下降
- spatial hallucination 明顯改善
- wheel counting error 改善
- side-view stability 提升

---

## 2. Camera Alignment Sensitivity

當 reference 與 test image 的拍攝角度不完全一致時，仍可能產生：

pseudo-positionerror

目前尤其容易出現在：

- back view
- 結構密集區域

---

## 3. Part Identity Ambiguity

GPT 仍可能混淆：

- extrapart
- positionerror

代表目前 anomaly type classification 仍需進一步優化。

---

# 下一步規劃

預計進行：

- orientationerror 測試
- multi-error 測試
- prompt refinement
- uncertainty handling
- camera angle tolerance 改善
- correction guidance generation

---

# Day 3 Progress — Vision Pipeline Refactoring

日期：2026/06/26-07/01

---

# 今日目標

本次主要目標為完成 Vision Pipeline 模組化重構，建立 Prompt、Schema 與 Analyzer 的統一規格，避免各模組使用不同 JSON 格式造成維護困難。

---

# 今日主要修改

## 一、建立 Single Source of Truth

重新設計整體 Pipeline，將 Prompt 作為唯一 JSON 規格來源。

設計原則：

Prompt
↓
Schema
↓
current_state_analyzer
↓
test_compare_reference
↓
evaluate_metrics
↓
image_annotator

所有模組皆使用相同 JSON Output，不再各自維護不同格式。

---

## 二、JSON Output Format 統一

重新設計固定 JSON 結構。

統一欄位：

- model_id
- step_id
- view_angle
- is_error
- overall_error_type
- detected_parts
- summary

Prompt 明確要求 GPT：

- 僅輸出 JSON
- 禁止 Markdown
- 禁止 errors
- 禁止 part_differences
- 禁止額外欄位

Schema 完全依照 Prompt 建立。

---

## 三、current_state_analyzer.py 重構

原本功能：

單張圖片分析

目前改為：

Reference Image
+
Test Image
+
expected_state JSON
↓
GPT-4o Vision
↓
Schema Validation

新增功能：

- Prompt Loading
- Schema Loading
- expected_state Loading
- Reference-guided Comparison
- GPT API 呼叫
- JSON Parse
- Schema Validation
- Retry 機制
- Raw Response Log
- Parsed JSON Log
- Failed JSON Log

目前已成為整個專案唯一 Vision API 呼叫入口。

---

## 四、test_compare_reference.py 重構

移除：

- AzureOpenAI Client
- Prompt Loading
- Schema Loading
- GPT API 呼叫

目前僅負責：

- 批次掃描圖片
- 自動尋找 Reference Image
- 讀取 expected_state
- 呼叫 current_state_analyzer.analyze_image()
- 計算 TP/TN/FP/FN
- 產生 Compare JSON
- 匯出 CSV Summary

完成 Vision Pipeline 模組化。

---

# 整合測試

已完成：

- current_state_analyzer.py 單元測試
- test_compare_reference.py 批次測試
- Compare Summary JSON
- Compare Summary CSV

目前 Pipeline 已可完成：

Image
↓
Reference Comparison
↓
GPT Vision
↓
JSON Parse
↓
Schema Validation
↓
Batch Summary

---

# 今日研究發現

目前測試發現：

- Prompt 與 Schema 已統一
- WrongPart 類別辨識仍可能誤判為 Extrapart
- Confidence 欄位仍需持續優化
- Error Type Classification Accuracy 仍有提升空間

---

# 下一步規劃

- image_annotator.py
- Prompt 持續微調
- Pipeline Stress Test
- evaluate_metrics.py 統計分析

### 📅 2026-07-11 | Phase 03 資料集擴充、標註一致性重構與自動化雙重校對

#### 1. 現狀盤點與核心痛點排解
* **實拍資料集入庫**：全面導入 `model08_step05` 之實拍測試圖，總計完成 40-50 張高畫質圖像之正名與分類。
* **發現隱藏架構衝突**：在將「畸形」案例導入資料庫時，及時發現舊版分析代碼（`current_state_analyzer.py`）之 `if-elif` 條件分支與最新版憲法（`schema.json`）存在 Enum 衝突（原代碼缺少對 `criticalerror` 的硬編碼兼容）。
* **架構對齊評估**：經審視 `current_state_analyzer.py` 第 57 行的 `parse_filename` 動態切字串邏輯後，確認其具備良好的泛用性與遠見，能直接解耦硬編碼，從而順利讓 `criticalerror` 機制「原地復活」，達成 **Schema - 代碼 - 實拍檔名** 100% 鐵三角對齊。

#### 2. 嚴格落實「無底線四/五大口袋」錯誤分類
為確保資料集嚴謹度超越審查指標，拒絕向模糊分類妥協，正式將「物料拿錯（`wrongpart`）」與「位置/角度裝錯（`positionerror`）」進行本質上的嚴格剝離：
* **`wrongpart` (錯件錯誤)**：零件本體錯誤。如「後輪大小輪裝反（`wrongpart-A01`）」。
* **`positionerror` (位置/方向錯誤)**：零件正確但孔位、角度有誤。如「後側紅桿旋轉90度（`positionerror-B01`）」、「黃色釘子移位至眼睛後方（`positionerror-D01`）」。
* **變體代號防呆**：採用跳躍式/順延式變體命名（如直接使用 `B01`、`D01`），確保跨資料夾之錯誤特徵唯一性。系統經測試，完全兼容非連續性變體代號。

#### 3. 部署「自動化交叉防呆檢查機制」
* 為徹底落實成員 C「確認每張圖標註正確」之 KPI，於根目錄新建並部署 **`generate_mid_report.py`** 終極檢查腳本。
* **核心邏輯**：腳本全域遞迴掃描 `input/`，強制比對每張圖片的「實際存放資料夾路徑」與其「檔名內嵌標籤（`error_type`）」。
* **攔截實錄**：首次執行時，腳本成功發揮威力，精準攔截並警報了 12 張手滑誤放至 `wrongpart` 資料夾的 `positionerror` 照片。
* **最終戰果**：經人工手動搬移完成後，二次執行順利觸發 **`🟢 檢查通過！100% 完美吻合`** 狀態，並全自動一鍵催生：
    1.  **`ground_truth.csv`** (標準標註對照表)
    2.  **`docs/data_status.md`** (中期資料整理報告)

#### 4. Git 雲端同步
* 全數高畫質圖像（約 384.91 MiB）及全自動生成之 CSV、MD 報告，已成功全面 Push 至 GitHub 倉庫 `main` 分支。資料完整性、安全性封印完成。