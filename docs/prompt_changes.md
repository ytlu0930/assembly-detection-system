# Prompt Change Log

---

# vision_v1.txt

初始版本。

主要功能：

- expected_state JSON comparison
- basic error classification

問題：

- 高 False Positive
- LEFT / RIGHT hallucination
- wheel counting error
- missingpart hallucination
- orientation hallucination

---

# vision_v1_1.txt

新增：

- structured JSON output
- confidence output
- position/orientation checking

改善：

- JSON 格式穩定
- error structure 更完整

問題：

- 對 correct case 過度敏感
- side view 誤判嚴重
- occlusion robustness 不足

---

# vision_v1_2.txt

重大改版。

核心改動：

Reference-guided structured comparison

---

## 新增功能

### 1. Correct Reference Image Comparison

新增：

- Correct Reference Image
- Test Image

雙圖比較機制。

降低：

- spatial hallucination
- false missingpart
- position hallucination

---

### 2. Conservative Decision Rules

新增：

- 視角容忍
- perspective tolerance
- occlusion tolerance
- uncertain handling

避免：

「看不清楚 = 判錯」

---

### 3. Extrapart Priority Rule

新增：

若出現 reference image 中不存在的額外結構：

優先判定：

extrapart

而非：

positionerror

---

### 4. Positionerror Restriction

新增：

只有當：

- 位移明顯
- 結構不一致
- 無遮擋可能

時才允許：

positionerror

---

### 5. Structured Comparison Output

新增：

- reference_image
- decision_level
- max_confidence
- structured anomaly output

---

## 目前仍存在問題

## 1. Camera Alignment Sensitivity

小幅拍攝角度差異仍可能造成：

pseudo-positionerror

---

## 2. Part Identity Ambiguity

目前 GPT 仍可能將：

extrapart → positionerror

---

## 下一版預計改善方向

預計新增：

- stronger uncertainty handling
- camera pose tolerance
- multi-view reasoning
- anomaly-type refinement
- visual alignment constraints

---

# vision_v2.txt

重大架構重構。

本版本不再只是 Prompt 優化，而是重新定義整個 Vision Pipeline 的 JSON Output Specification。

---

## 核心設計理念

Prompt 作為 Single Source of Truth。

JSON 格式由 Prompt 定義。

Schema 完全依照 Prompt 建立。

Analyzer 僅負責：

- Prompt Loading
- GPT API
- JSON Parse
- Schema Validation

不再進行欄位修補（Field Normalization）。

---

## JSON Output 統一

固定輸出：

- model_id
- step_id
- view_angle
- is_error
- overall_error_type
- detected_parts
- summary

每個 detected_part 包含：

- part_id
- error_type
- description
- confidence

---

## 明確限制

新增：

- 僅允許輸出 JSON
- 禁止 Markdown
- 禁止 errors
- 禁止 part_differences
- 禁止 detected
- 禁止額外欄位

避免不同版本 Prompt 產生不同 JSON 結構。

---

## Pipeline 重構

current_state_analyzer.py

成為唯一 GPT API 呼叫模組。

test_compare_reference.py

不再直接呼叫 GPT。

僅負責：

- Batch Test
- Reference Search
- Ground Truth Loading
- 呼叫 Analyzer
- TP/TN/FP/FN 統計
- Compare Summary

---

## 本版改善

改善：

- Prompt、Schema、Analyzer 三者規格一致
- JSON Output 穩定性提升
- 模組間耦合降低
- 後續 image_annotator.py、evaluate_metrics.py 可直接共用同一份 JSON

---

## 下一版預計改善

- Error Type Classification Accuracy
- Confidence Estimation
- Multi-error Detection
- Bounding Box Generation（供 image_annotator 使用）

---

# vision_v2_1.txt (或當前實戰重構演進)

基於 `model08_step05` 實拍資料集（40–50張測試圖）的大規模導入與「自動化交叉防呆檢查機制」的建立，對整體 Vision Pipeline 的 Error Type 邊界進行了歷史性的嚴謹重構與對齊。

## 核心改動與正名

### 1. 嚴重結構錯誤（criticalerror）正式解開封印
* **新增支援**：正式啟用 Schema 中的 `"criticalerror"` 標籤，用以歸類「歪歪斜斜、卡榫沒對上、嚴重變形」等無法單純以缺件/錯件定義的**畸形嚴重結構錯誤**。
* **規格對齊**：徹底打通了雲端實拍檔名、`schema.json` 的 Enum 限制、以及 `current_state_analyzer.py` 的解析泛化度，實現三維一體的 100% 完美對齊。

### 2. 「wrongpart」與「positionerror」的本質剝離
為解決過往 GPT Vision 在辨識組裝積木時對這兩者的界線模糊與混淆，在實戰標註中建立了嚴格的「防呆操作定義」，並由 `generate_mid_report.py` 進行路徑強卡控：
* **`wrongpart` (物料錯誤)**：定義為「零件本身拿錯」。如：大小輪子裝反。
* **`positionerror` (方向/孔位錯誤)**：定義為「零件拿對，但角度旋轉或插錯孔位」。如：後側紅桿旋轉90度、黃釘插錯孔。

### 3. 變體代號（Variant ID）防呆鬆綁
* 驗證並確認系統完美兼容「非連續性、跳躍式」的變體命名法（如跨類別直接使用 `B01`、`D01`），程式碼字串切割（`split("-")`）完全通關，大幅提升了多標註者間的防呆卡控與特徵唯一性。

## 本版改善

* 成功透過自動化交叉比對腳本，達成實拍照片分類與檔名內嵌標籤的 **100% 一致性**。
* 一鍵全自動生成標準 `ground_truth.csv` 與結構化中期報告（`data_status.md`），大幅降低人工檢驗的 False Positive / 誤放口袋率。

## 下一版預計改善（配合 v2 既定目標進階）

* **Error Type Classification Accuracy**：利用本階段精準剝離的五大口袋實拍資料集（特別是 `criticalerror` 與 `positionerror`），進行 GPT-4o Vision 的精準度邊界壓測與少樣本（Few-Shot）提示優化。
* **Multi-error Detection**：測試單張影像同時出現 `missingpart` 與 `positionerror` 時的信心度（Confidence）表現。
* **Ground Truth 規格同步**：全面完成 `model08` Step 01 至 Step 05 的 JSON Output Specification 定義，並同步至 `ground_truth/` 目錄，確保 GPT Vision Pipeline 在進行批次比對與精準度統計（TP/TN/FP/FN）時，擁有最嚴謹且一致的單一事實來源（Single Source of Truth）。