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