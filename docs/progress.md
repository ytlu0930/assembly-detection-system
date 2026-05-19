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

# Correct Case 測試結果

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

# Extrapart 測試結果

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

# MissingPart 測試結果

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

# WrongPart 測試結果

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