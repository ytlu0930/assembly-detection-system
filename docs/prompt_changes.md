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

# 目前仍存在問題

## 1. Camera Alignment Sensitivity

小幅拍攝角度差異仍可能造成：

pseudo-positionerror

---

## 2. Part Identity Ambiguity

目前 GPT 仍可能將：

extrapart → positionerror

---

# 下一版預計改善方向

預計新增：

- stronger uncertainty handling
- camera pose tolerance
- multi-view reasoning
- anomaly-type refinement
- visual alignment constraints