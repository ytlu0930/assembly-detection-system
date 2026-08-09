# Day 2 Progress — Structured Comparison Test

日期：2026/05/19


## 今日目標

今日主要目標為改善先前 JSON-only comparison 所產生的大量 False Positive 問題，並提升 GPT-4o Vision 對積木組裝狀態的穩定辨識能力。

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


## 三、更新 test_compare.py

使用檔案：

tests/test_compare.py

新增功能：

- 自動讀取 correct reference image
- 雙圖片輸入 GPT-4o Vision
- 自動比對同 step、同視角 reference
- structured JSON output
- decision level 判定（TP / TN / FP / FN）


## 四、Structured Comparison Test

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


## 五、Correct Case 測試結果

### STEP 1

Accuracy：100%

改善：

- 消除 LEFT / RIGHT hallucination
- 消除 false missingpart
- 消除 orientation hallucination

### STEP 2

Accuracy：100%

改善：

- wheel counting 穩定
- side-view hallucination 大幅降低
- occlusion robustness 提升

### STEP 3

Accuracy：75%

發現問題：

- camera alignment sensitivity
- pseudo-positionerror

特別是在：

- back view
- 結構密集區域

仍可能因拍攝角度差異而產生誤判。


## 六、Extrapart 測試結果

測試：

- extrapart-A01

結果：

Detection Rate：83.3%

發現：系統已能穩定察覺異常，但目前仍容易：

extrapart → positionerror

代表 GPT 傾向將：「新增零件」理解為：「既有零件位置偏移」。


## 七、MissingPart 測試結果

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


## 八、WrongPart 測試結果

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

## 九、今日研究發現

### 1. Reference-guided comparison 有效降低 hallucination

相較 JSON-only baseline：

- False Positive 大幅下降
- spatial hallucination 明顯改善
- wheel counting error 改善
- side-view stability 提升

### 2. Camera Alignment Sensitivity

當 reference 與 test image 的拍攝角度不完全一致時，仍可能產生：

pseudo-positionerror

目前尤其容易出現在：

- back view
- 結構密集區域

### 3. Part Identity Ambiguity

GPT 仍可能混淆：

- extrapart
- positionerror

代表目前 anomaly type classification 仍需進一步優化。


## 十、下一步規劃

預計進行：

- orientationerror 測試
- multi-error 測試
- prompt refinement
- uncertainty handling
- camera angle tolerance 改善
- correction guidance generation

-----------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------


# Day 3 Progress — Vision Pipeline Refactoring

日期：2026/06/26-07/01

## 今日目標

本次主要目標為完成 Vision Pipeline 模組化重構，建立 Prompt、Schema 與 Analyzer 的統一規格，避免各模組使用不同 JSON 格式造成維護困難。


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


## 五、整合測試

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


## 六、今日研究發現

目前測試發現：

- Prompt 與 Schema 已統一
- WrongPart 類別辨識仍可能誤判為 Extrapart
- Confidence 欄位仍需持續優化
- Error Type Classification Accuracy 仍有提升空間


## 七、下一步規劃

- image_annotator.py
- Prompt 持續微調
- Pipeline Stress Test
- evaluate_metrics.py 統計分析


-----------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------


# 2026-07-11 | Phase 03 資料集擴充、標註一致性重構與自動化雙重校對

## 1. 現狀盤點與核心痛點排解
* **實拍資料集入庫**：全面導入 `model08_step05` 之實拍測試圖，總計完成 40-50 張高畫質圖像之正名與分類。
* **發現隱藏架構衝突**：在將「畸形」案例導入資料庫時，及時發現舊版分析代碼（`current_state_analyzer.py`）之 `if-elif` 條件分支與最新版憲法（`schema.json`）存在 Enum 衝突（原代碼缺少對 `criticalerror` 的硬編碼兼容）。
* **架構對齊評估**：經審視 `current_state_analyzer.py` 第 57 行的 `parse_filename` 動態切字串邏輯後，確認其具備良好的泛用性與遠見，能直接解耦硬編碼，從而順利讓 `criticalerror` 機制「原地復活」，達成 **Schema - 代碼 - 實拍檔名** 100% 鐵三角對齊。

## 2. 嚴格落實「無底線四/五大口袋」錯誤分類
為確保資料集嚴謹度超越審查指標，拒絕向模糊分類妥協，正式將「物料拿錯（`wrongpart`）」與「位置/角度裝錯（`positionerror`）」進行本質上的嚴格剝離：
* **`wrongpart` (錯件錯誤)**：零件本體錯誤。如「後輪大小輪裝反（`wrongpart-A01`）」。
* **`positionerror` (位置/方向錯誤)**：零件正確但孔位、角度有誤。如「後側紅桿旋轉90度（`positionerror-B01`）」、「黃色釘子移位至眼睛後方（`positionerror-D01`）」。
* **變體代號防呆**：採用跳躍式/順延式變體命名（如直接使用 `B01`、`D01`），確保跨資料夾之錯誤特徵唯一性。系統經測試，完全兼容非連續性變體代號。

## 3. 部署「自動化交叉防呆檢查機制」
* 為徹底落實成員 C「確認每張圖標註正確」之 KPI，於根目錄新建並部署 **`generate_mid_report.py`** 終極檢查腳本。
* **核心邏輯**：腳本全域遞迴掃描 `input/`，強制比對每張圖片的「實際存放資料夾路徑」與其「檔名內嵌標籤（`error_type`）」。
* **攔截實錄**：首次執行時，腳本成功發揮威力，精準攔截並警報了 12 張手滑誤放至 `wrongpart` 資料夾的 `positionerror` 照片。
* **最終戰果**：經人工手動搬移完成後，二次執行順利觸發 **`🟢 檢查通過！100% 完美吻合`** 狀態，並全自動一鍵催生：
    1.  **`ground_truth.csv`** (標準標註對照表)
    2.  **`docs/data_status.md`** (中期資料整理報告)

## 4. Git 雲端同步
* 全數高畫質圖像（約 384.91 MiB）及全自動生成之 CSV、MD 報告，已成功全面 Push 至 GitHub 倉庫 `main` 分支。資料完整性、安全性封印完成。
* **規格書架構大對齊**：為配合分析代碼（`current_state_analyzer.py`）之讀取路徑，正式將 `expected/model08/` 中的 Step 01~05 規格書內容，完整全選複製並同步至 `ground_truth/model08/` 底下。實現「預期狀態」與「真實答案」的雙胞胎規格合體，徹底消除明天組員 A、B 執行時可能引發的 `FileNotFoundError` 潛在風險。



# 2026/07/12 | Phase 03 — Pipeline Regression Test & Image Annotator

---

## 今日目標

本次主要目標為確認最新版 Schema、Normalizer 與 Part Library 更新後的 Pipeline 相容性，並完成 `image_annotator.py` 獨立繪圖模組的開發與基本功能驗證。

主要工作包含：

- 進行小規模回歸測試
- 驗證新版 Schema 與設定檔相容性
- 完成 `image_annotator.py` 獨立繪圖模組
- 驗證 Bounding Box、紅綠框、文字標籤與圖片輸出功能

---

## 一、小規模回歸測試

為確認最新版 Schema、Normalizer 與 Part Library 更新後的 Pipeline 相容性，本次建立 `regression_subset/`，挑選 10 張代表性圖片進行小規模回歸測試。

### 測試涵蓋範圍

錯誤類型：

- correct
- extrapart
- missingpart
- wrongpart

視角：

- front
- back
- left
- right
- top

### 測試結果

- 成功執行：10／10
- TP：9
- TN：1
- FP：0
- FN：0
- JSON Parse Error：0
- Schema Validation Error：0

本次回歸測試子集之執行成功率為 100%，且未發生 JSON Parse Error 或 Schema Validation Error。

### 相容性驗證結果

本次測試確認：

- `schema/schema.json` 與 `schema/vision_output_schema.json` 目前內容一致
- `uncertain` 已正確納入 Schema 與 Normalizer 規格
- `view_angle` 可正規化為 `top`、`bottom`、`front`、`back`、`left`、`right`
- 最新設定檔可正常配合 `current_state_analyzer.py`
- 批次測試可正常輸出 JSON 與 CSV Summary
- 最新 Schema、Normalizer 與 Part Library 更新未造成目前測試案例的 Pipeline 錯誤

### 輸出檔案

- `logs/compare_summaries/compare_summary_20260712_174706.json`
- `logs/compare_summaries/compare_summary_20260712_174706.csv`

### 測試限制

本次為小規模回歸測試，主要目的為驗證 Pipeline 相容性，尚未涵蓋所有錯誤類型與視角。

目前未涵蓋：

- positionerror
- criticalerror
- uncertain
- bottom 視角
- API Rate Limit
- Retry 機制
- 長時間連續執行穩定性

因此，本次測試結果不代表完整資料集之最終準確率。

---

## 二、Image Annotator Module

完成 `utils/image_annotator.py` 獨立繪圖模組，用於在原始圖片上根據 Annotation 資訊繪製積木位置框與文字標籤。

### 函式介面

```python
annotate_image(
    image_path: str,
    annotations: list
) -> str
```

Annotation 基本格式：

{
  "part_id": "part_01",
  "bbox": [60, 60, 220, 190],
  "status": "correct",
  "error_type": "correct"
}

### 已實作功能

- annotate_image(image_path, annotations) -> str
- bbox 格式與座標驗證
- bbox 超出圖片範圍時進行座標限制
- correct 狀態使用綠色矩形框
- error 狀態使用紅色矩形框
- 正確積木顯示 part_id 標籤
- 錯誤積木顯示 part_id 與 error_type 標籤
- 自動建立 output/annotated/ 輸出資料夾
- 自動產生標記後圖片
- 回傳標記後圖片路徑
- 圖片不存在時的錯誤處理
- 圖片無法解碼時的錯誤處理
- Annotation 格式錯誤處理
- 無效 status 與 bbox 格式驗證

### 獨立測試結果

已使用實際積木圖片與人工設定的 bbox 座標進行獨立測試。

測試結果：
* OpenCV 圖片讀取成功
* 綠色矩形框繪製成功
* 紅色矩形框繪製成功
* 文字標籤顯示成功
* 標記後圖片輸出成功
* 輸出路徑回傳成功

測試輸出位置：output/annotated/

目前可確認 image_annotator.py 的獨立繪圖功能已正常運作。

## 三、目前限制

現行 Vision Output Schema 的 detected_parts 結構包含：

* part_id
* error_type
* description
* confidence

目前尚未包含：

* bbox
* status

因此，雖然 image_annotator.py 已能根據 Annotation 資訊正常繪製 Bounding Box 與標籤，但目前使用的是人工設定的測試座標，尚未與 GPT Vision 的實際定位結果完成端到端整合。

目前 Pipeline 狀態：

Test Image
    ↓
current_state_analyzer.py
    ↓
detected_parts
    ↓
目前缺少 bbox 定位資訊
    ↓
image_annotator.py

後續需確認 bbox 的來源與定位策略，再決定是否：

- 調整 Prompt，使模型輸出 bbox
- 調整 Vision Output Schema
- 使用其他視覺定位模組
- 採用獨立 Object Detection／Segmentation 方法提供定位資訊

若後續修改 Prompt、Schema 或 JSON Output Structure，需先通知其他成員並同步確認相關模組相容性。

## 四、今日完成項目
* 完成最新版 Repository Pull 與相容性確認
* 確認 schema/schema.json 與 schema/ vision_output_schema.json 內容一致
* 確認 uncertain Schema 與 Normalizer 規格一致
* 確認 view_angle 正規化設定正確
* 完成 10 張代表性圖片的小規模回歸測試
* 完成 utils/image_annotator.py 獨立繪圖模組
* 完成 Bounding Box、紅綠框與文字標籤功能驗證
* 確認標記圖片可正常輸出至 output/annotated/

## 五、下一步規劃

預計進行：
* 優化拍攝規範文件 v2.0
* 新增常見拍攝失誤案例
* 新增正確與錯誤拍攝範例對照
* 新增拍攝前 5 項檢查清單
* 確認 bbox 來源與定位策略
* 檢查 API Timeout 與 Retry 機制
* 記錄每次 API 呼叫耗時
* 進行全 Pipeline 壓力測試
* 連續測試 10 張圖片，確認系統穩定性
* 統計平均、最短與最長 API 回應時間


-----------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------


# 2026/07/22 | Phase 03 - Grounding DINO 文字定位 PoC

## 一、完成項目

- 建立 `utils/grounding_detector.py`
- 建立單張定位測試 `tests/test_grounding_detector.py`
- 建立 Prompt × Threshold 實驗工具
  `tests/run_grounding_experiments.py`
- 成功串接既有 `utils/image_annotator.py`
- 完成 5 個 Prompt × 3 組 Threshold，共 15 組實驗
- 在正式 `project/venv` 補齊 PyTorch 與相關套件
- 在正式專案環境中成功重現單張測試與完整實驗
- 未修改既有 Vision Prompt、Schema、Analyzer 與主流程
- 未建立 localization pipeline
- 未安裝 SAM 2

## 二、實驗結果

- 模型：Grounding DINO Base
- 建議 Prompt：
  `lime green rectangular block in the center`
- 建議 box threshold：`0.15`
- 建議 text threshold：`0.10`
- 正確中央零件候選 score：`0.2183`
- 15／15 組實驗均正常完成
- Grounding DINO bbox 可正常交由 `image_annotator.py` 產生標記圖片

## 三、判定

PoC 判定為 B：Grounding DINO 具備文字引導定位能力，但需加入 bbox
候選篩選機制。

目前最高分 bbox 經常包覆整組積木，因此不能直接取 top-1 detection。
下一階段應優先研究中心距離、bbox 面積比例與目標位置條件，不需立即
導入 SAM 2。

## 四、下一步規劃

1. 設計 bbox candidate selector
2. 比較 score-only 與多條件篩選結果
3. 使用多張不同角度及不同錯誤類型圖片驗證
4. 篩選穩定後，再建立獨立 localization pipeline
5. 若定位目標正確但邊界仍過粗，再評估 SAM 2


-----------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------


# 2026/07/22 | Phase 03 - BBox Candidate Selection Pipeline（完成）

## 一、目標

Grounding DINO 在部分案例中會將最高分 bbox 指向整體組裝物，而非欲定位的積木零件。
本階段目標為建立 bbox candidate selector，自多個 detections 中挑選較符合目標區域的 bbox，
改善單純採用 top-1 detection 的定位結果。


## 二、實作內容

新增模組：

- utils/bbox_candidate_selector.py
- utils/localization_pipeline.py

新增測試：

- tests/test_bbox_candidate_selector.py
- tests/test_localization_pipeline.py
- tests/run_bbox_selection_experiments.py

新增文件：

- docs/localization_pipeline.md


## 三、Selector 規則

目前採用加權評分方式：

- Detection confidence
- Center position
- Bounding box area

由 selector 對所有 candidate bbox 重新排序，
取代直接使用 Grounding DINO top-1 detection。


## 四、驗證結果

環境：

- Python 3.12.10
- PyTorch 2.13.0+cpu
- CPU inference
- CUDA unavailable

測試結果：

- Selector unit tests：11 / 11 PASS
- compileall：PASS
- CLI smoke test：PASS
- import smoke test：PASS
- git diff --check：PASS

Batch Experiment：

- 測試圖片：10 張
- 成功完成：10 / 10
- Selector 與 top-1 選擇不同：9 / 10

平均時間：

- Grounding DINO inference：約 6.889 sec / image
- BBox selection：約 0.107 ms / image


## 五、成果

指定 PoC case 可成功由 selector 選出中央 lime-green block，
改善原先 top-1 偏向整體模型 bbox 的問題。

實驗結果輸出：

- output/bbox_selection_experiments/
    - bbox_selection_results.json
    - bbox_selection_results.csv
    - images/


## 六、限制

目前 selector 在部分視角仍可能誤選：

- 白色球體
- 車輪區域
- 背景區域

跨視角泛化能力仍不足。


## 七、最終評估

評級：B

原因：

- 指定 PoC case 改善明顯。
- 尚未建立 bbox ground truth。
- 尚未完成 view-specific prompt 與 position / area 規則。
- 尚不適合直接整合 SAM 2。


## 八、後續規劃

預計完成：

- 建立 bbox ground truth
- 增加 view-specific prompt
- 建立 view-specific selection rule
- IoU / localization accuracy 評估

完成上述項目後，再評估是否進入 SAM 2。


-----------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------


# 2026/07/22 | Phase 03 - Output 結構統一與現有資料集盤點凍結

## 一、目標

本階段針對執行後產生的零散輸出資料夾進行整理，建立統一且可追溯的 output 結構。

同時，在不重新拍攝、不修改原始圖片的前提下，盤點目前專案中的完整資料集，檢查檔名、錯誤類型、重複檔案與 Correct Reference 配對狀況，並建立 freeze manifest，將現有資料集凍結為本專題後續測試使用的固定版本。

本階段未安裝或執行 SAM 2。


## 二、Output 結構統一

### 原始問題

執行後，`output/` 根目錄存在多個不同命名方式的資料夾：

```text
output/
├── annotated/
├── bbox_selection_experiments/
├── grounding/
├── grounding_experiments/
└── localization_pipeline/
```


-----------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------


# 2026/07/22 | Phase 03 - BBox Candidate Selector & Localization Pipeline

## 一、完成項目

- 新增 `utils/bbox_candidate_selector.py`：使用 confidence、position、area、oversized 與 boundary 指標做 deterministic bbox selection。
- 新增 11 個 selector 單元測試，涵蓋中心偏好、過大框、空 detections、非法 bbox、缺少欄位、`target_position=any` 與 deterministic behavior。
- 新增 `utils/localization_pipeline.py`，串接既有 `GroundingDetector`、`BBoxCandidateSelector` 與 `image_annotator`。
- 新增 localization CLI 與 bbox selection batch experiments。
- 使用專案既有 `venv\Scripts\python.exe` 完成單張 PoC 與 `regression_subset` 10 張比較實驗。
- JSON、CSV、score-only 圖與 selector 圖輸出至 `output/bbox_selection_experiments/`。
- 未修改 Vision prompt、Vision schema、`current_state_analyzer.py`、GPT Vision pipeline 或 `main.py`。
- 未安裝、未執行 SAM 2；未進入 Phase 9。

## 二、實驗摘要

- 單張 Phase 7 正面影像：top-1 框住接近整體組裝物；selector 選到中央 lime-green rectangular block。
- 10 張 batch：10/10 執行成功；selector 在 9/10 張選擇不同於 top-1。
- CPU 平均 inference 約 6.889 s/image；selection 約 0.107 ms/image。
- 跨視角人工檢視仍有誤選白球、車輪區與背景的案例；目前無 bbox ground truth，未計算 IoU。

## 三、判定

**B**：selector 已改善指定 case，但跨視角仍需要更多標註與視角別 prompt／position／area 規則。Phase 8 到此停止，不進 SAM 2 或 Phase 9。


-----------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------


# 2026/07/25 | Phase 04 Ground Truth 正式化與 Google Sheets 整理

## 一、本次完成內容

### Ground Truth 正式化

- 建立正式 Ground Truth：
  - `data/ground_truth.csv`
- Ground Truth 共 158 筆，與 frozen dataset 一致。
- 正式 taxonomy：
  - correct
  - position
  - missing
  - extra
  - wrongpart
  - criticalerror
  - orientation（0 筆，保留於 taxonomy，不納入本次評估）
- 建立：
  - `utils/taxonomy.py`
  - `utils/ground_truth_loader.py`
- 建立 Ground Truth 產生器：
  - `scripts/build_ground_truth.py`
- 新增 Ground Truth 與 taxonomy 測試。
- Batch compatibility 驗證通過。


### Legacy Ground Truth 版本辨識

確認專案存在兩份 Ground Truth：

- `ground_truth.csv`
  - Legacy（146 筆）
- `data/ground_truth.csv`
  - 正式版本（158 筆）

完成：

- Ground Truth 差異比較
- Legacy 引用盤點
- 建立版本說明文件

正式規範：

- `data/ground_truth.csv` 為唯一正式 Ground Truth。
- 根目錄 `ground_truth.csv` 保留為 Legacy，不作為正式 batch evaluation 使用。
- image_id 採用包含 split 的專案相對路徑作為唯一識別，避免 regression subset 與 input 發生同名衝突。


### Google Sheets 匯入資料建立

新增：

`data/google_sheets_import/`

包含：

- 01_ground_truth.csv
- 02_dataset_summary.csv
- 03_step_coverage.csv
- 04_batch_results_template.csv
- 05_failure_analysis_template.csv

新增：

- `scripts/export_google_sheets_csv.py`

可直接重新產生 Google Sheets 匯入資料。


## 二、Dataset Summary

正式資料集統計：

| 類型 | 數量 |
|------|-----:|
| Correct | 61 |
| Error | 97 |
| Position | 12 |
| Orientation | 0 |
| Missing | 36 |
| Extra | 15 |
| Wrongpart | 28 |
| Criticalerror | 6 |

Step Coverage：

- Position：1
- Orientation：0（Out of Scope）
- Missing：2
- Extra：2
- Wrongpart：2
- Criticalerror：1

補充：

- Correct 30 張、Error 80 張目標已達成。
- 未達成目標為：
  - Position 未達 20 張
  - Extra 未達 20 張
  - 各錯誤類型未完全涵蓋至少 3 個 Step
- Orientation 為研究未建立之資料類型，不納入本次正式評估。



## 三、 驗證結果

完成：

- Ground Truth 驗證
- Batch compatibility
- Google Sheets CSV 匯出
- Ground Truth Loader
- Taxonomy 驗證
- compileall
- pytest
- git diff --check

結果：

- Pytest：49 passed，19 subtests passed
- compileall：PASS
- git diff --check：PASS
- image_id 唯一性：158 / 158
- Ground Truth SHA-256 驗證通過
- Frozen dataset SHA-256 驗證通過
- 所有來源圖片皆未修改。


## 四、Git

完成：

- Ground Truth 正式化功能提交
- Google Sheets 匯出功能提交
- Dataset Audit taxonomy 更新提交

目前正式版本已同步至 GitHub。


## 五、Remaining

待完成：

- Google Sheets 匯入與人工整理
- Member A 完整 pipeline 整合
- Backend End-to-End Batch Test
- Batch Results 建立
- Failure Analysis
- 專題論文撰寫與整理

## 2026-08-04 Vision Part / SOP Integration

- GitHub sync confirmed `32375ae` is still `origin/main`; no pull delta.
- Isolated work is on `mirror-vision-part-sop-integration-20260804`.
- 2026-07-01 baseline: 58 unique images, error-type accuracy 55/58, affected-part at-least-one hit 8/32, composite full recall 0/16, unknown-part rate 9/43.
- Production Prompt and Schema were not changed because a controlled paid A/B has not been approved or run.
- Added a backward-compatible all-parts ErrorReport adapter, structured local-first SOP generator, provider-neutral step prompt/image interfaces, SOP-driven flowchart, full pipeline, and fixed UI adapter contract.
- Gradio now calls the UI adapter instead of rebuilding reports and flowcharts.
- Existing API smoke modules now require explicit `--execute`; pytest performs no paid calls.
- Validation: 66 tests and 19 subtests passed. Representative four-case offline full-pipeline and UI mock smoke tests passed.
- Remaining: human review of six extrapart views and wrongpart-A01 multiplicity; validate the selected image provider; approve a small Prompt/Schema A/B before production contract changes.

## 2026-08-04 Post-Commit Review Preparation

- Integration branch committed and pushed at `8a59f85`; `main` was not merged.
- Draft PR body prepared locally; GitHub CLI is not installed, so the Draft PR must be created manually.
- A 39-row affected-parts annotation workflow and review guide were created. Extrapart canonical identity and wrongpart-A01 multiplicity remain pending review.
- Formal image provider selected: OpenAI Image API / GPT Image 2 (model id: `gpt-image-2`), using Image Editing at `/v1/images/edits`.
- A disabled `OpenAIImageProvider` stub and explicit provider factory were implemented. Runtime remains `MockStepImageProvider`.
- No OpenAI client was initialized, no API key was read, and no image API request was executed. Real integration and formal image-quality evaluation remain pending.
- Experimental Prompt and Schema candidates were created under `experiments/`; neither is production and formal analyzer defaults are unchanged.
- The A/B runner defaults to dry-run and refuses API execution without a separately approved adapter. No Vision or image API experiment was executed.
- These post-commit review artifacts remain a second, uncommitted working-tree batch for human inspection.

## 2026-08-04 GPT Image 2 Adapter Implementation

- Replaced the disabled stub with a guarded `gpt-image-2` Images Edit adapter using ordered current-state and correct-reference inputs.
- Runtime remains `MockStepImageProvider`; selecting OpenAI cannot execute without two environment confirmations plus explicit code-level execution.
- Added lazy client creation, dependency injection, input/output validation, Base64/Pillow verification, redaction, finite 2/4-second retry backoff, request/step budgets, and structured provider statuses.
- Step generation is sequential and stops after a disabled/failed edit while preserving text SOP, annotation, flowchart, and the unchanged UI contract.
- Added a one-request smoke CLI that defaults to dry-run and writes standard output-manager artifacts.
- Adapter implemented: yes. API configured: not asserted. API smoke tested: no. Image quality validated: no.
- Production Vision Prompt/Schema, Ground Truth, input images, and `main` remain unchanged. No commit/push/merge was performed for this batch.

## 2026-08-08 Pipeline Convergence

- Safely stashed the second batch, fetched A's main at `e4646adc7b35b3eddea47b5d137475c90f0482a6`, rebased without conflict, and restored the stash without conflict.
- Established `main.run_pipeline` as the only formal full-pipeline entry.
- Integrated multi-ErrorReport Localization, canonical SOP aliases/swap support, V2 editing prompts, provider-backed V2 image generation, and instruction-book output.
- Replaced duplicated batch orchestration with calls to `run_pipeline` and changed Gradio to display `manifest.final_instruction_path`.
- Marked `flowchart_generator.py` and `utils.integration_pipeline.py` deprecated for runtime purposes.
- Added malformed API-key preflight; Mock remains default and no real API request was made.

## 2026-08-08 Azure GPT Image 2 Provider

- Azure GPT Image 2 Provider selected for the teacher-hosted deployment.
- Azure adapter implemented with single-image multipart edit, configurable Bearer/api-key authentication, endpoint validation, optional mask, response validation, error mapping, retry, timeout, and request budget.
- Azure API configured: pending user verification.
- Azure API smoke test: pending; Codex executed dry-run only.
- Image quality validation: pending.
- OpenAI Platform adapter retained as an alternate provider.
- Mock default retained for main, batch, UI, and tests.
- No real Azure or OpenAI image API request was executed by Codex.

## 2026-08-08 OpenAI GPT Image 2 Phase 2A E2E Validation

- OpenAI Platform single-image smoke prerequisites were confirmed: `gpt-image-2`, Images Edit, configured key, and both environment execution gates.
- Canonical `main.py` → `StepImageGeneratorV2` → `OpenAIImageProvider` Phase 2A completed with one authorized real request in an isolated output directory.
- The generation manifest records provider `openai`, model `gpt-image-2`, one requested/successful task, zero failed tasks, and operation `images.edit`.
- The generated Step 1 PNG passed Pillow validation at 1536 × 1024, and the instruction book was generated and embedded that real image for Step 1.
- Pipeline status is `partial` only because localization reliability is low and manual review is required; all five pipeline stages succeeded with no recorded errors.
- Semantic image quality remains under manual review. The generated white eyeball-like part follows the current prompt, but that localization target may not match the intended missing red-pin correction.
- Full multi-step Phase 2B and batch image generation were not executed.
- Offline validation: compileall passed; pytest reported 139 passed, 19 subtests passed, and one dry-run key-configuration assertion failure. No program/test fix was made.
- Detailed evidence: `docs/openai_e2e_phase2_validation.md`.

## 2026-08-08 Affected-Part Identity Verifier

- Implemented an evidence-based identity gate between paired test/reference localization and Correction SOP generation.
- Extended ErrorReports with identity status, confidence, evidence, verified ID, and ranked alternative candidates without hardcoded mappings.
- Valid taxonomy membership, high Vision confidence, and expected-inventory presence are no longer sufficient to create a named repair target.
- Conflict/uncertain/unresolved identities now force manual review and hard-block SOP image tasks, prompt generation, and provider calls even with manual-review override.
- missingpart-A01 regression preserves the original `EYE_BALL` prediction, rejects it as a conflict from equal test/reference evidence, leaves `verified_part_id` empty, produces no eye task, and makes zero provider calls.
- A fully offline canonical rerun independently observed eight eye candidates in both test/reference images and reached the same fail-closed `conflict` result; its mock manifest contains zero tasks and `execute_api=false`.
- Fixed OpenAI smoke dry-run environment isolation: dry-run does not load the project `.env`, and the test cannot read a real local key.
- Offline validation: 155 tests and 19 subtests passed. No OpenAI/Azure API or GPT Image request was made.
- Production Vision Prompt, Vision Schema, Ground Truth, and source images were unchanged. Phase 2B remains blocked.

## 2026-08-09 Affected-Part Baseline and Prompt A/B Preparation

- Established an offline identity baseline from 25 confirmed review rows and the latest matching 2026-07-01 parsed JSON; the 19-image A/B evaluation subset contains confirmed cases only.
- Baseline results: Exact Set Match 8.00%, part-level F1 10.53%, and false-confident identity rate 88.00% at thresholds 0.70, 0.80, and 0.90. The 0.90-1.00 confidence bin has 12.00% empirical identity accuracy.
- Prepared baseline, reference-guided, and reference+candidate experimental prompts under the current schema. Deterministic candidates use expected state/part library evidence only and never use human Ground Truth.
- Generated six-case × three-variant dry-run packages: 18 estimated requests, zero executed. Real A/B execution remains pending explicit authorization.
- Extrapart-A01 and wrongpart-A01 remain second-review cases and are excluded from primary accuracy despite appearing in dry-run demonstration packages.
- Production Prompt, Schema, Ground Truth, source images, SOP behavior, and verifier thresholds remain unchanged. Phase 2B remains blocked.
- Validation passed: 17 focused A/B tests, 19 verifier/regression tests, and 172 full-suite tests plus 19 subtests. The legacy fixed pytest output directories remain inaccessible due to Windows ACL, while fresh isolated pytest directories pass.

## 2026-08-09 Affected-Part Prompt A/B Execution

- A/B PARTIAL: executed the frozen Baseline / Reference / Reference+Candidate packages with Azure OpenAI `gpt-4o` and the current schema. Eighteen logical artifacts were recorded: 12 success and 6 failed.
- Primary execution metrics use only exact-image confirmed Ground Truth. Successful denominators were Baseline 2, Reference 0, and Reference+Candidate 3; unconfirmed extrapart-A01/wrongpart-A01 and the non-frozen front correct-control were excluded.
- Reference error cases failed schema validation. Candidate produced no A01 improvement (`EYE_BALL`, 0.95, verifier conflict), and false-confident identity remained 66.67% on its evaluable confirmed rows.
- Production verifier acceptance remained 0 and wrong-identity escape remained 0; verifier thresholds were unchanged.
- Request audit incident: the first shell timeout left its Python child running, and a resume process overlapped it. Reconstructed physical requests are 31, above the intended 18 budget. No further API request was made after discovery; the runner now uses an exclusive execution lock.
- Decision: `NO_CLEAR_IMPROVEMENT`; no production Prompt/Schema/Ground Truth/source-image change. GPT Image generation and Phase 2B were not executed. Phase 2B remains blocked.
- Post-run validation: 33 focused tests passed; full suite passed 173 tests and 19 subtests; compileall and diff check passed.

## 2026-08-09 Prompt A/B Offline Forensic and Safety Hardening

- Recovered all five Reference validation failures for analysis from the validator's retained instance payload; every call had returned parseable JSON, but schema metadata at the top level violated `additionalProperties: false`. Recovery artifacts are explicitly excluded from primary metrics.
- Confirmed missingpart-A01 Candidate included both `EYE_BALL` and `PIN_RED_SHORT` and covered the complete 15/15 canonical inventory, so `EYE_BALL` was in-set but semantically wrong. The candidate constraint was therefore weak rather than violated.
- Added deterministic experimental Candidate membership enforcement. Out-of-set predictions are marked `violation`, receive no verified ID, require manual review, and are never mapped to a nearest candidate. Added normal and high-confidence Candidate Violation Rate metrics.
- Added a PID-aware exclusive lock, experiment run UUID, atomic persistent pre-request reservation ledger, hard physical budget, completed-package resume skip, and explicit-retry-only behavior for attempted packages. The corrected incident count of 31 remains persisted and blocks further requests under the exhausted 18-request ledger.
- Prepared, but did not execute, the six-request targeted Reference/Reference+Candidate plan for missingpart-A01, missingpart-B01, and wrongpart-B01.
- No production Prompt or Schema was changed. No API or Phase 2B execution occurred. Phase 2B remains blocked.

## 2026-08-09 Targeted A/B Offline Evaluation

- The user-executed targeted Vision A/B completed exactly six logical and six physical requests with zero retries; the ledger and all six raw/parsed response artifacts are present, and no request-audit incident occurred.
- Offline evaluation joined exact-image confirmed frozen Ground Truth after inference only. No API, resume, retry, GPT Image, or Phase 2B execution occurred during evaluation.
- Reference schema validity was 0/3 because every response repeated the `$schema`/`title`/`type` metadata echo. Reference primary metric denominators are therefore null/N/A.
- Reference+Candidate schema validity was 3/3, but Exact Match was 0%, At-least-one Recall 33.33%, All-parts Recall 0%, Part F1 28.57%, and false-confident identity rate at 0.80 was 66.67%.
- missingpart-A01 and missingpart-B01 remained high-confidence `EYE_BALL` errors. wrongpart-B01 recovered `PIN_RED_SHORT` but missed the second swap identity `PIN_YELLOW`.
- All Candidate lists covered 15/15 inventory IDs and are weak constraints. Candidate violation rate was 0%, verifier acceptance 0%, and wrong-identity escape 0; all Candidate cases remained blocked for review.
- Decision: `NO_CLEAR_IMPROVEMENT`; recommended prompt variant: `NONE`; next experiment: `LOCALIZATION_GUIDED_ROI`; Phase 2B remains `BLOCK`.
- Production Vision Prompt, production Vision Schema, Ground Truth, and source images were unchanged by this offline evaluation.
- Offline validation passed: 37 focused tests; isolated no-network smoke test; 192 full-suite tests plus 19 subtests; compileall; and git diff check. A mocked Hugging Face socket attempt caused one initial full-suite ordering failure, but no real connection occurred and the fully offline rerun passed.

## 2026-08-09 Localization-Guided ROI Identity PoC

- Added an offline paired Test/Correct-Reference ROI pipeline using color-component deltas, assembly-relative position, expected state, part library, optional cached Grounding DINO corroboration, and the existing bbox selector.
- Added deterministic ROI candidate reduction without review CSV, Ground Truth, case-ID rules, A01 exceptions, or hardcoded target identities. Low localization evidence fails closed with no candidates.
- Across missingpart-A01, missingpart-B01, and wrongpart-B01, candidate counts fell from 15 to 5, 5, and 6 (64.44% mean reduction), while evaluation-only confirmed-GT coverage remained 3/3. `EYE_BALL` was excluded from all three reduced sets.
- missingpart-A01's top-view evidence includes the absent red short-pin location; missingpart-B01 retains a bottom-view small-wheel ROI; wrongpart-B01 retains both red/yellow identities and paired reference/test swap evidence. Cross-view false positives remain, so every package requires manual review.
- Production Vision Prompt/Schema, Ground Truth, source images, GPT Image, and Phase 2B were untouched. No external API request was made; Phase 2B remains blocked.

## 2026-08-09 ROI Direct vs Checklist Preflight

- Prepared the fixed three-case, two-method ROI Direct versus component-checklist experiment as exactly six logical request packages (`EXP-001`–`EXP-006`). No API request was executed.
- Froze existing ROI PoC packages, candidates, bboxes, crops, full source images, prompts, schemas, and runner with SHA-256 validation. Localization was not rerun and no bbox was manually selected.
- Added experiment-only Direct/Checklist prompts and schemas, dynamic candidate membership enforcement, schema-metadata sanitization, a deterministic checklist rule engine, fail-closed UNCERTAIN behavior, and paired wrongpart/swap handling.
- Added PID/process lock verification, fresh isolated ledger, pre-transport reservation, hard physical ceiling 6, SDK retry 0, schema retry 0, and resume no-resend behavior. The 31/18 incident ledger and targeted ledger are not reused.
- Added offline affected-part evaluation, checklist resolved-only confusion matrix with separate UNCERTAIN count, Direct/Checklist comparison chart, deterministic correction annotations, four-panel case figures, and stable thesis CSV contracts. Matplotlib has an OpenCV offline fallback in the current venv.
- Final preflight run: `analysis/roi_direct_vs_checklist/run_20260809_preflight`; UUID `d9d0c3f0-7a57-41ed-872c-b2ab42f4db97`; environment ready; GT leakage audit PASS; physical requests 0.
- Production Vision Prompt/Schema, Ground Truth, source images, GPT Image, and Phase 2B were untouched. Phase 2B remains blocked.

## 2026-08-09 ROI Direct vs Checklist Offline Evaluation

- The user-executed experiment completed exactly six logical and six physical requests with six completed reservations, zero retries, six response artifacts, and no request incident. This follow-up performed no API request, resume, or retry.
- Froze all six response artifacts with SHA-256 before loading labels. Exact-image confirmed Ground Truth was joined only after the label-free snapshot passed integrity checks.
- Direct original schema validity was 3/3; Checklist was 0/3 because every response returned `results` plus categorical fields instead of the experiment schema. Raw responses remain unchanged. A label-free analysis-only recovery enabled deterministic rule-engine semantic evaluation while remaining excluded from original schema-valid counts.
- Direct metrics: Exact Match 33.33%, at-least-one/all-parts recall 33.33%, Part F1 25.00%, false-confident identity @0.80 100.00%.
- Recovered Checklist semantic metrics: Exact Match 33.33%, at-least-one/all-parts recall 66.67%, Part F1 54.55%, false-confident identity @0.80 57.14%, Unknown Rate 33.33%.
- Checklist component resolved-only metrics: TN=3, FP=4, FN=0, TP=3, accuracy 60.00%, precision 42.86%, recall 100.00%, F1 60.00%; six of 16 checks were UNCERTAIN.
- missingpart-A01 was identified by both methods; missingpart-B01 Direct was a false-confident red-pin error while Checklist failed closed; wrongpart-B01 Checklist contained both swap identities but also four false positives.
- Deterministic annotations and thesis figures were generated without GPT Image. Unverified frozen ROI proposals are labeled as such; final correction panels do not assert a bbox because every result remains conflict/unresolved and requires manual review.
- Decision: `NO_CLEAR_IMPROVEMENT`; recommended production method `NONE`; deterministic fail-closed annotation retained; Phase 2B remains `BLOCK`.
- Production Vision Prompt/Schema, Ground Truth, and source images were unchanged. No GPT Image or Phase 2B execution occurred during evaluation.

## 2026-08-09 Checklist Schema Robustness and Thesis Consolidation

- Added an experiment-only, fail-closed Checklist response normalizer and removed duplicated normalization logic from the evaluator. It handles only the observed contract aliases/types, validates exact candidate membership, has no Ground Truth input, and does not alter raw responses.
- Reparsed `EXP-002`, `EXP-004`, and `EXP-006` directly from stored raw message content. Original model schema compliance remains 0/3; deterministic normalized analysis compliance is 3/3.
- Raw response SHA-256 values were identical before and after normalization. Normalized semantic fields were 3/3 identical to the prior label-free recovery; no identity, status, observation, or confidence was changed semantically.
- Consolidated the confusion matrix, comparison chart, three case figures, and five thesis tables under the run's `thesis_artifacts/` directory with a SHA-256 manifest.
- Added `docs/thesis_experiment_summary.md`. Production Prompt/Schema, Ground Truth, source images, GPT Image, API execution, and Phase 2B remain untouched; Phase 2B remains blocked.
