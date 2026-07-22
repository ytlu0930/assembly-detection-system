# Phase 8：BBox Candidate Selector 與 Localization Pipeline

## 目標與架構界線

Phase 8 建立獨立、可測試且可解釋的 bbox 候選篩選流程，並把它組合成獨立 localization PoC：

```text
image + prompt + target_position
  -> GroundingDetector
  -> BBoxCandidateSelector
  -> selected bbox
  -> image_annotator
```

此流程沒有接入正式 GPT Vision pipeline、Vision prompt、schema、`current_state_analyzer.py` 或 `main.py`，也沒有安裝或執行 SAM 2。

## 為何不能直接使用 Grounding DINO Top-1

Phase 7 的中央淺綠零件測試中，Grounding DINO 能產生貼合目標的 candidate，但最高 confidence bbox 經常包住整組積木。正確小框的 detection score 約為 0.218，而整體粗框約為 0.348。因此 confidence 只能代表文字與區域的相符程度，不能單獨代表 reference-guided localization 品質。

## Candidate Selector 設計

`BBoxCandidateSelector` 不匯入 PyTorch 或 Transformers。它只接受 detection dictionaries、影像尺寸與可選設定，輸出 JSON-serializable 的完整評分資訊。

每個合法 candidate 計算：

- 原始及 normalized detection score
- bbox 寬、高、面積及 area ratio
- bbox 中心點及 normalized 中心點
- target position distance、position/center score
- area score
- oversized penalty
- boundary penalty
- final selection score

超界 bbox 會先裁切到影像範圍；裁切後無面積、反向座標、非數字或格式錯誤的 bbox 會記錄在 `rejected_candidates`，不會使其他候選失敗。缺少 score 時以 0 計算，缺少 label 時使用 `unknown`。

## 評分公式與預設權重

```text
selection_score =
    0.20 * normalized_detection_score
  + 0.25 * position_score
  + 0.35 * area_score
  - 0.15 * oversized_penalty
  - 0.05 * boundary_penalty
```

所有權重可由呼叫端覆寫。相同 selection score 時依 detection score，再依原始 candidate index 決定，確保 deterministic behavior。

## Target Position

位置以 normalized `(x, y)` 表示：

| 名稱 | 目標點 |
|---|---:|
| center | (0.50, 0.50) |
| top | (0.50, 0.15) |
| bottom | (0.50, 0.85) |
| left | (0.15, 0.50) |
| right | (0.85, 0.50) |
| top_left | (0.15, 0.15) |
| top_right | (0.85, 0.15) |
| bottom_left | (0.15, 0.85) |
| bottom_right | (0.85, 0.85) |
| any | 不套用位置偏好，所有 candidate 的 position score 均為 1 |

位置距離為 candidate normalized center 到目標點的 Euclidean distance，再除以 `sqrt(2)` 並限制在 0–1；`position_score = 1 - distance`。

## Area Ratio 與 Penalty

```text
bbox_area_ratio = bbox_area / (image_width * image_height)
```

預期 area ratio 預設為 0.01–0.08。落在範圍內時 area score 為 1；低於下限時為 `ratio/minimum`，高於上限時為 `maximum/ratio`。

超過預期上限時：

```text
oversized_penalty = min(1, (area_ratio - expected_maximum) / expected_maximum)
```

bbox 任一邊距影像邊界小於 1% 時視為接近該邊界；`boundary_penalty` 為接近邊界數除以 4。

## 執行方法

所有命令使用專案既有 venv：

```powershell
$python = "C:\Users\mirro\OneDrive\Desktop\project\venv\Scripts\python.exe"

& $python -m unittest tests.test_bbox_candidate_selector -v
& $python tests/test_localization_pipeline.py --help
& $python tests/test_localization_pipeline.py `
  --image regression_subset/model03_step01_correct-01_front_01.jpg `
  --prompt "lime green rectangular block in the center" `
  --box-threshold 0.15 `
  --text-threshold 0.10 `
  --target-position center `
  --max-detections 10 `
  --device auto `
  --output-dir output/localization_pipeline

& $python tests/run_bbox_selection_experiments.py
```

## Phase 8 實驗結果（2026-07-22）

執行環境：

- Python executable：`C:\Users\mirro\OneDrive\Desktop\project\venv\Scripts\python.exe`
- Python：3.12.10
- PyTorch：2.13.0+cpu
- Transformers：5.14.1
- CUDA available：False
- `torch.version.cuda`：None
- Grounding DINO selected device：CPU
- 此輪沒有安裝或執行 SAM 2。

單張 localization pipeline（`model03_step01_correct-01_front_01.jpg`）結果：

- Grounding DINO detections：2
- score-only top-1：score 0.3481，框住接近整個組裝物
- selector：candidate index 1，detection score 0.2183，selection score 0.6415
- selector bbox：`[2952.02, 2586.43, 3926.41, 4506.31]`
- area score：1.0；position score：約 0.991；oversized/boundary penalty：0
- model load：7.450 s；inference：9.928 s；selection：0.146 ms
- 輸出：`output/localization_pipeline/model03_step01_correct-01_front_01_annotated.jpg`

`regression_subset` 共 10 張 bbox selection experiments：

- 10/10 程式執行成功，沒有 pipeline error。
- selector 與 score-only top-1 的 candidate index 在 9/10 張不同。
- 平均 inference time：約 6.889 s/image（CPU）。
- 平均 selection time：約 0.107 ms/image。
- JSON：`output/bbox_selection_experiments/bbox_selection_results.json`
- CSV：`output/bbox_selection_experiments/bbox_selection_results.csv`
- 比較圖：`output/bbox_selection_experiments/images/`

目前沒有 ground-truth bbox，因此不宣稱 IoU 或 localization accuracy。人工檢視顯示，指定的 Phase 7 正面影像上，selector 明顯優於 score-only top-1；但把相同 prompt、`target_position=center` 與 `(0.01, 0.08)` area range 套到不同視角時，部分影像會選到白球、車輪區、背景或只涵蓋目標的一部分。這表示 selector 能有效抑制過大框，但尚未形成跨視角穩定的目標識別規則。

## Phase 8 最終判定

判定為 **B**：Candidate selector 對指定 Phase 7 case 有明確改善，但仍需要 bbox ground truth、視角別 prompt／target position／area range，或 reference metadata 來校準；目前不進入 SAM 2，也不進入 Phase 9。

## 輸出格式

`LocalizationPipeline.localize()` 回傳：

- `all_detections`
- `selection_result`（含所有 candidate 指標、拒絕原因及 selection reason）
- selected bbox、label、detection score、selection score
- annotated image path
- model load、inference、selection、total time
- status 與 error message

比較實驗輸出：

- `output/bbox_selection_experiments/bbox_selection_results.json`
- `output/bbox_selection_experiments/bbox_selection_results.csv`
- `output/bbox_selection_experiments/images/*_top1.jpg`
- `output/bbox_selection_experiments/images/*_selector.jpg`

## 測試與實驗限制

selector 單元測試使用人工 detections，不載入模型。實際比較會使用 `regression_subset`，但 Repository 目前沒有這些圖片的人工 bbox ground truth，因此不能計算正式 IoU。是否改善只能透過 top-1 與 selector 標記圖進行人工視覺檢查；candidate index 的改變本身不等於定位正確。

## 後續改善方向

- 為 regression subset 建立人工 bbox ground truth 並計算 IoU/recall。
- 依零件類別或 reference metadata 提供不同 expected area range。
- 在多視角資料上校正 target position 與權重。
- 若 selector 已選對物件但 bbox 邊界仍過粗，再評估是否需要 segmentation。
