# Grounding DINO 單張影像 PoC（Phase 0–7）

## 結論

本次 PoC 判定為 **B：Grounding DINO 可用，但需要 Prompt 與 bbox 候選篩選**。

在指定影像上，低閾值 A 的五個 Prompt 都能產生貼合中央淺綠零件的 bbox，證明 Grounding DINO 能提供可用的局部定位候選。不過，模型對「整組積木」的分數永遠高於正確的小框，所以不能直接採用 top-1 detection。中、高閾值會先移除正確小框，只留下整組積木或完全沒有 detection。

建議下一階段優先使用：

- Prompt：`lime green rectangular block in the center`
- `box_threshold=0.15`
- `text_threshold=0.10`
- 保留多個候選，再以影像中心距離及 bbox 面積篩選，不可只取最高分

此組共有 2 個候選；正確中央零件框的 score 為 0.2183，另有 1 個包住整組積木的粗框。這是五組 Prompt 中「正確框分數／額外框數」較平衡的設定。

目前不需要因 bbox 本身而直接進入 SAM 2：低閾值已能得到邊界良好的矩形框。應先驗證 bbox selection 規則在更多影像上的穩定性。

## 範圍與架構界線

本次只完成 Grounding DINO 獨立 PoC、`utils/image_annotator.py` 串接、Prompt／Threshold 矩陣與效果判定。

- 沒有修改 `prompts/`、schema、`utils/current_state_analyzer.py` 或 `main.py`。
- Grounding DINO detection 沒有寫回 Vision Output JSON。
- 沒有建立 `utils/localization_pipeline.py`。
- 沒有安裝或串接 SAM 2。

## 環境

- OS：Windows
- Python：3.12.10
- 模型：`IDEA-Research/grounding-dino-base`
- Transformers：5.14.1
- PyTorch：2.13.0+cpu
- GPU 硬體：NVIDIA GeForce RTX 5050 Laptop GPU（8 GB）
- 本次推論裝置：CPU；目前安裝的 PyTorch wheel 沒有 CUDA runtime
- 權重：933,400,872 bytes；SHA-256 `5548F844C928C4B6F411FA8CBCC2BFA8DBBBA437CB1D513975519F93C2A9ED21`

原專案 `venv` 指向已不存在的 Python 3.14，無法啟動。另因 Windows 應用程式控制會封鎖 OneDrive 目錄中的 PyTorch DLL，本次使用 AppData 下的隔離環境執行；環境與 Hugging Face cache 均未放入 Repository。

## 執行方式

安裝必要套件：

```powershell
python -m pip install torch torchvision transformers accelerate Pillow opencv-python
```

單張 baseline：

```powershell
python tests/test_grounding_detector.py `
  --image regression_subset/model03_step01_correct-01_front_01.jpg `
  --prompt "green vertical rectangular block" `
  --box-threshold 0.25 `
  --text-threshold 0.20 `
  --max-detections 5
```

完整 5×3 實驗：

```powershell
python tests/run_grounding_experiments.py
```

輸出位置：

- `output/grounding/`
- `output/grounding_experiments/grounding_experiment_results.json`
- `output/grounding_experiments/grounding_experiment_results.csv`
- `output/grounding_experiments/images/`

`output/`、`logs/`、虛擬環境及模型 cache 均由 `.gitignore` 排除。

## 單張 baseline

設定：`green vertical rectangular block`、box 0.25、text 0.20、max detections 5。

- model load：8.863 秒
- inference：8.154 秒
- detections：1
- top score：0.3459
- top bbox：`[1390.1, 2550.7, 5424.3, 4519.4]`
- 人工判定：`target_found=partial`、`box_quality=poor`、`extra_boxes=0`
- 原因：bbox 包住整組積木，沒有隔離中央淺綠零件

該 detection 已轉成 `part_id`、`bbox`、`status=correct`、`error_type=correct` 格式並交由既有 `annotate_image()` 繪圖，證明 annotator 串接可行。

## Prompt／Threshold 實驗

第二次正式矩陣執行的 model load 為 7.264 秒；15 次 CPU inference 平均 6.737 秒（6.373–7.107 秒）。

| Prompt | 組合 | detections | top score | target candidate score | target_found | box_quality | extra_boxes |
|---|---:|---:|---:|---:|---|---|---:|
| green vertical rectangular block | A | 2 | 0.3459 | 0.1969 | yes | good | 1 |
| green vertical rectangular block | B | 1 | 0.3459 | — | partial | poor | 0 |
| green vertical rectangular block | C | 0 | — | — | no | poor | 0 |
| light green vertical rectangular plastic block | A | 2 | 0.3151 | 0.1842 | yes | good | 1 |
| light green vertical rectangular plastic block | B | 1 | 0.3151 | — | partial | poor | 0 |
| light green vertical rectangular plastic block | C | 0 | — | — | no | poor | 0 |
| lime green rectangular block in the center | A | 2 | 0.3481 | 0.2183 | yes | good | 1 |
| lime green rectangular block in the center | B | 1 | 0.3481 | — | partial | poor | 0 |
| lime green rectangular block in the center | C | 0 | — | — | no | poor | 0 |
| central light green plastic plate | A | 3 | 0.4726 | 0.2142 | yes | good | 2 |
| central light green plastic plate | B | 1 | 0.4726 | — | partial | poor | 0 |
| central light green plastic plate | C | 1 | 0.4726 | — | partial | poor | 0 |
| green toy construction piece | A | 3 | 0.5112 | 0.2333 | yes | good | 2 |
| green toy construction piece | B | 1 | 0.5112 | — | partial | poor | 0 |
| green toy construction piece | C | 1 | 0.5112 | — | partial | poor | 0 |

Threshold 組合：

- A：box 0.15、text 0.10
- B：box 0.25、text 0.20
- C：box 0.35、text 0.25

人工判定中的 `good` 指正確 target candidate 貼合中央淺綠直立零件；表中的 top score 則全部對應整組積木粗框。完整 bbox、分數、耗時、輸出影像與錯誤欄位保存在 JSON／CSV。

## Phase 7 效果判定

- **A（可直接整合）**：否。top-1 detection 穩定選到整組積木。
- **B（可用，但需 Prompt／bbox 篩選）**：是。低閾值 5/5 Prompt 都有 good target candidate。
- **C（不穩定，需回頭調整）**：目前不是主要結論；指定影像上的候選框具有一致性，但仍需跨影像驗證。
- **D（bbox 太粗，直接進 SAM 2）**：否。已有貼合零件的 bbox，問題是候選排序而非邊界精度。

下一步若獲准進入 Phase 8，應先在 localization pipeline 中保留多候選並採用中心性、面積或 reference-guided 規則選框；在跨影像驗證前，不應把 `score` 最高等同於定位成功。
