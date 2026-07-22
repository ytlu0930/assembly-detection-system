# Dataset Audit and Freeze

## 目的與掃描範圍

`scripts/audit_dataset.py` 對 `input/` 與 `regression_subset/` 建立唯讀 inventory、檔名驗證、SHA-256 重複檢查、correct reference 配對檢查，以及可重現的 freeze manifest。工具不會重新拍攝、改名、搬移、編輯或刪除來源檔案。

## 命名解析規則

標準格式為：

```text
<model_id>_<step_id>_<label>-<variant>_<view_angle>_<sequence_index>.<ext>
```

例如 `model03_step03_wrongpart-A01_right_01.jpg`。檔名必須正好有五個 underscore fields；view 必須是 `top`、`bottom`、`front`、`back`、`left` 或 `right`；sequence 必須是數字；合法副檔名沿用現有 Vision pipeline 的 `.jpg`、`.jpeg`、`.png`。

Error type mapping：

| 原始 label | 標準 error_type |
|---|---|
| `correct` | `correct` |
| `positionerror` | `position` |
| `orientationerror` | `orientation` |
| `missingpart` | `missing` |
| `extrapart` | `extra` |
| `wrongpart` | `wrongpart` |
| 其他 | `unknown`（保留原始 label） |

實際資料含 `criticalerror`，因此列為 unknown，沒有改寫成 position、orientation 或 wrongpart。

## Correct reference matching

規則直接沿用 `tests/test_compare_reference.py`：先搜尋 `input/normal/<model_id>_<step_id>/`，再搜尋 `input/normal/`；pattern 為 `<model_id>_<step_id>_correct-*_<view_angle>_*`。候選排序後優先選 `correct-01`，否則選第一個。audit 會記錄 candidate count、pattern 與 selection rule，不呼叫 GPT Vision API。

## Duplicate 與 freeze 定義

完全相同的 SHA-256 才視為 duplicate。分類包括 `expected_regression_copy`、`duplicate_within_input`、`duplicate_within_regression`、`cross_source_duplicate`，不會刪除任何檔案。`freeze_manifest.json` 記錄來源、相對路徑、大小與 SHA-256；freeze 是基準清單，不會鎖定 Windows 權限。

## 2026-07-22 實際結果

- 158 dataset files：input 148、regression subset 10
- 156 個合法 `.jpg`；2 個 `.jpg_` 的內容 magic bytes 為 JPEG，但副檔名不合法
- valid filenames 150；invalid 8
- correct 61（其中 59 個可由合法副檔名直接辨識）；error 97
- position 12、orientation 0、missing 36、extra 15、wrongpart 28、unknown criticalerror 6
- step coverage：position 1、orientation 0、missing 2、extra 2、wrongpart 2、unknown 1
- duplicate 12 groups／24 participating files：10 expected regression-copy groups、2 input-internal groups
- missing correct references 0
- targets：correct 30、error 80、missing 20 已達成；position 20、orientation 20、extra 20 未達成

正式 audit 位於 `output/dataset_audit/20260722_170322/`；正式 freeze 位於 `output/dataset_audit/20260722_170328/`。每個 run 包含 `dataset_inventory.csv`、`dataset_summary.json`、三份異常清單與 `run_summary.json`，freeze run 另含 `freeze_manifest.json`。

## 已知限制

- unknown label 需要人工決定是否納入未來正式 taxonomy。
- 兩個 `.jpg_` 檔案需人工決定是否在未來版本修名；本次按限制保留原名。
- 沒有任何 error type 涵蓋至少三個 step。
- audit 沒有 ground-truth bbox 或影像語意正確性判定。

## 重新執行與驗證

```powershell
& .\venv\Scripts\python.exe scripts\audit_dataset.py
& .\venv\Scripts\python.exe scripts\audit_dataset.py --freeze
```

若要驗證資料集是否變更，重新執行 `--freeze`，比較兩個 freeze manifest 的 `total_files`、`total_bytes` 與逐檔 `source_root + relative_path + sha256 + file_size_bytes`。只要任一值不同，就代表凍結基準後來源曾發生變更。
