# Google Sheets CSV 匯入說明

## 單一資料來源

所有匯出都來自正式 `data/ground_truth.csv`。Repository root 的
`ground_truth.csv` 是 146 筆 Legacy 版本，不得匯入作為正式 Ground Truth，
也不得用於新 batch evaluation。

執行：

```powershell
.\venv\Scripts\python.exe scripts\export_google_sheets_csv.py
.\venv\Scripts\python.exe scripts\export_google_sheets_csv.py --check-only
```

自訂輸出位置：

```powershell
.\venv\Scripts\python.exe scripts\export_google_sheets_csv.py `
  --output-dir data/google_sheets_import
```

`--check-only` 不寫入檔案，只確認既有輸出完整且內容與正式 Ground Truth
重新計算的結果完全一致。

## 建議的 Google Sheets 分頁

依序建立以下分頁並匯入對應 CSV：

| 分頁 | CSV | 用途 |
|---|---|---|
| Ground Truth | `01_ground_truth.csv` | 158 筆正式標註與人工覆核欄位 |
| Dataset Summary | `02_dataset_summary.csv` | 類別數量與目標達成狀態 |
| Step Coverage | `03_step_coverage.csv` | 各類別的 unique model-step coverage |
| Batch Results | `04_batch_results_template.csv` | 後續批次結果表頭 |
| Failure Analysis | `05_failure_analysis_template.csv` | 後續失敗分析表頭 |

Google Sheets 中選擇「檔案 → 匯入 → 上傳」，每份 CSV 選擇「插入新工作表」。
檔案皆為 UTF-8、單一 header、無公式、無合併儲存格，也沒有 Python
list/dict、`NaN` 或 `None` 字串。

## Ground Truth 使用注意事項

- `image_id` 是含 `input/` 或 `regression_subset/` 的唯一鍵。
- `file_name` 只是顯示欄位，不保證唯一；目前有 10 個 filename 同時存在於
  兩個 split。
- `image_path` 是專案相對路徑，不是本機絕對路徑。
- `review_status`、`reviewer`、`review_notes` 是預留的人工覆核欄位。
- 建議不要修改 `image_id`、`image_path`、`source_split`、`model_id`、
  `step_id`、`view_angle`、`is_error`、`error_type` 與
  `evaluation_scope` 等核心欄位。

## Taxonomy 與資料限制

- correct 61，all error 97，已達成 30／80 aggregate targets。
- missing 36 已達 20；position 12、extra 15 未達 20。
- position、missing、extra 的 step coverage 目標為至少 3 個不同的
  `model_id-step_id` 組合。
- wrongpart 與 criticalerror 沒有既定的 20 筆／3-step 數量目標，因此標為
  `not_applicable`，但兩者都在評估範圍內。
- criticalerror 保持獨立 taxonomy，不映射到其他類別。
- orientation 目前 0 筆且 schema 不支援，因此是 out of scope；匯出器不會
  製造 orientation 資料。

Batch Results 與 Failure Analysis 目前只有 header，刻意不填入假資料；待日後
實際執行 pipeline 後再逐列追加結果。
