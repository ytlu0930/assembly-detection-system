# Ground Truth CSV 版本比較

## 結論

`data/ground_truth.csv` 是目前唯一正式、可供評估與匯出的 Ground Truth。
Repository root 的 `ground_truth.csv` 是 146 筆舊版資料，只保留作為歷史紀錄，
不得再作為新 batch evaluation 的輸入。

本次沒有移動、覆寫或刪除任一檔案，也沒有建立
`data/archive/ground_truth_legacy.csv`。原因是 root 檔案已有明確的 Legacy
定位，再製作第三份副本只會增加版本混淆。

## 結構與內容比較

| 項目 | Legacy `ground_truth.csv` | Formal `data/ground_truth.csv` |
|---|---|---|
| 筆數 | 146 | 158 |
| 欄位數 | 6 | 18 |
| 唯一鍵 | 實際上依賴 `image_name` | 含 split 的 `image_id`／`image_path` |
| Split | 未記錄 | `input`、`regression_subset` |
| 錯誤旗標 | 無獨立欄位 | `is_error` |
| Taxonomy | schema/raw 名稱，如 `missingpart` | 正式名稱，如 `missing` |
| 評估範圍 | 無 | `evaluation_scope` |
| 追溯資料 | 無 hash | raw label、inventory 狀態、SHA-256 |
| Batch adapter | 無正式 loader | `utils/ground_truth_loader.py` |

Legacy 欄位：

```text
image_name, model_id, step_id, view_angle, expected_error_type, variant_id
```

Formal 欄位：

```text
image_id, image_name, image_path, model_id, step_id, view_angle, is_error,
error_type, schema_error_type, error_detail, evaluation_scope, source_split,
raw_label, filename_valid, inventory_filename_valid,
inventory_validation_errors, duplicate_group_id, sha256
```

兩版共同的 146 個 filename 經 taxonomy normalization 後：

- error type 差異：0
- model_id 差異：0
- step_id 差異：0
- view_angle 差異：0

Formal 版多出兩個 `.jpg_` 來源檔記錄，並保留 10 筆
`regression_subset` 複本。Formal 共有 148 個不同 filename，但有 158 個不同
`image_id`；因此 filename 不能當作唯一鍵。

## Taxonomy 差異

| Legacy/raw | Formal | 說明 |
|---|---|---|
| `correct` | `correct` | `is_error=false` |
| `positionerror` | `position` | in scope |
| `missingpart` | `missing` | in scope |
| `extrapart` | `extra` | in scope |
| `wrongpart` | `wrongpart` | in scope |
| `criticalerror` | `criticalerror` | schema 支援，保持獨立類別 |
| `orientationerror` | `orientation` | 目前 0 筆，out of scope |

Formal 版計數為 correct 61、position 12、missing 36、extra 15、
wrongpart 28、criticalerror 6、orientation 0。全部 158 筆現有資料皆為
in scope；orientation 只存在於 taxonomy，沒有實際列。

## 使用規則

- 新程式與匯出工作一律透過 `utils/ground_truth_loader.py` 讀取
  `data/ground_truth.csv`。
- 跨 split 查詢必須使用完整 `image_id`。
- 若 filename 在兩個 split 中重複，loader 的 filename fallback 會明確拒絕，
  避免取到錯誤列。
- Legacy CSV 不得默默加入新評估流程；若歷史報告仍需使用，必須在文件中標明
  `Legacy` 與 146 筆限制。
