# Ground Truth 引用稽核

## 稽核範圍

以 repository 全域文字搜尋盤點 `ground_truth.csv`、`ground_truth` 與
`Ground Truth`，排除 `.git/`、`venv/`、`output/`、`__pycache__/` 與二進位檔。
本次只建立報告，沒有修改任何既有引用。

## 正式版引用

| 檔案 | 用途 | 判定 |
|---|---|---|
| `utils/ground_truth_loader.py` | 預設讀取 `data/ground_truth.csv` | 正式 |
| `scripts/build_ground_truth.py` | 由 frozen inventory 產生正式 CSV | 正式 |
| `tests/test_ground_truth.py` | 驗證正式 CSV 與 loader | 正式 |
| `docs/ground_truth.md` | Phase 8.1 規格 | 正式 |

本次新增的 `scripts/export_google_sheets_csv.py` 也只接受正式版；若明確傳入
repository-root Legacy 路徑會拋出錯誤。

## Legacy CSV 引用

| 檔案 | 用途 | 風險／後續 |
|---|---|---|
| `generate_mid_report.py` | 直接讀寫 root `ground_truth.csv`，產生舊版資料報告 | 保留為 Legacy 報告工具；不可用於新 batch evaluation |
| `docs/data_status.md` | 描述舊 146 筆資料集與 root CSV | 歷史文件，數字不可視為 Phase 8.1 現況 |
| `docs/progress.md` | 含多個歷史 Ground Truth 記錄 | 歷史時間線，不是執行期來源 |
| `docs/prompt_changes.md` | 歷史規格與產生流程說明 | 歷史文件 |

## 容易誤判但不是 CSV 引用

- `ground_truth/modelXX/stepXX.json` 是 expected-state JSON 規格目錄，不是
  root CSV。
- `utils/current_state_analyzer.py`、`tests/test_compare.py` 與
  `tests/test_compare_reference.py` 目前使用 filename parsing 與上述 JSON，
  沒有直接讀取兩份 CSV。
- `app.py` 中的 Ground Truth 是 JSON 流程說明，不是 CSV path。

## 批次評估狀態

現有 `tests/test_compare_reference.py` 尚未直接用 CSV 驅動；正式 loader 提供
`image_id`、`is_error`、`error_type`、`model_id`、`step_id`、`view_angle`、
`evaluation_scope` 與 schema-compatible aliases，可作為後續 batch adapter。
本階段沒有改寫或執行 Vision pipeline。

## 建議

1. 新增功能只引用 `utils.ground_truth_loader`，不要硬編碼 root CSV。
2. 後續若重構 `generate_mid_report.py`，應新增明確的 Legacy mode，或改由正式
   loader 產生新版報告。
3. Legacy root CSV 暫時保留原位，避免破壞歷史工具；不再建立額外 archive
   副本。
