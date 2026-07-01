# Development Rules

**Project:** Assembly Error Detection and Corrective Guidance System

---

# 目的

本文件旨在建立團隊共同的開發規範，確保所有成員在修改 Prompt、程式、文件或測試流程時，都能保持一致的紀錄方式與版本管理，降低多人協作造成的衝突與資訊遺失。

---

# 一、共同開發原則

## 1. Single Source of Truth

所有模組皆應遵循相同規格。

目前專案規格如下：

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

若修改其中任何一項規格，請同步確認其他模組是否需要更新。

禁止不同模組自行定義不同 JSON Format。

---

## 2. 不直接修改他人程式

若需修改其他成員負責模組：

* 先閱讀程式內容
* 理解修改影響
* 更新 Progress
* Commit 時說明修改原因

避免直接覆蓋他人程式而未留下紀錄。

---

## 3. 每次修改都必須可追蹤

所有重要修改皆需：

* Git Commit
* Progress 紀錄
* Prompt 紀錄（若有）

避免只有程式變更，沒有任何說明。

---

# 二、何時需要更新文件？

## (一) progress.md

當完成以下任一事項時，請更新 Progress：

* 新增功能
* 完成一項模組
* 修改流程
* 修正重大 Bug
* 完成測試
* 發現重要研究結果
* 完成重要里程碑

建議格式：

---

日期

今日目標

今日修改

測試結果

研究發現

下一步規劃

---

Progress 著重於：

「今天完成了什麼」

而非詳細程式修改。

---

## (二) prompt_changes.md

只要修改 Prompt，就必須更新。

包含：

* Prompt 新增規則
* Few-shot 修改
* JSON Output 修改
* Error Type 修改
* Decision Rule 修改
* Constraint 修改

請記錄：

* 修改原因
* 修改內容
* 改善效果
* 已知問題
* 下一版規劃

避免只更新 Prompt 檔案，而沒有留下修改紀錄。

---

## (三) shooting_guide.md

若拍攝方式有調整：

例如：

* 新增角度
* 新增命名規則
* 修改光線要求
* 修改背景要求

請同步更新 Shooting Guide。

---

# 三、Prompt 修改規範

## 不直接覆蓋舊版本

建議版本：

- vision_v1.txt
- vision_v1_1.txt
- vision_v1_2.txt
- vision_v2.txt

保留至少上一版。

若新版效果較差，可快速回復。

---

修改 Prompt 後請確認：

□ Prompt 可正常讀取
□ JSON Output 合法
□ Schema 可驗證
□ current_state_analyzer 可解析
□ test_compare_reference 可正常執行

---

# 四、Schema 修改規範

若修改：

schema/
vision_output_schema.json

請同步確認：

- Prompt
- Analyzer
- Compare
- Annotator

皆已更新。

禁止：

Prompt 與 Schema 不一致。

---

# 五、Python 程式修改規範

修改程式後至少確認：

- 程式可執行
- 沒有 Syntax Error
- 沒有 Import Error
- JSON 可正常 Parse
- Schema Validation 通過

不要提交無法執行的版本。

---

# 六、Git 使用規範

## Commit Frequency

完成一項功能後即可 Commit。

不要累積大量修改再一次 Commit。

建議：

一天至少一次 Commit。

---

## Commit Message

請使用以下格式：

- feat: 新增功能
- fix: 修正錯誤
- refactor: 重構程式
- docs: 文件更新
- test: 新增測試

example：

- feat: add schema validation
- fix: correct wrongpart detection
- docs: update prompt changes
- test: add structured comparison test

避免：

- update
- change
- modify
- test

等無法理解內容的 Commit。

---

## Push

確認：

- 程式正常
- 文件同步
- Commit 完成

再 Push 至 GitHub。

---

# 七、測試規範

若修改會影響辨識結果，請重新測試。

至少記錄：

- Prompt Version
- 測試日期
- Ground Truth
- GPT Result
- Confidence
- TP / TN / FP / FN

若 Accuracy 有明顯變化，請更新 Progress。

---

# 八、Logs 規範

logs/

主要存放：

- Raw GPT Response
- Parsed JSON
- Failed JSON
- Debug Output

logs 僅供本地除錯使用。

除非有特殊需求，請勿提交至 GitHub Repository。

若需保存正式測試結果，請整理後放入：docs/ 或 reports/

而非直接提交所有 Logs。

---

# 九、Output 規範

output/

主要為：

- Annotator Output
- Compare Result
- Visualization

皆屬於程式執行產生。

除正式 Demo 成果外，不建議提交。

---

# 十、檔案命名規範

Prompts：vision_v2.txt

Schema：vision_output_schema.json

Analyzer：current_state_analyzer.py

Testing：test_compare_reference.py

避免：
- new.py
- test_new.py
- final.py
- new_final.py
- new_final2.py

等難以辨識用途的名稱。

---

# 十一、重大修改流程

若修改：

- Prompt JSON Structure
- Schema
- Pipeline
- Folder Structure
- API Interface

請依序完成：

1. 修改程式
↓
2. 更新 Prompt
↓
3. 更新 Schema
↓
4. 更新 Progress
↓
5. 更新 Prompt Change Log（若有）
↓
6. Commit
↓
7. Push

---

# 十二、團隊協作提醒

修改前：先 Pull 最新版本。

修改後：先確認程式可執行。

Push 前：確認沒有遺漏重要文件。

若發生 Merge Conflict：不要直接覆蓋他人內容，請先確認差異後再合併。

---

# 十三、版本管理原則

所有重要修改應保留歷史紀錄。

不要：

* 刪除舊 Prompt
* 刪除測試紀錄
* 覆蓋重要設定

除非確認不再使用。

若需重大重構，請保留可回復版本。

---

# 十四、目標

本專案希望達成：

* 程式容易維護
* Prompt 容易追蹤
* JSON 規格一致
* Git 紀錄清楚
* 每位成員皆可快速理解最新專案狀態

請所有成員遵守以上規範，共同維護專案品質與協作品質。
