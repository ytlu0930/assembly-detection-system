from scripts.render_checklist_confusion_matrix import confusion_counts, render


def test_confusion_counts_keep_uncertain_separate(tmp_path):
    rows = [
        {"gt_status": "NORMAL", "predicted_status": "PASS"},
        {"gt_status": "NORMAL", "predicted_status": "FAIL"},
        {"gt_status": "MISMATCH", "predicted_status": "PASS"},
        {"gt_status": "MISMATCH", "predicted_status": "FAIL"},
        {"gt_status": "MISMATCH", "predicted_status": "UNCERTAIN"},
    ]
    assert confusion_counts(rows) == {"TN": 1, "FP": 1, "FN": 1, "TP": 1, "UNCERTAIN": 1}
    counts = render(rows, tmp_path / "matrix.png", tmp_path / "matrix.csv")
    assert counts["UNCERTAIN"] == 1
    assert (tmp_path / "matrix.png").is_file() and (tmp_path / "matrix.csv").is_file()
