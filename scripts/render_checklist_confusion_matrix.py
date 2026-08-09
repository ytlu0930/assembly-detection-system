"""Render a publication-ready resolved-only checklist confusion matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def confusion_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"TN": 0, "FP": 0, "FN": 0, "TP": 0, "UNCERTAIN": 0}
    for row in rows:
        predicted = str(row.get("predicted_status") or "").upper()
        mismatch = str(row.get("gt_status") or "").upper() == "MISMATCH"
        if predicted == "UNCERTAIN":
            counts["UNCERTAIN"] += 1
        elif mismatch and predicted == "FAIL": counts["TP"] += 1
        elif mismatch and predicted == "PASS": counts["FN"] += 1
        elif not mismatch and predicted == "FAIL": counts["FP"] += 1
        elif not mismatch and predicted == "PASS": counts["TN"] += 1
    return counts


def render(rows: list[dict[str, Any]], output_png: Path, output_csv: Path) -> dict[str, int]:
    counts = confusion_counts(rows)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["actual", "predicted_pass", "predicted_fail"])
        writer.writeheader()
        writer.writerows([
            {"actual": "GT NORMAL", "predicted_pass": counts["TN"], "predicted_fail": counts["FP"]},
            {"actual": "GT MISMATCH", "predicted_pass": counts["FN"], "predicted_fail": counts["TP"]},
        ])
    matrix = [[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]]
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, max(max(row) for row in matrix)))
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                label = (("TN", "FP"), ("FN", "TP"))[y][x]
                ax.text(x, y, f"{label}\n{value}", ha="center", va="center", fontsize=15, fontweight="bold")
        ax.set_xticks([0, 1], ["PASS", "FAIL"]); ax.set_yticks([0, 1], ["GT NORMAL", "GT MISMATCH"])
        ax.set_xlabel("Predicted checklist status"); ax.set_ylabel("Ground truth component status")
        ax.set_title(f"ROI Checklist Confusion Matrix\nResolved checks only; UNCERTAIN={counts['UNCERTAIN']}")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(output_png, dpi=300); plt.close(fig)
    except ModuleNotFoundError:
        import cv2
        import numpy as np
        canvas = np.full((1620, 1920, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, "ROI Checklist Confusion Matrix", (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 20, 20), 4)
        cv2.putText(canvas, f"Resolved only; UNCERTAIN={counts['UNCERTAIN']}", (530, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)
        labels = (("TN", "FP"), ("FN", "TP"))
        for y in range(2):
            for x in range(2):
                x1, y1 = 450 + x * 550, 300 + y * 500
                intensity = 245 - int(140 * matrix[y][x] / max(1, max(max(row) for row in matrix)))
                cv2.rectangle(canvas, (x1, y1), (x1 + 500, y1 + 450), (255, intensity, intensity), -1)
                cv2.rectangle(canvas, (x1, y1), (x1 + 500, y1 + 450), (80, 80, 80), 3)
                cv2.putText(canvas, f"{labels[y][x]}  {matrix[y][x]}", (x1 + 145, y1 + 250), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (20, 20, 20), 4)
        cv2.putText(canvas, "PASS", (620, 1350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
        cv2.putText(canvas, "FAIL", (1180, 1350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
        cv2.putText(canvas, "GT NORMAL", (90, 560), cv2.FONT_HERSHEY_SIMPLEX, .9, (20, 20, 20), 2)
        cv2.putText(canvas, "GT MISMATCH", (60, 1060), cv2.FONT_HERSHEY_SIMPLEX, .9, (20, 20, 20), 2)
        if not cv2.imwrite(str(output_png), canvas): raise OSError(f"Failed to write {output_png}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    print(render(rows, args.output_png, args.output_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
