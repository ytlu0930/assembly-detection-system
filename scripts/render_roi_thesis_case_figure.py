"""Render deterministic four-panel ROI experiment case figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def render_case_figure(
    *, reference_image: Path, test_image: Path, roi_image: Path,
    annotated_image: Path, output_path: Path, title: str,
) -> str:
    paths = (reference_image, test_image, roi_image, annotated_image)
    labels = ("(a) Correct Reference", "(b) Test Image", "(c) ROI Localization", "(d) Final Annotated Correction")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for axis, path, label in zip(axes.flat, paths, labels):
            image = cv2.imread(str(path))
            if image is None: raise ValueError(f"Could not decode thesis panel: {path}")
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); axis.set_title(label, fontsize=12); axis.axis("off")
        fig.suptitle(title, fontsize=15, fontweight="bold"); fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    except ModuleNotFoundError:
        import numpy as np
        canvas = np.full((2400, 2880, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, title, (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 20, 20), 4)
        for index, (path, label) in enumerate(zip(paths, labels)):
            image = cv2.imread(str(path))
            if image is None: raise ValueError(f"Could not decode thesis panel: {path}")
            image = cv2.resize(image, (1320, 980), interpolation=cv2.INTER_AREA)
            x, y = 80 + (index % 2) * 1400, 180 + (index // 2) * 1100
            canvas[y:y + 980, x:x + 1320] = image
            cv2.putText(canvas, label, (x, y + 1030), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
        if not cv2.imwrite(str(output_path), canvas): raise OSError(f"Failed to write {output_path}")
    return str(output_path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--annotated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    print(render_case_figure(reference_image=args.reference, test_image=args.test, roi_image=args.roi, annotated_image=args.annotated, output_path=args.output, title=args.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
