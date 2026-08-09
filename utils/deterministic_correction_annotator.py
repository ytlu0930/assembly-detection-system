"""Create deterministic ROI correction overlays; no generative image model."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2

from utils.image_annotator import _validate_bbox


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def annotate_correction(
    *, test_image: str | Path, bbox_evidence: list[dict[str, Any]],
    affected_parts: list[dict[str, Any]], error_type: str,
    output_path: str | Path, requires_manual_review: bool,
    evidence_role: str = "test",
    label_prefix: str = "",
) -> str:
    source = Path(test_image).resolve()
    before = file_sha256(source)
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Could not decode source image: {source}")
    height, width = image.shape[:2]
    ids = [str(item.get("part_id") or "UNKNOWN") for item in affected_parts] or ["UNKNOWN"]
    evidence_by_id = {
        str(item.get("candidate_part_id") or ""): item
        for item in bbox_evidence if str(item.get("role") or "") == evidence_role
    }
    colors = [(0, 0, 255), (0, 165, 255)]
    for index, part_id in enumerate(ids):
        evidence = evidence_by_id.get(part_id)
        if evidence is None:
            continue
        x1, y1, x2, y2 = _validate_bbox(evidence.get("bbox"), width, height)
        color = colors[index % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
        cv2.arrowedLine(image, (max(0, x1 - 120), max(0, y1 - 100)), (x1, y1), color, 5, tipLength=0.18)
        label = f"{label_prefix}:{part_id}" if label_prefix else part_id
        cv2.putText(image, label, (x1, max(28, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    panel_height = 180
    canvas = cv2.copyMakeBorder(image, 0, panel_height, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    confidence = min((float(item.get("confidence") or 0.0) for item in affected_parts), default=0.0)
    located_count = sum(part_id in evidence_by_id for part_id in ids)
    lines = [
        f"Affected: {', '.join(ids)}",
        f"Error: {error_type}   Confidence: {confidence:.2f}",
        f"Evidence frame: {evidence_role}; localized boxes: {located_count}/{len(ids)}",
        "Action: verify highlighted ROI and correct the identified component." if located_count else "No aligned bbox in this frame; verify paired ROI manually.",
        "MANUAL REVIEW REQUIRED" if requires_manual_review else "Rule-engine result",
    ]
    for row, line in enumerate(lines[:4]):
        cv2.putText(canvas, line, (24, height + 30 + row * 29), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, lines[4], (24, height + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (30, 30, 30), 2, cv2.LINE_AA)
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        raise OSError(f"Failed to write annotation: {target}")
    if file_sha256(source) != before:
        raise RuntimeError("Source image was modified")
    return str(target)
