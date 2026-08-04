"""Deterministic, local-first correction SOP generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PART_NAMES = {
    "PIN_RED_SHORT": "紅色短柱",
    "WHEEL_BLUE_SMALL": "藍色小輪",
    "WHEEL_BLUE_LARGE": "藍色大輪",
    "EYE_BALL": "眼睛零件",
    "PIN_YELLOW": "黃色喇叭形插銷",
}


def _part_name(part_id: str) -> str:
    return PART_NAMES.get(part_id, part_id.replace("_", " ").lower())


def _load_expected(value: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    with Path(value).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("expected_state must contain a JSON object")
    return result


def _step(
    number: int,
    action: str,
    instruction: str,
    parts: list[str],
    reference_step: str,
    preserve: list[str],
) -> dict[str, Any]:
    return {
        "step_number": number,
        "action": action,
        "instruction": instruction,
        "visual_instruction": f"以紅色箭頭標示{instruction}；其餘車體、視角與零件保持不變。",
        "affected_parts": list(parts),
        "preserve_parts": list(preserve),
        "reference_step": reference_step,
        "requires_generated_image": True,
    }


def generate_correction_sop(
    error_reports: list[dict[str, Any]],
    expected_state: dict[str, Any] | str | Path | None,
    current_step: str,
    reference_metadata: dict[str, Any] | None = None,
    instruction_steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate a structured SOP without consulting evaluation ground truth."""
    if not isinstance(error_reports, list):
        raise TypeError("error_reports must be a list")
    expected = _load_expected(expected_state)
    expected_ids = [
        str(item.get("part_id"))
        for item in expected.get("expected_parts", [])
        if isinstance(item, dict) and item.get("part_id")
    ]
    targets = list(dict.fromkeys(str(r.get("part_id", "unknown_part")) for r in error_reports))
    preserve = [part for part in dict.fromkeys(expected_ids) if part not in targets]

    if not error_reports:
        return {
            "repair_scope": "local",
            "source_step_id": current_step,
            "rollback_to_step": None,
            "target_parts": [],
            "steps": [],
            "status": "correct",
            "message": "組裝結果正確，無需修正。",
            "reference_metadata": reference_metadata or {},
        }

    types = {str(report.get("error_type", "uncertain")) for report in error_reports}
    if "criticalerror" in types:
        scope = "full_rollback"
    elif any(bool(report.get("requires_rollback")) for report in error_reports):
        scope = "partial_rollback"
    else:
        scope = "local"

    steps: list[dict[str, Any]] = []
    # A swap is represented by both affected identities.  Never invent the pair.
    swap_parts = [report for report in error_reports if report.get("error_type") == "wrongpart"]
    if len(swap_parts) >= 2:
        pair = list(dict.fromkeys(str(item.get("part_id")) for item in swap_parts))
        if len(pair) >= 2:
            names = " 與 ".join(_part_name(part) for part in pair[:2])
            actions = [
                ("locate", f"在測試圖中定位 {names} 的目前位置"),
                ("remove", f"只拆下 {names}，保留周圍結構"),
                ("swap", f"依正確參考圖交換 {names} 的安裝位置"),
                ("verify", f"確認 {names} 的位置、方向與數量皆符合參考圖"),
            ]
            for action, instruction in actions:
                steps.append(_step(len(steps) + 1, action, instruction, pair[:2], current_step, preserve))

    handled = set(targets if steps else [])
    for report in error_reports:
        part_id = str(report.get("part_id", "unknown_part"))
        if part_id in handled:
            continue
        name = _part_name(part_id)
        error_type = str(report.get("error_type", "uncertain"))
        if error_type == "missingpart":
            actions = [
                ("locate", f"依正確參考圖定位 {name} 的缺口"),
                ("insert", f"取用正確的 {name} 並安裝至缺口"),
                ("verify", f"確認 {name} 的位置、方向與數量正確"),
            ]
        elif error_type == "extrapart":
            actions = [
                ("locate", f"定位多出的 {name}"),
                ("remove", f"移除多出的 {name}，不要拆動相鄰零件"),
                ("verify", f"依參考圖確認 {name} 數量及周圍結構"),
            ]
        elif error_type in {"positionerror", "wrongpart"}:
            actions = [
                ("locate", f"定位位置或身分錯誤的 {name}"),
                ("remove", f"拆下 {name} 並保留周圍結構"),
                ("move", f"依正確參考圖將 {name} 安裝到正確位置與方向"),
                ("verify", f"確認 {name} 與參考圖一致"),
            ]
        elif error_type == "criticalerror":
            actions = [
                ("remove", "回退目前嚴重錯誤的結構，直到上一個已確認正確步驟"),
                ("insert", "依原始說明書重建此步驟，不沿用錯誤結構"),
                ("verify", "從多個視角確認重建結果"),
            ]
        else:
            actions = [
                ("locate", f"人工確認 {name} 的差異位置"),
                ("verify", "對照正確參考圖後再執行修正"),
            ]
        for action, instruction in actions:
            steps.append(_step(len(steps) + 1, action, instruction, [part_id], current_step, preserve))

    return {
        "repair_scope": scope,
        "source_step_id": current_step,
        "rollback_to_step": current_step if scope == "partial_rollback" else ("step01" if scope == "full_rollback" else None),
        "target_parts": targets,
        "steps": steps,
        "status": "repair_required",
        "reference_metadata": reference_metadata or {},
        "instruction_source_available": bool(instruction_steps),
    }
