"""
correction_sop_generator.py

將 pipeline_smoke_test.py 產生的 results.json 轉成結構化局部修正 SOP。
本模組不呼叫 Vision API、Grounding DINO 或圖片生成 API。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
PART_LIBRARY_PATH = PROJECT_ROOT / "config" / "part_library.json"
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "output"
    / "pipeline"
    / "error_aware_localization_smoke_test"
)

PART_CONFIDENCE_THRESHOLD = 0.85
LOCALIZATION_DETECTION_THRESHOLD = 0.18
LOCALIZATION_SELECTION_THRESHOLD = 0.55

ERROR_ZH = {
    "correct": "組裝正確",
    "missingpart": "缺少零件",
    "extrapart": "多餘零件",
    "wrongpart": "錯誤零件",
    "positionerror": "位置或方向錯誤",
    "criticalerror": "嚴重組裝錯誤",
    "uncertain": "無法確定",
}

POSITION_ZH = {
    "LEFT": "左側",
    "RIGHT": "右側",
    "CENTER": "中央",
    "TOP": "上方",
    "BOTTOM": "下方",
    "FRONT": "前方",
    "BACK": "後方",
}

ORIENTATION_ZH = {
    "VERTICAL": "垂直",
    "HORIZONTAL": "水平",
}

ACTION_TITLES = {
    "inspect_target": "確認目標區域",
    "prepare_part": "準備正確零件",
    "locate_installation_point": "確認安裝位置",
    "insert_part": "安裝零件",
    "remove_part": "移除零件",
    "detach_part": "鬆開零件",
    "replace_part": "更換零件",
    "reposition_part": "調整位置",
    "reorient_part": "調整方向",
    "disassemble_local_area": "拆除局部結構",
    "rebuild_local_area": "重新組裝局部結構",
    "verify_local_result": "確認局部修正",
    "compare_reference": "比對正確參考圖",
    "manual_review": "人工確認",
    "retake_photo": "重新拍照",
    "rerun_detection": "再次執行 AI 檢測",
    "finish": "完成",
}


@dataclass
class Evidence:
    source_image: Optional[str] = None
    reference_image: Optional[str] = None
    annotated_image: Optional[str] = None
    bbox: Optional[list[float]] = None
    localization_role: Optional[str] = None
    detection_score: Optional[float] = None
    selection_score: Optional[float] = None


@dataclass
class SOPStep:
    step_no: int
    action: str
    title: str
    instruction: str
    target_part_id: Optional[str]
    target_part_name: str
    error_type: str
    expected_position: Optional[str] = None
    expected_orientation: Optional[str] = None
    requires_image_generation: bool = True
    image_generation_mode: str = "edit_previous_image"
    verification: Optional[str] = None
    safety_note: Optional[str] = None
    evidence: Evidence = field(default_factory=Evidence)


@dataclass
class CorrectionSOP:
    schema_version: str
    model_id: str
    step_id: str
    image_name: str
    is_error: bool
    overall_error_type: str
    overall_error_type_zh: str
    summary: str
    source_results_json: str
    test_image_path: Optional[str]
    reference_image_path: Optional[str]
    expected_state_path: Optional[str]
    part_identity_reliable: bool
    localization_reliable: bool
    requires_manual_review: bool
    correction_plan: list[SOPStep]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        normalized_steps = []
        targets: list[str] = []
        for raw in payload["correction_plan"]:
            part_id = raw.get("target_part_id")
            affected = [value for value in str(part_id).split("|") if value] if part_id else []
            for value in affected:
                if value not in targets:
                    targets.append(value)
            raw.update({
                "step_number": raw["step_no"],
                "visual_instruction": raw["instruction"],
                "affected_parts": affected,
                "preserve_parts": [],
                "requires_generated_image": raw["requires_image_generation"],
            })
            normalized_steps.append(raw)
        payload.update({
            "repair_scope": "full_rollback" if self.overall_error_type == "criticalerror" else "local",
            "source_step_id": self.step_id,
            "rollback_to_step": "step01" if self.overall_error_type == "criticalerror" else None,
            "target_parts": targets,
            "steps": normalized_steps,
        })
        payload["correction_plan"] = normalized_steps
        return payload


class CorrectionSOPGenerator:
    def __init__(
        self,
        part_library_path: Path = PART_LIBRARY_PATH,
        part_confidence_threshold: float = PART_CONFIDENCE_THRESHOLD,
        localization_detection_threshold: float = LOCALIZATION_DETECTION_THRESHOLD,
        localization_selection_threshold: float = LOCALIZATION_SELECTION_THRESHOLD,
    ) -> None:
        self.part_library = self._load_json(part_library_path)
        self.part_confidence_threshold = part_confidence_threshold
        self.localization_detection_threshold = localization_detection_threshold
        self.localization_selection_threshold = localization_selection_threshold

    def generate_from_results(self, results_json_path: str | Path) -> CorrectionSOP:
        results_path = Path(results_json_path).expanduser().resolve()
        payload = self._load_json(results_path)

        vision_result = payload.get("vision_result", {})
        model_response = vision_result.get("model_response", {})
        file_info = vision_result.get("file_info", {})

        model_id = str(model_response.get("model_id") or file_info.get("model_id") or "")
        step_id = str(model_response.get("step_id") or file_info.get("step_id") or "")
        image_name = str(file_info.get("image_name") or "")

        test_image = self._as_text(payload.get("test_image_path"))
        reference_image = self._as_text(payload.get("reference_image_path"))
        expected_state_path = self._resolve_expected_state_path(payload, model_id, step_id)
        expected_state = self._load_json(expected_state_path)

        error_parts = payload.get("error_reports") or payload.get("error_parts")
        if not isinstance(error_parts, list):
            error_parts = self._extract_error_parts(model_response)

        overall_error_type = str(
            model_response.get("overall_error_type", "uncertain")
        ).lower()
        is_error = bool(model_response.get("is_error", False))

        if not is_error or overall_error_type == "correct":
            return CorrectionSOP(
                schema_version="1.0",
                model_id=model_id,
                step_id=step_id,
                image_name=image_name,
                is_error=False,
                overall_error_type="correct",
                overall_error_type_zh=ERROR_ZH["correct"],
                summary="AI 判定目前組裝正確，無須產生修正教學。",
                source_results_json=str(results_path),
                test_image_path=test_image,
                reference_image_path=reference_image,
                expected_state_path=str(expected_state_path),
                part_identity_reliable=True,
                localization_reliable=True,
                requires_manual_review=False,
                correction_plan=[],
                warnings=[],
            )

        localization = payload.get("localization", {})
        strategy = payload.get("localization_strategy", {})
        evidence = self._build_evidence(
            localization,
            strategy,
            test_image,
            reference_image,
        )

        steps: list[SOPStep] = []
        warnings: list[str] = []
        identity_flags: list[bool] = []

        if not error_parts:
            steps.extend(
                self._uncertain_steps(
                    overall_error_type,
                    "unknown_part",
                    "標記區域",
                    evidence,
                )
            )
            warnings.append("Vision 判定為錯誤，但沒有可用的 detected_parts。")
            identity_flags.append(False)
        else:
            wrong_parts = [item for item in error_parts if isinstance(item, dict) and str(item.get("error_type", overall_error_type)).lower() == "wrongpart"]
            if len(wrong_parts) >= 2:
                pair_ids = [str(item.get("part_id", "unknown_part")) for item in wrong_parts[:2]]
                identity_flags.extend(
                    self._part_identity_reliable(part_id, self._to_float(item.get("confidence")))
                    for part_id, item in zip(pair_ids, wrong_parts[:2])
                )
                steps.append(self._step(
                    "swap_parts",
                    f"Swap the positions of {pair_ids[0]} and {pair_ids[1]} to match the correct reference.",
                    "|".join(pair_ids),
                    f"{pair_ids[0]} and {pair_ids[1]}",
                    "wrongpart",
                    evidence,
                    verification="Verify both parts match the reference positions without changing surrounding bricks.",
                ))
                error_parts = [item for item in error_parts if item not in wrong_parts[:2]]
            for index, part in enumerate(error_parts):
                if not isinstance(part, dict):
                    continue

                error_type = str(
                    part.get("error_type", overall_error_type)
                ).lower()
                part_id = str(part.get("part_id", "unknown_part"))
                confidence = self._to_float(part.get("confidence"))
                expected_part = self._find_expected_part(expected_state, part_id)

                identity_reliable = self._part_identity_reliable(
                    part_id,
                    confidence,
                )
                identity_flags.append(identity_reliable)

                part_name = self._part_name(
                    part_id,
                    error_type,
                    identity_reliable,
                )

                current_evidence = self._build_evidence(
                    part.get("localization", {}) if isinstance(part.get("localization"), dict) else {},
                    part.get("localization_strategy", {}) if isinstance(part.get("localization_strategy"), dict) else {},
                    test_image,
                    reference_image,
                )

                if not identity_reliable:
                    warnings.append(
                        f"{part_id} 的零件身分可信度不足，SOP 使用泛化名稱。"
                    )

                steps.extend(
                    self._steps_for_error(
                        error_type=error_type,
                        part_id=part_id,
                        part_name=part_name,
                        expected_part=expected_part,
                        evidence=current_evidence,
                        identity_reliable=identity_reliable,
                    )
                )

        localization_reliable = self._localization_reliable(localization)
        part_identity_reliable = bool(identity_flags) and all(identity_flags)
        requires_manual_review = (
            not part_identity_reliable
            or not localization_reliable
            or overall_error_type == "uncertain"
        )

        if not localization_reliable:
            warnings.append(
                "Localization 分數未達可靠門檻，bbox 僅供參考。"
            )

        if requires_manual_review:
            steps.append(
                self._step(
                    "manual_review",
                    "在實際拆裝或圖片生成前，人工確認目標零件與標示位置。",
                    None,
                    "目標區域",
                    overall_error_type,
                    evidence,
                    requires_image_generation=False,
                    image_generation_mode="none",
                    verification="確認 Vision、Ground Truth 與標示位置一致。",
                )
            )

        steps.extend(
            self._common_finish_steps(
                overall_error_type,
                test_image,
                reference_image,
            )
        )
        self._renumber(steps)

        return CorrectionSOP(
            schema_version="1.0",
            model_id=model_id,
            step_id=step_id,
            image_name=image_name,
            is_error=True,
            overall_error_type=overall_error_type,
            overall_error_type_zh=ERROR_ZH.get(
                overall_error_type,
                overall_error_type,
            ),
            summary=(
                f"{model_id} {step_id} 偵測到"
                f"「{ERROR_ZH.get(overall_error_type, overall_error_type)}」，"
                f"共 {len(error_parts)} 個錯誤項目。"
            ),
            source_results_json=str(results_path),
            test_image_path=test_image,
            reference_image_path=reference_image,
            expected_state_path=str(expected_state_path),
            part_identity_reliable=part_identity_reliable,
            localization_reliable=localization_reliable,
            requires_manual_review=requires_manual_review,
            correction_plan=steps,
            warnings=warnings,
        )

    def save(self, sop: CorrectionSOP, output_dir: str | Path) -> tuple[Path, Path]:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)

        json_path = output / "correction_sop.json"
        md_path = output / "correction_sop.md"

        json_path.write_text(
            json.dumps(
                sop.to_dict(),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        md_path.write_text(self.to_markdown(sop), encoding="utf-8")
        return json_path, md_path

    def to_markdown(self, sop: CorrectionSOP) -> str:
        lines = [
            "# 積木局部修正 SOP",
            "",
            f"- Model：`{sop.model_id}`",
            f"- Step：`{sop.step_id}`",
            f"- Image：`{sop.image_name}`",
            f"- Result：{sop.overall_error_type_zh}",
            f"- Part identity reliable：`{sop.part_identity_reliable}`",
            f"- Localization reliable：`{sop.localization_reliable}`",
            f"- Manual review required：`{sop.requires_manual_review}`",
            "",
            f"> {sop.summary}",
            "",
            "## 修正步驟",
            "",
        ]

        if not sop.correction_plan:
            lines.append("無須修正。")
        else:
            for step in sop.correction_plan:
                lines.extend(
                    [
                        f"### Step {step.step_no}｜{step.title}",
                        "",
                        f"- Action token：`{step.action}`",
                        f"- Target：{step.target_part_name}",
                        f"- Instruction：{step.instruction}",
                        f"- Generate image：`{step.requires_image_generation}`",
                    ]
                )
                if step.expected_position:
                    lines.append(f"- Expected position：{step.expected_position}")
                if step.expected_orientation:
                    lines.append(
                        f"- Expected orientation：{step.expected_orientation}"
                    )
                if step.evidence.bbox:
                    lines.append(f"- BBox：`{step.evidence.bbox}`")
                if step.verification:
                    lines.append(f"- Verification：{step.verification}")
                if step.safety_note:
                    lines.append(f"- Note：{step.safety_note}")
                lines.append("")

        if sop.warnings:
            lines.extend(["## 警告", ""])
            lines.extend(f"- {warning}" for warning in sop.warnings)

        return "\n".join(lines)

    def _steps_for_error(
        self,
        *,
        error_type: str,
        part_id: str,
        part_name: str,
        expected_part: Optional[dict[str, Any]],
        evidence: Evidence,
        identity_reliable: bool,
    ) -> list[SOPStep]:
        position, orientation = self._expected_text(expected_part)

        if error_type == "missingpart":
            target = (
                part_name
                if identity_reliable
                else "Ground Truth 指定的缺少零件"
            )
            return [
                self._step(
                    "prepare_part",
                    f"準備一個正確的 {target}。",
                    part_id,
                    target,
                    error_type,
                    evidence,
                    position,
                    orientation,
                    verification="確認顏色、尺寸、形狀與孔位。",
                ),
                self._step(
                    "locate_installation_point",
                    f"查看正確參考圖中的標示區域，確認 {target} 的安裝位置。",
                    part_id,
                    target,
                    error_type,
                    evidence,
                    position,
                    orientation,
                    verification="此 bbox 來自正確參考圖。",
                ),
                self._step(
                    "insert_part",
                    self._insert_instruction(target, position, orientation),
                    part_id,
                    target,
                    error_type,
                    evidence,
                    position,
                    orientation,
                    verification="確認零件完全插入且周圍零件未位移。",
                ),
                self._step(
                    "verify_local_result",
                    f"確認 {target} 的數量、位置與方向皆符合參考圖。",
                    part_id,
                    target,
                    error_type,
                    evidence,
                    position,
                    orientation,
                    verification="局部輪廓應與參考圖一致。",
                ),
            ]

        if error_type == "extrapart":
            target = (
                part_name
                if identity_reliable
                else "標示區域中的多餘零件"
            )
            return [
                self._step(
                    "inspect_target",
                    f"確認 {target} 不存在於正確參考圖。",
                    part_id,
                    target,
                    error_type,
                    evidence,
                    verification="至少再次確認一次零件身分。",
                ),
                self._step(
                    "remove_part",
                    f"沿原連接方向輕輕移除 {target}。",
                    part_id,
                    target,
                    error_type,
                    evidence,
                    verification="避免拉動周圍正確零件。",
                    safety_note="若零件身分不確定，先停止拆除。",
                ),
                self._step(
                    "verify_local_result",
                    "確認拆除區域的零件數量與輪廓符合正確參考圖。",
                    part_id,
                    target,
                    error_type,
                    evidence,
                ),
            ]

        if error_type == "wrongpart":
            target = (
                part_name
                if identity_reliable
                else "標示區域中的錯誤零件"
            )
            return [
                self._step(
                    "inspect_target",
                    f"確認 {target} 的顏色、大小或形狀與參考圖不符。",
                    part_id,
                    target,
                    error_type,
                    evidence,
                ),
                self._step(
                    "remove_part",
                    f"移除目前安裝錯誤的 {target}。",
                    part_id,
                    target,
                    error_type,
                    evidence,
                ),
                self._step(
                    "prepare_part",
                    "依 part library 與參考圖準備正確替換零件。",
                    part_id,
                    "正確替換零件",
                    error_type,
                    evidence,
                    position,
                    orientation,
                ),
                self._step(
                    "replace_part",
                    self._insert_instruction(
                        "正確替換零件",
                        position,
                        orientation,
                    ),
                    part_id,
                    "正確替換零件",
                    error_type,
                    evidence,
                    position,
                    orientation,
                ),
                self._step(
                    "verify_local_result",
                    "確認替換後零件與周圍零件連接穩固。",
                    part_id,
                    "正確替換零件",
                    error_type,
                    evidence,
                ),
            ]

        if error_type == "positionerror":
            return [
                self._step(
                    "inspect_target",
                    f"比對 {part_name} 與正確參考圖的位置及方向差異。",
                    part_id,
                    part_name,
                    error_type,
                    evidence,
                    position,
                    orientation,
                ),
                self._step(
                    "detach_part",
                    f"稍微鬆開 {part_name}，不要拆動周圍正確零件。",
                    part_id,
                    part_name,
                    error_type,
                    evidence,
                ),
                self._step(
                    "reposition_part",
                    f"將 {part_name} 調整至{position or '參考圖所示位置'}。",
                    part_id,
                    part_name,
                    error_type,
                    evidence,
                    position,
                    orientation,
                ),
                self._step(
                    "reorient_part",
                    f"將 {part_name} 的方向調整為"
                    f"{orientation or '參考圖所示方向'}。",
                    part_id,
                    part_name,
                    error_type,
                    evidence,
                    position,
                    orientation,
                ),
                self._step(
                    "verify_local_result",
                    "重新壓緊連接點並確認位置與方向。",
                    part_id,
                    part_name,
                    error_type,
                    evidence,
                ),
            ]

        if error_type == "criticalerror":
            return [
                self._step(
                    "inspect_target",
                    "停止施力並檢查標示區域是否變形、卡死或嚴重偏離。",
                    part_id,
                    part_name,
                    error_type,
                    evidence,
                    safety_note="嚴重錯誤不得直接強行扳回。",
                ),
                self._step(
                    "disassemble_local_area",
                    "拆除受影響的局部結構，直到回到穩定且可辨識狀態。",
                    part_id,
                    part_name,
                    error_type,
                    evidence,
                ),
                self._step(
                    "rebuild_local_area",
                    "依正確參考圖重新組裝該局部區域。",
                    part_id,
                    part_name,
                    error_type,
                    evidence,
                ),
                self._step(
                    "verify_local_result",
                    "確認局部結構恢復穩定且沒有新增其他錯誤。",
                    part_id,
                    part_name,
                    error_type,
                    evidence,
                ),
            ]

        return self._uncertain_steps(
            error_type,
            part_id,
            part_name,
            evidence,
        )

    def _uncertain_steps(
        self,
        error_type: str,
        part_id: str,
        part_name: str,
        evidence: Evidence,
    ) -> list[SOPStep]:
        return [
            self._step(
                "manual_review",
                "AI 無法可靠確認錯誤零件或位置，請先人工比對測試圖與參考圖。",
                part_id,
                part_name,
                error_type,
                evidence,
                requires_image_generation=False,
                image_generation_mode="none",
            )
        ]

    def _common_finish_steps(
        self,
        error_type: str,
        test_image: Optional[str],
        reference_image: Optional[str],
    ) -> list[SOPStep]:
        evidence = Evidence(
            source_image=test_image,
            reference_image=reference_image,
        )
        return [
            self._step(
                "compare_reference",
                "將修正後模型與正確參考圖進行整體比對。",
                None,
                "整體組裝物",
                error_type,
                evidence,
                verification="確認零件數量、位置、方向與顏色一致。",
            ),
            self._step(
                "retake_photo",
                "依專案拍攝規範重新拍攝修正後模型。",
                None,
                "整體組裝物",
                error_type,
                evidence,
                requires_image_generation=False,
                image_generation_mode="none",
            ),
            self._step(
                "rerun_detection",
                "將新照片重新送入 AI 檢測流程。",
                None,
                "整體組裝物",
                error_type,
                evidence,
                requires_image_generation=False,
                image_generation_mode="none",
            ),
            self._step(
                "finish",
                "AI 判定組裝正確後，完成本次局部修正。",
                None,
                "整體組裝物",
                error_type,
                evidence,
                requires_image_generation=False,
                image_generation_mode="none",
            ),
        ]

    def _step(
        self,
        action: str,
        instruction: str,
        part_id: Optional[str],
        part_name: str,
        error_type: str,
        evidence: Evidence,
        expected_position: Optional[str] = None,
        expected_orientation: Optional[str] = None,
        requires_image_generation: bool = True,
        image_generation_mode: str = "edit_previous_image",
        verification: Optional[str] = None,
        safety_note: Optional[str] = None,
    ) -> SOPStep:
        return SOPStep(
            step_no=0,
            action=action,
            title=ACTION_TITLES.get(action, action),
            instruction=instruction,
            target_part_id=part_id,
            target_part_name=part_name,
            error_type=error_type,
            expected_position=expected_position,
            expected_orientation=expected_orientation,
            requires_image_generation=requires_image_generation,
            image_generation_mode=image_generation_mode,
            verification=verification,
            safety_note=safety_note,
            evidence=evidence,
        )

    def _build_evidence(
        self,
        localization: dict[str, Any],
        strategy: dict[str, Any],
        test_image: Optional[str],
        reference_image: Optional[str],
    ) -> Evidence:
        bbox = localization.get("selected_bbox")
        valid_bbox = (
            [float(value) for value in bbox]
            if isinstance(bbox, list) and len(bbox) == 4
            else None
        )
        return Evidence(
            source_image=self._as_text(
                localization.get("localization_source_image")
                or strategy.get("image_path")
                or test_image
            ),
            reference_image=reference_image,
            annotated_image=self._as_text(
                localization.get("annotated_image_path")
            ),
            bbox=valid_bbox,
            localization_role=self._as_text(
                localization.get("localization_role")
                or strategy.get("localization_role")
            ),
            detection_score=self._to_float(
                localization.get("selected_detection_score")
            ),
            selection_score=self._to_float(
                localization.get("selected_selection_score")
            ),
        )

    def _localization_reliable(self, localization: dict[str, Any]) -> bool:
        if localization.get("status") != "success":
            return False
        bbox = localization.get("selected_bbox")
        detection = self._to_float(
            localization.get("selected_detection_score")
        )
        selection = self._to_float(
            localization.get("selected_selection_score")
        )
        return (
            isinstance(bbox, list)
            and len(bbox) == 4
            and detection is not None
            and detection >= self.localization_detection_threshold
            and selection is not None
            and selection >= self.localization_selection_threshold
        )

    def _part_identity_reliable(
        self,
        part_id: str,
        confidence: Optional[float],
    ) -> bool:
        return (
            part_id in self.part_library
            and not part_id.lower().startswith("unknown")
            and confidence is not None
            and confidence >= self.part_confidence_threshold
        )

    def _part_name(
        self,
        part_id: str,
        error_type: str,
        reliable: bool,
    ) -> str:
        if not reliable:
            return (
                "標示區域中的多餘零件"
                if error_type == "extrapart"
                else "標示區域中的目標零件"
            )

        aliases = self.part_library.get(part_id, [])
        for alias in aliases:
            text = str(alias)
            if any("\u4e00" <= char <= "\u9fff" for char in text):
                return text
        return str(aliases[0]) if aliases else part_id.replace("_", " ")

    @staticmethod
    def _find_expected_part(
        expected_state: dict[str, Any],
        part_id: str,
    ) -> Optional[dict[str, Any]]:
        for part in expected_state.get("expected_parts", []):
            if isinstance(part, dict) and part.get("part_id") == part_id:
                return part
        return None

    @staticmethod
    def _expected_text(
        expected_part: Optional[dict[str, Any]],
    ) -> tuple[Optional[str], Optional[str]]:
        if not expected_part:
            return None, None

        raw_position = expected_part.get("position")
        raw_orientation = expected_part.get("orientation")
        position = (
            POSITION_ZH.get(str(raw_position).upper(), str(raw_position))
            if raw_position
            else None
        )
        orientation = (
            ORIENTATION_ZH.get(
                str(raw_orientation).upper(),
                str(raw_orientation),
            )
            if raw_orientation
            else None
        )
        return position, orientation

    @staticmethod
    def _insert_instruction(
        part_name: str,
        position: Optional[str],
        orientation: Optional[str],
    ) -> str:
        text = f"將 {part_name} 安裝至{position or '參考圖所示位置'}"
        if orientation:
            text += f"，並保持{orientation}方向"
        return text + "。"

    @staticmethod
    def _extract_error_parts(
        model_response: dict[str, Any],
    ) -> list[dict[str, Any]]:
        detected = model_response.get("detected_parts", [])
        if not isinstance(detected, list):
            return []
        return [
            part
            for part in detected
            if isinstance(part, dict)
            and str(part.get("error_type", "uncertain")).lower() != "correct"
        ]

    @staticmethod
    def _resolve_expected_state_path(
        payload: dict[str, Any],
        model_id: str,
        step_id: str,
    ) -> Path:
        value = payload.get("expected_state_path")
        if value:
            path = Path(str(value))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
        else:
            path = PROJECT_ROOT / "ground_truth" / model_id / f"{step_id}.json"

        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Expected-state JSON not found: {path}")
        return path

    @staticmethod
    def _load_json(path: str | Path) -> dict[str, Any]:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"JSON file not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected JSON object: {resolved}")
        return payload

    @staticmethod
    def _renumber(steps: list[SOPStep]) -> None:
        for index, step in enumerate(steps, start=1):
            step.step_no = index

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


def find_latest_results_json() -> Path:
    files = sorted(
        DEFAULT_RESULTS_ROOT.glob("*/results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(
            f"No results.json found under: {DEFAULT_RESULTS_ROOT}"
        )
    return files[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate correction SOP from pipeline results.json."
    )
    parser.add_argument("--results-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    results_json = (
        args.results_json.expanduser().resolve()
        if args.results_json
        else find_latest_results_json()
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else results_json.parent
    )

    generator = CorrectionSOPGenerator()
    sop = generator.generate_from_results(results_json)
    json_path, md_path = generator.save(sop, output_dir)

    print("=" * 70)
    print("Correction SOP generated")
    print("=" * 70)
    print(f"Source: {results_json}")
    print(f"JSON:   {json_path}")
    print(f"MD:     {md_path}")
    print(f"Steps:  {len(sop.correction_plan)}")
    print(f"Manual review required: {sop.requires_manual_review}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
