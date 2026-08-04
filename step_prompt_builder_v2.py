"""
step_prompt_builder_v2.py

將 correction_sop.json 轉換為最佳化的 V2.1 圖片教學任務。

核心流程
--------
prepare_part
    └─ 獨立支線：generate
       生成單一零件白底產品圖，不參與後續組裝圖串接

locate_installation_point
    └─ 主組裝支線起點：edit
       永遠以原始錯誤測試圖為 base image

insert / remove / reposition / reorient / rebuild ...
    └─ edit
       以上一張主組裝支線輸出作為 base image

verify_local_result
    └─ edit
       以上一張主組裝支線輸出作為 base image
       產生乾淨完成圖

compare_reference
    └─ compose_python
       使用 Python 合成「修正結果 vs 正確參考圖」
       不呼叫圖片模型

本模組不呼叫 OpenAI、不修改圖片、不建立 mask，只輸出：
- step_prompts_v2.json
- step_prompts_v2.md
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
DEFAULT_SOP_ROOT = PROJECT_ROOT / "output" / "pipeline" / "error_aware_localization_smoke_test"

POSITION_EN = {
    "左側": "left side", "右側": "right side", "中央": "center",
    "上方": "upper area", "下方": "lower area", "前方": "front",
    "後方": "rear", "左上方": "upper-left area", "右上方": "upper-right area",
    "左下方": "lower-left area", "右下方": "lower-right area",
    "LEFT": "left side", "RIGHT": "right side", "CENTER": "center",
    "TOP": "upper area", "BOTTOM": "lower area", "FRONT": "front",
    "BACK": "rear", "TOP_LEFT": "upper-left area", "TOP_RIGHT": "upper-right area",
    "BOTTOM_LEFT": "lower-left area", "BOTTOM_RIGHT": "lower-right area",
}

ORIENTATION_EN = {
    "水平": "horizontal", "垂直": "vertical",
    "HORIZONTAL": "horizontal", "VERTICAL": "vertical",
}


@dataclass
class PromptImageInput:
    role: str
    path: Optional[str]
    description: str
    required: bool = True


@dataclass
class InstructionStructure:
    objective: str
    expected_result: str
    edit_region: str
    visual_style: str
    operation_constraints: list[str]


@dataclass
class ImageTask:
    task_type: str
    api_mode: str
    branch: str
    base_image_role: str
    use_previous_output: bool
    use_reference_image: bool
    use_annotated_image: bool
    use_mask_if_available: bool
    draw_arrow: bool
    arrow_type: Optional[str]
    arrow_count: int
    create_product_shot: bool
    compose_with_python: bool
    remove_overlays_in_output: bool


@dataclass
class StepPromptV2:
    sequence_index: int
    sop_step_no: int
    action: str
    title: str
    error_type: str
    target_part_id: Optional[str]
    target_part_name_zh: str
    target_part_visual_name_en: str
    expected_position_zh: Optional[str]
    expected_position_en: Optional[str]
    expected_orientation_zh: Optional[str]
    expected_orientation_en: Optional[str]
    generation_mode: str
    image_task: ImageTask
    instruction_structure: InstructionStructure
    prompt_version: str
    prompt_language: str
    prompt_en: str
    negative_prompt_en: str
    localization_hint_en: str
    bbox: Optional[list[float]]
    image_inputs: list[PromptImageInput]
    previous_output_filename: Optional[str]
    output_filename: str
    requires_manual_review: bool
    generation_allowed: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class PromptPackageV2:
    schema_version: str
    prompt_version: str
    model_id: str
    step_id: str
    image_name: str
    source_sop_json: str
    is_error: bool
    overall_error_type: str
    requires_manual_review: bool
    generation_allowed: bool
    assembly_branch_start_image: Optional[str]
    standalone_branch_outputs: list[str]
    assembly_branch_outputs: list[str]
    global_editing_policy_en: str
    global_negative_prompt_en: str
    step_prompts: list[StepPromptV2]
    skipped_steps: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StepPromptBuilderV2:
    SUPPORTED_ACTIONS = {
        "prepare_part", "locate_installation_point", "inspect_target",
        "insert_part", "remove_part", "detach_part", "replace_part",
        "reposition_part", "reorient_part", "disassemble_local_area",
        "rebuild_local_area", "verify_local_result", "compare_reference",
    }

    def __init__(self, *, block_on_manual_review: bool = True, prompt_version: str = "2.1") -> None:
        self.block_on_manual_review = block_on_manual_review
        self.prompt_version = prompt_version
        self.part_library = self._load_part_library(PART_LIBRARY_PATH)

    def build_from_sop(self, sop_json_path: str | Path) -> PromptPackageV2:
        sop_path = Path(sop_json_path).expanduser().resolve()
        sop = self._load_json(sop_path)
        correction_plan = sop.get("correction_plan", [])
        if not isinstance(correction_plan, list):
            raise TypeError("correction_plan must be a list.")

        requires_manual_review = bool(sop.get("requires_manual_review", False))
        generation_allowed = not (self.block_on_manual_review and requires_manual_review)

        warnings: list[str] = []
        if requires_manual_review:
            warnings.append("此案例需要人工確認。")
            if self.block_on_manual_review:
                warnings.append("圖片任務已建立，但後續生成器不得自動呼叫 API。")

        step_prompts: list[StepPromptV2] = []
        skipped_steps: list[dict[str, Any]] = []
        standalone_outputs: list[str] = []
        assembly_outputs: list[str] = []
        assembly_previous_output: Optional[str] = None
        sequence_index = 0

        for raw_step in correction_plan:
            if not isinstance(raw_step, dict):
                skipped_steps.append({"reason": "step is not a JSON object"})
                continue

            sop_step_no = int(raw_step.get("step_no", 0))
            action = str(raw_step.get("action", "")).strip()

            if not bool(raw_step.get("requires_image_generation", False)):
                skipped_steps.append({
                    "step_no": sop_step_no,
                    "action": action,
                    "reason": "requires_image_generation is false",
                })
                continue

            if action not in self.SUPPORTED_ACTIONS:
                skipped_steps.append({
                    "step_no": sop_step_no,
                    "action": action,
                    "reason": "unsupported V2 image task",
                })
                continue

            sequence_index += 1
            previous_for_step = self._previous_output_for_action(
                action=action,
                assembly_previous_output=assembly_previous_output,
            )

            prompt = self._build_step_prompt(
                raw_step=raw_step,
                sop=sop,
                sequence_index=sequence_index,
                previous_output_filename=previous_for_step,
                requires_manual_review=requires_manual_review,
                generation_allowed=generation_allowed,
            )
            step_prompts.append(prompt)

            if prompt.image_task.branch == "standalone":
                standalone_outputs.append(prompt.output_filename)
            elif prompt.image_task.branch == "assembly":
                assembly_outputs.append(prompt.output_filename)
                if prompt.image_task.api_mode == "edit":
                    assembly_previous_output = prompt.output_filename

        if not assembly_outputs:
            warnings.append("沒有建立任何主組裝支線圖片任務。")

        return PromptPackageV2(
            schema_version="2.1",
            prompt_version=self.prompt_version,
            model_id=str(sop.get("model_id", "")),
            step_id=str(sop.get("step_id", "")),
            image_name=str(sop.get("image_name", "")),
            source_sop_json=str(sop_path),
            is_error=bool(sop.get("is_error", False)),
            overall_error_type=str(sop.get("overall_error_type", "uncertain")),
            requires_manual_review=requires_manual_review,
            generation_allowed=generation_allowed,
            assembly_branch_start_image=self._optional_string(sop.get("test_image_path")),
            standalone_branch_outputs=standalone_outputs,
            assembly_branch_outputs=assembly_outputs,
            global_editing_policy_en=self._global_editing_policy(),
            global_negative_prompt_en=self._global_negative_prompt(),
            step_prompts=step_prompts,
            skipped_steps=skipped_steps,
            warnings=warnings,
        )

    def save(self, package: PromptPackageV2, output_dir: str | Path) -> tuple[Path, Path]:
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "step_prompts_v2.json"
        md_path = output_path / "step_prompts_v2.md"
        json_path.write_text(
            json.dumps(package.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        md_path.write_text(self.to_markdown(package), encoding="utf-8")
        return json_path, md_path

    def to_markdown(self, package: PromptPackageV2) -> str:
        lines = [
            "# Step Image Prompts V2.1", "",
            f"- Model：`{package.model_id}`",
            f"- Step：`{package.step_id}`",
            f"- Image：`{package.image_name}`",
            f"- Error Type：`{package.overall_error_type}`",
            f"- Manual Review：`{package.requires_manual_review}`",
            f"- Generation Allowed：`{package.generation_allowed}`",
            f"- Assembly Branch Start：`{package.assembly_branch_start_image}`", "",
            "## Branch Summary", "", "### Standalone Branch Outputs", "",
        ]
        lines += [f"- `{x}`" for x in package.standalone_branch_outputs] or ["- None"]
        lines += ["", "### Assembly Branch Outputs", ""]
        lines += [f"- `{x}`" for x in package.assembly_branch_outputs] or ["- None"]
        lines += [
            "", "## Global Editing Policy", "", "```text",
            package.global_editing_policy_en, "```", "",
            "## Global Negative Prompt", "", "```text",
            package.global_negative_prompt_en, "```", "",
        ]

        if package.warnings:
            lines += ["## Warnings", ""] + [f"- {w}" for w in package.warnings] + [""]

        lines += ["## Image Tasks", ""]
        for step in package.step_prompts:
            lines += [
                f"### Sequence {step.sequence_index}｜SOP Step {step.sop_step_no}｜{step.title}", "",
                f"- Action：`{step.action}`",
                f"- Branch：`{step.image_task.branch}`",
                f"- Task Type：`{step.image_task.task_type}`",
                f"- API Mode：`{step.image_task.api_mode}`",
                f"- Base Role：`{step.image_task.base_image_role}`",
                f"- Target：{step.target_part_name_zh}",
                f"- Visual Target：`{step.target_part_visual_name_en}`",
                f"- Previous Output：`{step.previous_output_filename}`",
                f"- Output：`{step.output_filename}`", "",
                "**Objective**", "", step.instruction_structure.objective, "",
                "**Expected Result**", "", step.instruction_structure.expected_result, "",
                "**Edit Region**", "", step.instruction_structure.edit_region, "",
                "**Prompt**", "", "```text", step.prompt_en, "```", "",
                "**Negative Prompt**", "", "```text", step.negative_prompt_en, "```", "",
                "**Image Inputs**", "",
            ]
            for item in step.image_inputs:
                lines.append(f"- `{item.role}`：{item.path or 'resolved at runtime'}")
            lines.append("")
            if step.warnings:
                lines += ["**Step Warnings**", ""] + [f"- {w}" for w in step.warnings] + [""]

        if package.skipped_steps:
            lines += ["## Skipped Steps", ""]
            for item in package.skipped_steps:
                lines.append(
                    f"- Step {item.get('step_no')} `{item.get('action', '')}`：{item.get('reason')}"
                )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _previous_output_for_action(*, action: str, assembly_previous_output: Optional[str]) -> Optional[str]:
        if action == "prepare_part":
            return None
        if action == "locate_installation_point":
            return None
        return assembly_previous_output

    def _build_step_prompt(
        self,
        *,
        raw_step: dict[str, Any],
        sop: dict[str, Any],
        sequence_index: int,
        previous_output_filename: Optional[str],
        requires_manual_review: bool,
        generation_allowed: bool,
    ) -> StepPromptV2:
        action = str(raw_step.get("action", ""))
        sop_step_no = int(raw_step.get("step_no", 0))
        evidence = raw_step.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}

        part_id = self._optional_string(raw_step.get("target_part_id"))
        part_name_zh = str(raw_step.get("target_part_name", "目標零件"))
        visual_name = self._part_visual_name(part_id, part_name_zh)
        pos_zh = self._optional_string(raw_step.get("expected_position"))
        ori_zh = self._optional_string(raw_step.get("expected_orientation"))
        pos_en = self._position_en(pos_zh)
        ori_en = self._orientation_en(ori_zh)
        bbox = self._normalize_bbox(evidence.get("bbox"))
        localization_hint = self._build_localization_hint(
            evidence=evidence,
            expected_position_en=pos_en,
            bbox=bbox,
        )
        image_task = self._build_image_task(
            action=action,
            previous_output_filename=previous_output_filename,
            bbox=bbox,
        )
        structure = self._build_instruction_structure(
            action=action,
            visual_part_name=visual_name,
            expected_position_en=pos_en,
            expected_orientation_en=ori_en,
            localization_hint_en=localization_hint,
        )
        prompt_en = self._compose_prompt(
            sop=sop,
            action=action,
            visual_part_name=visual_name,
            image_task=image_task,
            instruction_structure=structure,
        )
        negative = self._step_negative_prompt(action=action, image_task=image_task)
        output_filename = self._output_filename(
            sequence_index=sequence_index,
            sop_step_no=sop_step_no,
            action=action,
            api_mode=image_task.api_mode,
        )
        image_inputs = self._build_image_inputs(
            sop=sop,
            evidence=evidence,
            image_task=image_task,
            previous_output_filename=previous_output_filename,
        )

        warnings: list[str] = []
        if requires_manual_review:
            warnings.append("SOP requires manual review before generation.")
        if bbox is None and action not in {"prepare_part", "compare_reference"}:
            warnings.append("No bbox is available. Confirm the target region manually.")
        if not bool(sop.get("localization_reliable", False)):
            warnings.append("Localization is unreliable; bbox is only metadata guidance.")
        if action == "prepare_part":
            warnings.append("Standalone product-shot task; it does not feed the assembly branch.")
        if action == "locate_installation_point":
            warnings.append("Assembly-branch start; use the original erroneous test image.")
        if action == "compare_reference":
            warnings.append("Use Python composition rather than an image-model call.")

        return StepPromptV2(
            sequence_index=sequence_index,
            sop_step_no=sop_step_no,
            action=action,
            title=str(raw_step.get("title", action)),
            error_type=str(raw_step.get("error_type", sop.get("overall_error_type", "uncertain"))),
            target_part_id=part_id,
            target_part_name_zh=part_name_zh,
            target_part_visual_name_en=visual_name,
            expected_position_zh=pos_zh,
            expected_position_en=pos_en,
            expected_orientation_zh=ori_zh,
            expected_orientation_en=ori_en,
            generation_mode=image_task.api_mode,
            image_task=image_task,
            instruction_structure=structure,
            prompt_version=self.prompt_version,
            prompt_language="English",
            prompt_en=prompt_en,
            negative_prompt_en=negative,
            localization_hint_en=localization_hint,
            bbox=bbox,
            image_inputs=image_inputs,
            previous_output_filename=previous_output_filename,
            output_filename=output_filename,
            requires_manual_review=requires_manual_review,
            generation_allowed=generation_allowed,
            warnings=warnings,
        )

    def _build_image_task(
        self,
        *,
        action: str,
        previous_output_filename: Optional[str],
        bbox: Optional[list[float]],
    ) -> ImageTask:
        if action == "prepare_part":
            return ImageTask(
                "product_shot", "generate", "standalone", "none",
                False, True, False, False, False, None, 0, True, False, True,
            )
        if action == "compare_reference":
            return ImageTask(
                "comparison_panel", "compose_python", "composition", "assembly_final_output",
                True, True, False, False, False, None, 0, False, True, True,
            )
        if action == "verify_local_result":
            return ImageTask(
                "verification_edit", "edit", "assembly",
                "previous_assembly_output" if previous_output_filename else "original_test_image",
                bool(previous_output_filename), True, False, bool(bbox),
                False, None, 0, False, False, True,
            )

        draw_arrow = action in {
            "locate_installation_point", "insert_part", "remove_part",
            "reposition_part", "reorient_part", "disassemble_local_area",
        }
        arrow_type = None
        if action == "locate_installation_point":
            arrow_type = "pointer"
        elif action in {"insert_part", "remove_part", "reposition_part", "disassemble_local_area"}:
            arrow_type = "linear"
        elif action == "reorient_part":
            arrow_type = "curved"

        return ImageTask(
            "localized_instruction_edit", "edit", "assembly",
            "previous_assembly_output" if previous_output_filename else "original_test_image",
            bool(previous_output_filename), True, True, bool(bbox),
            draw_arrow, arrow_type, 1 if draw_arrow else 0,
            False, False, action in {"replace_part", "rebuild_local_area"},
        )

    def _build_instruction_structure(
        self,
        *,
        action: str,
        visual_part_name: str,
        expected_position_en: Optional[str],
        expected_orientation_en: Optional[str],
        localization_hint_en: str,
    ) -> InstructionStructure:
        position = expected_position_en or "the reference-indicated location"
        orientation = expected_orientation_en or "the orientation shown in the correct reference"

        structures = {
            "prepare_part": InstructionStructure(
                f"Prepare one correct {visual_part_name}.",
                f"A single isolated {visual_part_name} is centered on a clean white background, fully visible, and not connected to an assembly.",
                "No assembly edit region. Generate a standalone part product shot.",
                "Realistic product photograph for a professional assembly manual.",
                [
                    "Show exactly one part.", "Do not show the full assembly.",
                    "Do not show hands.", "Do not draw arrows.", "Do not add text.",
                    "Match reference geometry, material, color, and hole pattern.",
                ],
            ),
            "locate_installation_point": InstructionStructure(
                f"Show where the {visual_part_name} should be installed.",
                f"The original erroneous assembly remains unchanged, with exactly one pointer arrow indicating the connector in {position}.",
                localization_hint_en,
                "Realistic photo-based assembly instruction frame.",
                [
                    "Use the original erroneous test image as the editable base.",
                    "Do not add the target part.", "Do not alter the assembly.",
                    "Use exactly one pointer arrow.", "Do not copy the whole reference image.",
                ],
            ),
            "insert_part": InstructionStructure(
                f"Demonstrate insertion of the {visual_part_name}.",
                f"The {visual_part_name} is aligned with the connector at {position}, immediately before full insertion, with {orientation} orientation.",
                localization_hint_en,
                "Realistic photo-based exploded assembly instruction.",
                [
                    "Use one short linear arrow.", "Keep all non-target parts unchanged.",
                    "Do not hide the connector.", "Do not duplicate the target part.",
                ],
            ),
            "inspect_target": InstructionStructure(
                f"Identify the {visual_part_name} or target region.",
                "The assembly remains physically unchanged while one subtle indicator identifies the target.",
                localization_hint_en,
                "Diagnostic instruction frame.",
                ["Do not add, remove, rotate, or move parts.", "Use one subtle indicator only."],
            ),
            "remove_part": InstructionStructure(
                f"Demonstrate removal of the {visual_part_name}.",
                f"The {visual_part_name} is slightly separated from its connector with one outward arrow, while surrounding parts remain unchanged.",
                localization_hint_en,
                "Realistic exploded removal instruction.",
                ["Move only the target part.", "Preserve the original connector.", "Use one outward arrow."],
            ),
            "detach_part": InstructionStructure(
                f"Loosen the {visual_part_name} for adjustment.",
                "The target part is slightly detached but remains close to the original connection.",
                localization_hint_en,
                "Realistic local adjustment instruction.",
                ["Use minimal separation.", "Do not disturb surrounding parts."],
            ),
            "replace_part": InstructionStructure(
                "Replace the incorrect component with the correct component.",
                "The incorrect component is absent and the correct replacement is aligned with the same connector.",
                localization_hint_en,
                "Realistic replacement assembly instruction.",
                [
                    "Use the reference only for replacement geometry.",
                    "Preserve all non-target parts.", "Do not copy the full reference scene.",
                ],
            ),
            "reposition_part": InstructionStructure(
                f"Move the {visual_part_name} to the correct position.",
                f"The target part is shown moving toward {position} while remaining mechanically aligned.",
                localization_hint_en,
                "Realistic positional correction instruction.",
                ["Use one directional arrow.", "Do not change part geometry.", "Preserve surrounding parts."],
            ),
            "reorient_part": InstructionStructure(
                f"Rotate the {visual_part_name} to the correct orientation.",
                f"The target part is shown rotating toward a {orientation} orientation.",
                localization_hint_en,
                "Realistic orientation correction instruction.",
                ["Use one curved arrow.", "Do not move unrelated parts."],
            ),
            "disassemble_local_area": InstructionStructure(
                "Disassemble only the affected local structure.",
                "Only the required local components are separated in a clear exploded view.",
                localization_hint_en,
                "Local exploded disassembly instruction.",
                ["Keep unaffected structure intact.", "Preserve removed-part identity and order.", "Use simple outward arrows."],
            ),
            "rebuild_local_area": InstructionStructure(
                "Rebuild the affected local structure correctly.",
                "Previously separated components are reconnected in the correct order.",
                localization_hint_en,
                "Realistic local rebuild instruction.",
                ["Use the reference only for connection verification.", "Preserve unaffected structure."],
            ),
            "verify_local_result": InstructionStructure(
                "Show the corrected local result.",
                "The target component is fully connected, with no arrows, loose parts, highlights, boxes, or overlays.",
                localization_hint_en,
                "Clean final verification photograph.",
                [
                    "Remove all instructional overlays.",
                    "Preserve the rest of the previous assembly image.",
                    "Match the reference only in the corrected local region.",
                ],
            ),
            "compare_reference": InstructionStructure(
                "Create a comparison panel between the corrected result and the correct reference.",
                "A clean two-panel comparison is produced by Python composition without regenerating either source image.",
                "Whole-image comparison.",
                "Simple professional comparison layout.",
                ["Do not call the image API.", "Do not modify either source image.", "Use Python composition only."],
            ),
        }
        return structures[action]

    def _compose_prompt(
        self,
        *,
        sop: dict[str, Any],
        action: str,
        visual_part_name: str,
        image_task: ImageTask,
        instruction_structure: InstructionStructure,
    ) -> str:
        constraints = "\n".join(f"- {x}" for x in instruction_structure.operation_constraints)
        if image_task.api_mode == "generate":
            image_roles = (
                "The reference image, when provided, is used only to reproduce the target component's "
                "exact geometry, color, material, scale, and hole pattern. Do not reproduce the full assembly."
            )
        elif image_task.api_mode == "edit":
            image_roles = (
                "Image 1 is the editable base image. Image 2 is the correct structural reference only. "
                "Image 3, when present, is a localization aid only."
            )
        else:
            image_roles = "Do not send this task to the image model. Complete it using Python image composition."

        return (
            f"MODEL CONTEXT\nModel: {sop.get('model_id', '')}\n"
            f"Assembly stage: {sop.get('step_id', '')}\nAction: {action}\n"
            f"Target component: {visual_part_name}\nBranch: {image_task.branch}\n\n"
            f"OBJECTIVE\n{instruction_structure.objective}\n\n"
            f"EXPECTED RESULT\n{instruction_structure.expected_result}\n\n"
            f"EDIT REGION\n{instruction_structure.edit_region}\n\n"
            f"VISUAL STYLE\n{instruction_structure.visual_style}\n\n"
            f"INPUT IMAGE ROLES\n{image_roles}\n\n"
            f"OPERATION CONSTRAINTS\n{constraints}\n\n"
            "Return exactly one image for this task."
        )

    @staticmethod
    def _step_negative_prompt(*, action: str, image_task: ImageTask) -> str:
        rules = [
            "Do not add text, labels, captions, numbers, logos, borders, or watermarks.",
            "Do not distort, melt, bend, fuse, duplicate, or recolor components.",
            "Do not invent unlisted parts.",
        ]
        if image_task.api_mode == "generate":
            rules += [
                "Do not show the full assembly.", "Do not show hands.",
                "Do not show arrows.", "Do not show more than one component.",
            ]
        elif image_task.api_mode == "edit":
            rules += [
                "Do not change camera angle, crop, perspective, background, lighting, or shadows.",
                "Do not regenerate the entire scene.",
                "Do not replace the base image with the reference image.",
                "Do not alter non-target components.",
            ]
        if action == "verify_local_result":
            rules.append("Do not keep arrows, boxes, highlights, or loose parts.")
        if action == "compare_reference":
            rules.append("Do not call the image model.")
        return " ".join(rules)

    def _build_image_inputs(
        self,
        *,
        sop: dict[str, Any],
        evidence: dict[str, Any],
        image_task: ImageTask,
        previous_output_filename: Optional[str],
    ) -> list[PromptImageInput]:
        test_image = self._optional_string(sop.get("test_image_path"))
        reference = self._optional_string(evidence.get("reference_image") or sop.get("reference_image_path"))
        annotated = self._optional_string(evidence.get("annotated_image"))

        if image_task.api_mode == "generate":
            return [PromptImageInput(
                "reference_image", reference,
                "Reference only for exact target-part geometry, color, material, scale, and hole pattern.",
                True,
            )]

        if image_task.api_mode == "compose_python":
            return [
                PromptImageInput(
                    "generated_result", None,
                    f"Resolve at runtime to the last assembly-branch output: {previous_output_filename}",
                    True,
                ),
                PromptImageInput(
                    "reference_image", reference,
                    "Correct reference used as the comparison panel.",
                    True,
                ),
            ]

        inputs: list[PromptImageInput] = []
        if image_task.use_previous_output:
            inputs.append(PromptImageInput(
                "base_image", None,
                f"Resolve at runtime to the previous assembly-branch output: {previous_output_filename}",
                True,
            ))
        else:
            inputs.append(PromptImageInput(
                "base_image", test_image,
                "Original erroneous test image used as the editable base.",
                True,
            ))

        if image_task.use_reference_image:
            inputs.append(PromptImageInput(
                "reference_image", reference,
                "Correct structural reference only. Never replace the base image.",
                True,
            ))
        if image_task.use_annotated_image and annotated:
            inputs.append(PromptImageInput(
                "annotated_image", annotated,
                "Localization aid only. Do not preserve boxes or overlays.",
                False,
            ))
        return inputs

    @staticmethod
    def _global_editing_policy() -> str:
        return (
            "For edit tasks, Image 1 is always the editable base. Preserve camera angle, crop, perspective, "
            "background, lighting, shadows, and all non-target components. Reference images are only for "
            "identity, geometry, connector, position, orientation, and final verification. Never copy the full "
            "reference image. Annotated images are localization aids only. Modify only the current target region."
        )

    @staticmethod
    def _global_negative_prompt() -> str:
        return (
            "Do not redesign or regenerate the full assembly. Do not change camera angle, crop, perspective, "
            "background, lighting, or shadows. Do not alter non-target components. Do not add duplicate or "
            "unlisted parts. Do not remove correct parts. Do not change colors, scale, proportions, hole counts, "
            "or connector geometry. Do not include text, logos, watermarks, UI elements, red boxes, or labels."
        )

    def _part_visual_name(self, part_id: Optional[str], fallback_name: str) -> str:
        if part_id and part_id in self.part_library:
            aliases = self.part_library[part_id]
            english = [x for x in aliases if not self._contains_chinese(x)]
            if english:
                return sorted(english, key=lambda text: (-len(text.split()), len(text)))[0]
        if part_id:
            return part_id.replace("_", " ").lower()
        return "target construction component" if self._contains_chinese(fallback_name) else fallback_name

    @staticmethod
    def _build_localization_hint(
        *,
        evidence: dict[str, Any],
        expected_position_en: Optional[str],
        bbox: Optional[list[float]],
    ) -> str:
        role = str(evidence.get("localization_role", ""))
        if role == "reference_missing_part_location":
            text = "Use the annotated reference image to identify the approximate correct installation region"
        elif role == "test_extra_part_location":
            text = "Use the annotated test image to identify the approximate extra-part region"
        elif role == "test_error_part_location":
            text = "Use the annotated test image to identify the approximate incorrect-part region"
        else:
            text = "Use visual comparison with the correct reference to identify the target region"

        if expected_position_en:
            text += f", generally near the {expected_position_en}"
        text += (
            ". A machine-readable bbox is available in metadata for optional mask creation"
            if bbox is not None
            else ". No reliable bbox is available"
        )
        return text + "."

    @staticmethod
    def _position_en(value: Optional[str]) -> Optional[str]:
        return None if not value else POSITION_EN.get(value, value.lower())

    @staticmethod
    def _orientation_en(value: Optional[str]) -> Optional[str]:
        return None if not value else ORIENTATION_EN.get(value, value.lower())

    @staticmethod
    def _output_filename(*, sequence_index: int, sop_step_no: int, action: str, api_mode: str) -> str:
        suffix = "_comparison" if api_mode == "compose_python" else ""
        return f"sequence_{sequence_index:02d}_sop_{sop_step_no:02d}_{action}{suffix}.png"

    @staticmethod
    def _normalize_bbox(value: Any) -> Optional[list[float]]:
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _load_part_library(path: Path) -> dict[str, list[str]]:
        if not path.is_file():
            raise FileNotFoundError(f"Part library not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("part_library.json must be a JSON object.")
        return {
            str(k): [str(x) for x in v] if isinstance(v, list) else [str(v)]
            for k, v in raw.items()
        }

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Expected JSON object: {path}")
        return payload

    @staticmethod
    def _contains_chinese(value: str) -> bool:
        return any("\u4e00" <= c <= "\u9fff" for c in value)

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


def find_latest_sop_json() -> Path:
    files = sorted(
        DEFAULT_SOP_ROOT.glob("*/correction_sop.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No correction_sop.json found under:\n{DEFAULT_SOP_ROOT}")
    return files[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build optimized V2.1 image tasks from correction_sop.json."
    )
    parser.add_argument("--sop-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-manual-review", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sop_json = args.sop_json.expanduser().resolve() if args.sop_json else find_latest_sop_json()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else sop_json.parent

    builder = StepPromptBuilderV2(block_on_manual_review=not args.allow_manual_review)
    package = builder.build_from_sop(sop_json)
    json_path, md_path = builder.save(package, output_dir)

    print("=" * 70)
    print("Step Prompt Builder V2.1 finished")
    print("=" * 70)
    print(f"Source SOP:              {sop_json}")
    print(f"JSON:                    {json_path}")
    print(f"Markdown:                {md_path}")
    print(f"Image tasks:             {len(package.step_prompts)}")
    print(f"Standalone outputs:      {len(package.standalone_branch_outputs)}")
    print(f"Assembly branch outputs: {len(package.assembly_branch_outputs)}")
    print(f"Generation allowed:      {package.generation_allowed}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise