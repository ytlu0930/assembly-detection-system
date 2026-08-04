"""
step_image_generator_v2.py

讀取 step_prompts_v2.json，依 image_task.api_mode 分流執行：

1. api_mode = "generate"
   - 用於 prepare_part 的獨立零件產品圖。
   - 若任務附有 reference_image，使用 images.edit() 將參考照片中的
     目標零件抽離成單一白底產品圖；因 Images Generations endpoint
     本身不接受參考圖片。
   - 若沒有 reference_image，才使用 images.generate() 純文字生圖。

2. api_mode = "edit"
   - 用於主組裝支線。
   - locate 永遠以原始錯誤測試圖為 base。
   - 後續 edit 以上一張 assembly branch 輸出為 base。
   - 可附 correct reference 與 annotated localization image。
   - 可選擇依 bbox 建立 mask；若 bbox 源自 reference image 而非
     editable base，會自動跳過 mask，避免座標誤套。

3. api_mode = "compose_python"
   - 不呼叫 OpenAI。
   - 使用 Pillow 將最後修正圖與正確參考圖做左右比較。

輸出：
generated_steps_v2/
├── standalone/
├── assembly/
├── comparison/
├── masks/
├── request_previews/
├── generation_manifest_v2.json
└── generation_manifest_v2.md

環境：
專案根目錄 .env
OPENAI_API_KEY=sk-...

安裝：
python -m pip install --upgrade openai Pillow python-dotenv

安全測試：
python step_image_generator_v2.py --dry-run

只執行第一個任務：
python step_image_generator_v2.py --allow-manual-review --max-tasks 1

只執行主組裝支線第一張：
python step_image_generator_v2.py --allow-manual-review --branch assembly --max-tasks 1

完整執行：
python step_image_generator_v2.py --allow-manual-review
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_PROMPTS_ROOT = (
    PROJECT_ROOT
    / "output"
    / "pipeline"
    / "error_aware_localization_smoke_test"
)

DEFAULT_MODEL = "gpt-image-2"
DEFAULT_SIZE = "1024x1024"
DEFAULT_QUALITY = "medium"
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_BACKGROUND = "opaque"
DEFAULT_INPUT_FIDELITY = "high"

SUPPORTED_API_MODES = {
    "generate",
    "edit",
    "compose_python",
}
SUPPORTED_BRANCHES = {
    "standalone",
    "assembly",
    "composition",
}
SUPPORTED_INPUT_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}
SUPPORTED_OUTPUT_FORMATS = {
    "png",
    "jpeg",
    "webp",
}
SUPPORTED_QUALITIES = {
    "low",
    "medium",
    "high",
    "auto",
}
SUPPORTED_SIZES = {
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "auto",
}
SUPPORTED_INPUT_FIDELITIES = {
    "low",
    "high",
}

MAX_INPUT_FILE_BYTES = 49 * 1024 * 1024
MAX_MASK_FILE_BYTES = 4 * 1024 * 1024


@dataclass
class TaskRecord:
    sequence_index: int
    sop_step_no: int
    action: str
    title: str
    api_mode: str
    branch: str
    status: str

    model: str
    size: str
    quality: str
    output_format: str

    base_image_path: Optional[str] = None
    reference_image_path: Optional[str] = None
    annotated_image_path: Optional[str] = None
    mask_path: Optional[str] = None

    previous_assembly_output: Optional[str] = None
    output_path: Optional[str] = None

    actual_api_operation: Optional[str] = None
    attempts: int = 0
    elapsed_seconds: Optional[float] = None
    prompt_characters: int = 0

    error_type: Optional[str] = None
    error_message: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class GenerationManifestV2:
    schema_version: str
    created_at: str

    source_prompts_json: str
    model_id: str
    assembly_step_id: str
    source_image_name: str

    image_model: str
    size: str
    quality: str
    output_format: str
    input_fidelity: str

    dry_run: bool
    use_mask: bool
    include_annotated_image: bool

    package_generation_allowed: bool
    package_requires_manual_review: bool
    manual_review_override: bool

    requested_task_count: int
    successful_task_count: int
    existing_task_count: int
    failed_task_count: int
    skipped_task_count: int

    output_directory: str
    standalone_outputs: list[str]
    assembly_outputs: list[str]
    comparison_outputs: list[str]

    final_assembly_image_path: Optional[str]
    final_comparison_image_path: Optional[str]

    tasks: list[TaskRecord]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StepImageGeneratorV2:
    """執行 V2.1 image-task package。"""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        size: str = DEFAULT_SIZE,
        quality: str = DEFAULT_QUALITY,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        background: str = DEFAULT_BACKGROUND,
        input_fidelity: str = DEFAULT_INPUT_FIDELITY,
        use_mask: bool = False,
        mask_padding_ratio: float = 0.20,
        include_annotated_image: bool = True,
        max_retries: int = 2,
        retry_delay_seconds: float = 3.0,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.model = model
        self.size = size
        self.quality = quality
        self.output_format = output_format
        self.background = background
        self.input_fidelity = input_fidelity

        self.use_mask = use_mask
        self.mask_padding_ratio = mask_padding_ratio
        self.include_annotated_image = include_annotated_image

        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        self._validate_settings()
        self.client = client or OpenAI()

    # ======================================================
    # Public API
    # ======================================================

    def run(
        self,
        *,
        prompts_json_path: str | Path,
        output_dir: str | Path,
        allow_manual_review: bool = False,
        dry_run: bool = False,
        max_tasks: Optional[int] = None,
        branch_filter: Literal[
            "all",
            "standalone",
            "assembly",
            "composition",
        ] = "all",
        continue_on_error: bool = False,
        overwrite: bool = False,
    ) -> GenerationManifestV2:
        prompts_path = Path(
            prompts_json_path
        ).expanduser().resolve()

        package = self._load_json(
            prompts_path
        )

        self._validate_prompt_package(
            package
        )

        package_generation_allowed = bool(
            package.get(
                "generation_allowed",
                False,
            )
        )
        package_requires_manual_review = bool(
            package.get(
                "requires_manual_review",
                False,
            )
        )

        if not dry_run:
            if (
                not package_generation_allowed
                and not allow_manual_review
            ):
                raise PermissionError(
                    "step_prompts_v2.json has "
                    "generation_allowed=false. "
                    "Review the localization result and rerun with "
                    "--allow-manual-review only after manual confirmation."
                )

            if (
                package_requires_manual_review
                and not allow_manual_review
            ):
                raise PermissionError(
                    "This case requires manual review. "
                    "Use --allow-manual-review only after confirming "
                    "the target part and target region."
                )

        raw_tasks = package.get(
            "step_prompts",
            [],
        )

        selected_tasks = self._select_tasks(
            raw_tasks=raw_tasks,
            branch_filter=branch_filter,
            max_tasks=max_tasks,
        )

        output_root = Path(
            output_dir
        ).expanduser().resolve()

        standalone_dir = (
            output_root
            / "standalone"
        )
        assembly_dir = (
            output_root
            / "assembly"
        )
        comparison_dir = (
            output_root
            / "comparison"
        )
        masks_dir = (
            output_root
            / "masks"
        )
        previews_dir = (
            output_root
            / "request_previews"
        )

        for directory in (
            output_root,
            standalone_dir,
            assembly_dir,
            comparison_dir,
            previews_dir,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        if self.use_mask:
            masks_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        manifest = GenerationManifestV2(
            schema_version="2.0",
            created_at=(
                datetime.now()
                .astimezone()
                .isoformat()
            ),
            source_prompts_json=str(
                prompts_path
            ),
            model_id=str(
                package.get(
                    "model_id",
                    "",
                )
            ),
            assembly_step_id=str(
                package.get(
                    "step_id",
                    "",
                )
            ),
            source_image_name=str(
                package.get(
                    "image_name",
                    "",
                )
            ),
            image_model=self.model,
            size=self.size,
            quality=self.quality,
            output_format=self.output_format,
            input_fidelity=self.input_fidelity,
            dry_run=dry_run,
            use_mask=self.use_mask,
            include_annotated_image=(
                self.include_annotated_image
            ),
            package_generation_allowed=(
                package_generation_allowed
            ),
            package_requires_manual_review=(
                package_requires_manual_review
            ),
            manual_review_override=(
                allow_manual_review
            ),
            requested_task_count=len(
                selected_tasks
            ),
            successful_task_count=0,
            existing_task_count=0,
            failed_task_count=0,
            skipped_task_count=0,
            output_directory=str(
                output_root
            ),
            standalone_outputs=[],
            assembly_outputs=[],
            comparison_outputs=[],
            final_assembly_image_path=None,
            final_comparison_image_path=None,
            tasks=[],
            warnings=[],
        )

        if package_requires_manual_review:
            manifest.warnings.append(
                "The source package requires manual review."
            )

        if allow_manual_review:
            manifest.warnings.append(
                "Manual-review protection was explicitly overridden."
            )

        global_policy = str(
            package.get(
                "global_editing_policy_en",
                "",
            )
        ).strip()

        global_negative = str(
            package.get(
                "global_negative_prompt_en",
                "",
            )
        ).strip()

        assembly_previous_output: Optional[
            Path
        ] = None

        for raw_task in selected_tasks:
            if not isinstance(
                raw_task,
                dict,
            ):
                record = TaskRecord(
                    sequence_index=0,
                    sop_step_no=0,
                    action="unknown",
                    title="Invalid task",
                    api_mode="unknown",
                    branch="unknown",
                    status="skipped",
                    model=self.model,
                    size=self.size,
                    quality=self.quality,
                    output_format=(
                        self.output_format
                    ),
                    error_type="InvalidTask",
                    error_message=(
                        "Task is not a JSON object."
                    ),
                )
                manifest.tasks.append(
                    record
                )
                manifest.skipped_task_count += 1
                continue

            record = self._process_task(
                raw_task=raw_task,
                package=package,
                global_policy=global_policy,
                global_negative=global_negative,
                assembly_previous_output=(
                    assembly_previous_output
                ),
                standalone_dir=standalone_dir,
                assembly_dir=assembly_dir,
                comparison_dir=comparison_dir,
                masks_dir=masks_dir,
                previews_dir=previews_dir,
                dry_run=dry_run,
                overwrite=overwrite,
            )

            manifest.tasks.append(
                record
            )

            if record.status == "generated":
                manifest.successful_task_count += 1

            elif record.status == "dry_run":
                manifest.successful_task_count += 1

            elif record.status == "existing":
                manifest.existing_task_count += 1

            elif record.status == "skipped":
                manifest.skipped_task_count += 1

            else:
                manifest.failed_task_count += 1

            output_path = (
                Path(record.output_path)
                if record.output_path
                else None
            )

            if (
                record.status
                in {
                    "generated",
                    "existing",
                }
                and output_path is not None
            ):
                if record.branch == "standalone":
                    manifest.standalone_outputs.append(
                        str(output_path)
                    )

                elif record.branch == "assembly":
                    manifest.assembly_outputs.append(
                        str(output_path)
                    )
                    assembly_previous_output = (
                        output_path
                    )
                    manifest.final_assembly_image_path = (
                        str(output_path)
                    )

                elif record.branch == "composition":
                    manifest.comparison_outputs.append(
                        str(output_path)
                    )
                    manifest.final_comparison_image_path = (
                        str(output_path)
                    )

            elif (
                record.status == "dry_run"
                and output_path is not None
            ):
                # Dry run 不產生檔案，但維持預定的主支線路徑，
                # 讓後續 request preview 能檢查串接。
                if record.branch == "assembly":
                    assembly_previous_output = (
                        output_path
                    )

            if (
                record.status == "failed"
                and not continue_on_error
            ):
                self._save_manifest(
                    output_root
                    / "generation_manifest_v2.json",
                    manifest,
                )
                self._write_markdown_manifest(
                    output_root
                    / "generation_manifest_v2.md",
                    manifest,
                )
                raise RuntimeError(
                    f"Sequence {record.sequence_index} "
                    f"({record.action}) failed: "
                    f"{record.error_message}"
                )

        self._copy_final_outputs(
            manifest=manifest,
            output_root=output_root,
            dry_run=dry_run,
        )

        self._save_manifest(
            output_root
            / "generation_manifest_v2.json",
            manifest,
        )

        self._write_markdown_manifest(
            output_root
            / "generation_manifest_v2.md",
            manifest,
        )

        return manifest

    # ======================================================
    # Task dispatch
    # ======================================================

    def _process_task(
        self,
        *,
        raw_task: dict[str, Any],
        package: dict[str, Any],
        global_policy: str,
        global_negative: str,
        assembly_previous_output: Optional[Path],
        standalone_dir: Path,
        assembly_dir: Path,
        comparison_dir: Path,
        masks_dir: Path,
        previews_dir: Path,
        dry_run: bool,
        overwrite: bool,
    ) -> TaskRecord:
        sequence_index = int(
            raw_task.get(
                "sequence_index",
                0,
            )
        )
        sop_step_no = int(
            raw_task.get(
                "sop_step_no",
                0,
            )
        )
        action = str(
            raw_task.get(
                "action",
                "unknown",
            )
        )
        title = str(
            raw_task.get(
                "title",
                action,
            )
        )

        image_task = raw_task.get(
            "image_task",
            {},
        )
        if not isinstance(
            image_task,
            dict,
        ):
            image_task = {}

        api_mode = str(
            image_task.get(
                "api_mode",
                raw_task.get(
                    "generation_mode",
                    "",
                ),
            )
        )

        branch = str(
            image_task.get(
                "branch",
                "",
            )
        )

        if api_mode not in SUPPORTED_API_MODES:
            return self._failed_record(
                sequence_index=sequence_index,
                sop_step_no=sop_step_no,
                action=action,
                title=title,
                api_mode=api_mode,
                branch=branch,
                exc=ValueError(
                    f"Unsupported api_mode: {api_mode}"
                ),
            )

        if branch not in SUPPORTED_BRANCHES:
            return self._failed_record(
                sequence_index=sequence_index,
                sop_step_no=sop_step_no,
                action=action,
                title=title,
                api_mode=api_mode,
                branch=branch,
                exc=ValueError(
                    f"Unsupported branch: {branch}"
                ),
            )

        output_filename = str(
            raw_task.get(
                "output_filename",
                (
                    f"sequence_{sequence_index:02d}_"
                    f"{action}.png"
                ),
            )
        )

        output_filename = (
            f"{Path(output_filename).stem}."
            f"{self.output_format}"
        )

        if branch == "standalone":
            output_path = (
                standalone_dir
                / output_filename
            )

        elif branch == "assembly":
            output_path = (
                assembly_dir
                / output_filename
            )

        else:
            output_path = (
                comparison_dir
                / output_filename
            )

        record = TaskRecord(
            sequence_index=sequence_index,
            sop_step_no=sop_step_no,
            action=action,
            title=title,
            api_mode=api_mode,
            branch=branch,
            status="pending",
            model=self.model,
            size=self.size,
            quality=self.quality,
            output_format=self.output_format,
            output_path=str(
                output_path
            ),
            previous_assembly_output=(
                str(assembly_previous_output)
                if assembly_previous_output
                else None
            ),
        )

        try:
            if (
                output_path.is_file()
                and not overwrite
                and not dry_run
            ):
                record.status = "existing"
                record.warnings.append(
                    "Output already exists; API/composition skipped. "
                    "Use --overwrite to regenerate."
                )
                return record

            image_inputs = self._index_image_inputs(
                raw_task.get(
                    "image_inputs",
                    [],
                )
            )

            full_prompt = self._combine_prompts(
                global_policy=global_policy,
                step_prompt=str(
                    raw_task.get(
                        "prompt_en",
                        "",
                    )
                ),
                global_negative=global_negative,
                step_negative=str(
                    raw_task.get(
                        "negative_prompt_en",
                        "",
                    )
                ),
                api_mode=api_mode,
            )

            record.prompt_characters = len(
                full_prompt
            )

            if api_mode == "generate":
                return self._process_generate_task(
                    raw_task=raw_task,
                    image_inputs=image_inputs,
                    full_prompt=full_prompt,
                    output_path=output_path,
                    record=record,
                    previews_dir=previews_dir,
                    dry_run=dry_run,
                )

            if api_mode == "edit":
                return self._process_edit_task(
                    raw_task=raw_task,
                    image_inputs=image_inputs,
                    full_prompt=full_prompt,
                    assembly_previous_output=(
                        assembly_previous_output
                    ),
                    output_path=output_path,
                    record=record,
                    masks_dir=masks_dir,
                    previews_dir=previews_dir,
                    dry_run=dry_run,
                )

            return self._process_composition_task(
                raw_task=raw_task,
                image_inputs=image_inputs,
                assembly_previous_output=(
                    assembly_previous_output
                ),
                output_path=output_path,
                record=record,
                previews_dir=previews_dir,
                dry_run=dry_run,
                package=package,
            )

        except Exception as exc:
            record.status = "failed"
            record.error_type = (
                type(exc).__name__
            )
            record.error_message = str(
                exc
            )
            return record

    # ======================================================
    # Generate branch
    # ======================================================

    def _process_generate_task(
        self,
        *,
        raw_task: dict[str, Any],
        image_inputs: dict[str, dict[str, Any]],
        full_prompt: str,
        output_path: Path,
        record: TaskRecord,
        previews_dir: Path,
        dry_run: bool,
    ) -> TaskRecord:
        reference_path = (
            self._resolve_optional_image(
                image_inputs.get(
                    "reference_image"
                )
            )
        )

        record.reference_image_path = (
            str(reference_path)
            if reference_path
            else None
        )

        if reference_path is not None:
            self._validate_input_image(
                reference_path
            )

        # Generations endpoint 不接受圖片。
        # 若有 reference，使用 edits endpoint 生成「參考導向的新圖」。
        actual_operation = (
            "images.edit(reference_guided_generation)"
            if reference_path is not None
            else "images.generate(text_only)"
        )
        record.actual_api_operation = (
            actual_operation
        )

        if dry_run:
            record.status = "dry_run"
            self._save_request_preview(
                previews_dir=previews_dir,
                raw_task=raw_task,
                actual_operation=actual_operation,
                full_prompt=full_prompt,
                base_image_path=None,
                reference_image_path=reference_path,
                annotated_image_path=None,
                mask_path=None,
                output_path=output_path,
            )
            return record

        start = time.perf_counter()

        if reference_path is not None:
            response, attempts = (
                self._call_edit_with_retry(
                    image_paths=[
                        reference_path
                    ],
                    mask_path=None,
                    prompt=full_prompt,
                )
            )
        else:
            response, attempts = (
                self._call_generate_with_retry(
                    prompt=full_prompt
                )
            )

        record.attempts = attempts

        self._save_response_image(
            response=response,
            output_path=output_path,
        )

        record.elapsed_seconds = round(
            time.perf_counter()
            - start,
            3,
        )
        record.status = "generated"
        return record

    # ======================================================
    # Edit branch
    # ======================================================

    def _process_edit_task(
        self,
        *,
        raw_task: dict[str, Any],
        image_inputs: dict[str, dict[str, Any]],
        full_prompt: str,
        assembly_previous_output: Optional[Path],
        output_path: Path,
        record: TaskRecord,
        masks_dir: Path,
        previews_dir: Path,
        dry_run: bool,
    ) -> TaskRecord:
        image_task = raw_task.get(
            "image_task",
            {},
        )
        if not isinstance(
            image_task,
            dict,
        ):
            image_task = {}

        use_previous_output = bool(
            image_task.get(
                "use_previous_output",
                False,
            )
        )

        if use_previous_output:
            if (
                assembly_previous_output is None
            ):
                previous_filename = raw_task.get(
                    "previous_output_filename"
                )
                raise FileNotFoundError(
                    "This edit task requires the previous assembly output, "
                    f"but none is available. Expected: {previous_filename}"
                )

            base_path = (
                assembly_previous_output
            )

        else:
            base_path = self._resolve_required_image(
                image_inputs.get(
                    "base_image"
                ),
                role="base_image",
            )

        reference_path = (
            self._resolve_optional_image(
                image_inputs.get(
                    "reference_image"
                )
            )
        )

        annotated_path = (
            self._resolve_optional_image(
                image_inputs.get(
                    "annotated_image"
                )
            )
        )

        record.base_image_path = str(
            base_path
        )
        record.reference_image_path = (
            str(reference_path)
            if reference_path
            else None
        )
        record.annotated_image_path = (
            str(annotated_path)
            if annotated_path
            else None
        )
        record.actual_api_operation = (
            "images.edit"
        )

        # Dry run 後續步驟的 base 是預定輸出，可能尚不存在。
        if base_path.is_file():
            self._validate_input_image(
                base_path
            )
        elif not dry_run:
            raise FileNotFoundError(
                f"Editable base image not found: {base_path}"
            )
        else:
            record.warnings.append(
                "Dry run: previous assembly output does not exist yet; "
                "base-image validation skipped."
            )

        if reference_path is not None:
            self._validate_input_image(
                reference_path
            )

        if (
            annotated_path is not None
            and self.include_annotated_image
        ):
            self._validate_input_image(
                annotated_path
            )

        mask_path: Optional[Path] = None

        if self.use_mask:
            mask_path, warning = (
                self._maybe_create_mask(
                    raw_task=raw_task,
                    base_image_path=base_path,
                    masks_dir=masks_dir,
                    dry_run=dry_run,
                )
            )

            if mask_path is not None:
                record.mask_path = str(
                    mask_path
                )

            if warning:
                record.warnings.append(
                    warning
                )

        if dry_run:
            record.status = "dry_run"
            self._save_request_preview(
                previews_dir=previews_dir,
                raw_task=raw_task,
                actual_operation=(
                    record.actual_api_operation
                ),
                full_prompt=full_prompt,
                base_image_path=base_path,
                reference_image_path=(
                    reference_path
                ),
                annotated_image_path=(
                    annotated_path
                    if self.include_annotated_image
                    else None
                ),
                mask_path=mask_path,
                output_path=output_path,
            )
            return record

        image_paths = [
            base_path
        ]

        if reference_path is not None:
            image_paths.append(
                reference_path
            )

        if (
            self.include_annotated_image
            and annotated_path is not None
        ):
            image_paths.append(
                annotated_path
            )

        start = time.perf_counter()

        response, attempts = (
            self._call_edit_with_retry(
                image_paths=image_paths,
                mask_path=mask_path,
                prompt=full_prompt,
            )
        )

        record.attempts = attempts

        self._save_response_image(
            response=response,
            output_path=output_path,
        )

        record.elapsed_seconds = round(
            time.perf_counter()
            - start,
            3,
        )
        record.status = "generated"
        return record

    # ======================================================
    # Python composition branch
    # ======================================================

    def _process_composition_task(
        self,
        *,
        raw_task: dict[str, Any],
        image_inputs: dict[str, dict[str, Any]],
        assembly_previous_output: Optional[Path],
        output_path: Path,
        record: TaskRecord,
        previews_dir: Path,
        dry_run: bool,
        package: dict[str, Any],
    ) -> TaskRecord:
        if assembly_previous_output is None:
            expected = raw_task.get(
                "previous_output_filename"
            )
            raise FileNotFoundError(
                "Comparison task requires the final assembly output, "
                f"but none is available. Expected: {expected}"
            )

        reference_path = self._resolve_required_image(
            image_inputs.get(
                "reference_image"
            ),
            role="reference_image",
        )

        record.base_image_path = str(
            assembly_previous_output
        )
        record.reference_image_path = str(
            reference_path
        )
        record.actual_api_operation = (
            "python_pillow_composition"
        )

        if assembly_previous_output.is_file():
            self._validate_input_image(
                assembly_previous_output
            )
        elif not dry_run:
            raise FileNotFoundError(
                "Final assembly image not found: "
                f"{assembly_previous_output}"
            )
        else:
            record.warnings.append(
                "Dry run: final assembly output does not exist yet."
            )

        self._validate_input_image(
            reference_path
        )

        if dry_run:
            record.status = "dry_run"
            self._save_request_preview(
                previews_dir=previews_dir,
                raw_task=raw_task,
                actual_operation=(
                    record.actual_api_operation
                ),
                full_prompt=(
                    "Python comparison composition; "
                    "no API prompt is submitted."
                ),
                base_image_path=(
                    assembly_previous_output
                ),
                reference_image_path=(
                    reference_path
                ),
                annotated_image_path=None,
                mask_path=None,
                output_path=output_path,
            )
            return record

        start = time.perf_counter()

        self._compose_comparison_panel(
            corrected_path=(
                assembly_previous_output
            ),
            reference_path=reference_path,
            output_path=output_path,
            model_id=str(
                package.get(
                    "model_id",
                    "",
                )
            ),
            step_id=str(
                package.get(
                    "step_id",
                    "",
                )
            ),
        )

        record.elapsed_seconds = round(
            time.perf_counter()
            - start,
            3,
        )
        record.attempts = 0
        record.status = "generated"
        return record

    # ======================================================
    # OpenAI calls
    # ======================================================

    def _call_generate_with_retry(
        self,
        *,
        prompt: str,
    ) -> tuple[Any, int]:
        last_error: Optional[
            Exception
        ] = None

        for attempt in range(
            1,
            self.max_retries + 2,
        ):
            try:
                response = (
                    self.client.images.generate(
                        model=self.model,
                        prompt=prompt,
                        size=self.size,
                        quality=self.quality,
                        output_format=(
                            self.output_format
                        ),
                        background=self.background,
                        n=1,
                    )
                )

                return response, attempt

            except Exception as exc:
                last_error = exc

                if attempt >= (
                    self.max_retries + 1
                ):
                    break

                delay = (
                    self.retry_delay_seconds
                    * attempt
                )

                print(
                    "[WARNING] images.generate "
                    f"attempt {attempt} failed: {exc}",
                    file=sys.stderr,
                )
                print(
                    f"[WARNING] Retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(
                    delay
                )

        assert last_error is not None
        raise last_error

    def _call_edit_with_retry(
        self,
        *,
        image_paths: list[Path],
        mask_path: Optional[Path],
        prompt: str,
    ) -> tuple[Any, int]:
        last_error: Optional[
            Exception
        ] = None

        for attempt in range(
            1,
            self.max_retries + 2,
        ):
            try:
                with ExitStack() as stack:
                    image_files: list[
                        BinaryIO
                    ] = []

                    for path in image_paths:
                        image_files.append(
                            stack.enter_context(
                                path.open(
                                    "rb"
                                )
                            )
                        )

                    mask_file: Optional[
                        BinaryIO
                    ] = None

                    if mask_path is not None:
                        mask_file = (
                            stack.enter_context(
                                mask_path.open(
                                    "rb"
                                )
                            )
                        )

                    request: dict[
                        str,
                        Any,
                    ] = {
                        "model": self.model,
                        "image": image_files,
                        "prompt": prompt,
                        "size": self.size,
                        "quality": self.quality,
                        "output_format": (
                            self.output_format
                        ),
                        "background": (
                            self.background
                        ),
                        "n": 1,
                    }

                    if mask_file is not None:
                        request["mask"] = (
                            mask_file
                        )

                    response = (
                        self.client.images.edit(
                            **request
                        )
                    )

                return response, attempt

            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()

                # 400 類型的固定請求錯誤，重試也不會成功。
                non_retryable_markers = (
                    "invalid_input_fidelity_model",
                    "invalid_parameter",
                    "invalid_request_error",
                    "unsupported",
                    "does not support",
                )

                if any(marker in error_text for marker in non_retryable_markers):
                    raise

                if attempt >= (
                    self.max_retries + 1
                ):
                    break

                delay = (
                    self.retry_delay_seconds
                    * attempt
                )

                print(
                    "[WARNING] images.edit "
                    f"attempt {attempt} failed: {exc}",
                    file=sys.stderr,
                )
                print(
                    f"[WARNING] Retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(
                    delay
                )

        assert last_error is not None
        raise last_error

    @staticmethod
    def _save_response_image(
        *,
        response: Any,
        output_path: Path,
    ) -> None:
        data = getattr(
            response,
            "data",
            None,
        )

        if not data:
            raise RuntimeError(
                "OpenAI Image API returned no image data."
            )

        first_item = data[0]

        image_base64 = getattr(
            first_item,
            "b64_json",
            None,
        )

        if (
            not image_base64
            and isinstance(
                first_item,
                dict,
            )
        ):
            image_base64 = (
                first_item.get(
                    "b64_json"
                )
            )

        if not image_base64:
            raise RuntimeError(
                "OpenAI Image API response does not contain b64_json."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_bytes(
            base64.b64decode(
                image_base64
            )
        )

    # ======================================================
    # Mask
    # ======================================================

    def _maybe_create_mask(
        self,
        *,
        raw_task: dict[str, Any],
        base_image_path: Path,
        masks_dir: Path,
        dry_run: bool,
    ) -> tuple[
        Optional[Path],
        Optional[str],
    ]:
        image_task = raw_task.get(
            "image_task",
            {},
        )

        if not isinstance(
            image_task,
            dict,
        ):
            return (
                None,
                "Mask skipped: invalid image_task.",
            )

        if not bool(
            image_task.get(
                "use_mask_if_available",
                False,
            )
        ):
            return (
                None,
                "Mask skipped by task configuration.",
            )

        bbox = self._normalize_bbox(
            raw_task.get(
                "bbox"
            )
        )

        if bbox is None:
            return (
                None,
                "Mask skipped because no bbox is available.",
            )

        hint = str(
            raw_task.get(
                "localization_hint_en",
                "",
            )
        ).lower()

        # bbox 若源自 reference image，不能直接套在 test/previous base。
        if (
            "annotated reference image"
            in hint
        ):
            return (
                None,
                "Mask skipped because the bbox originates from the "
                "reference image and may not align with the editable base.",
            )

        if not base_image_path.is_file():
            if dry_run:
                return (
                    None,
                    "Dry run: mask skipped because the future base image "
                    "does not exist yet.",
                )

            raise FileNotFoundError(
                "Cannot create mask because base image is missing: "
                f"{base_image_path}"
            )

        with Image.open(
            base_image_path
        ) as image:
            width, height = (
                image.size
            )

        x1, y1, x2, y2 = (
            self._clip_and_pad_bbox(
                bbox=bbox,
                width=width,
                height=height,
                padding_ratio=(
                    self.mask_padding_ratio
                ),
            )
        )

        # GPT Image mask:
        # transparent region = editable guidance
        # opaque region = preserve guidance
        mask = Image.new(
            "RGBA",
            (width, height),
            (255, 255, 255, 255),
        )

        draw = ImageDraw.Draw(
            mask
        )

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            fill=(
                0,
                0,
                0,
                0,
            ),
        )

        mask_path = (
            masks_dir
            / (
                f"sequence_"
                f"{int(raw_task.get('sequence_index', 0)):02d}"
                f"_mask.png"
            )
        )

        mask.save(
            mask_path,
            format="PNG",
            optimize=True,
        )

        if (
            mask_path.stat().st_size
            >= MAX_MASK_FILE_BYTES
        ):
            mask_path.unlink(
                missing_ok=True
            )
            return (
                None,
                "Mask skipped because the PNG exceeds 4 MB.",
            )

        return mask_path, None

    @staticmethod
    def _clip_and_pad_bbox(
        *,
        bbox: list[float],
        width: int,
        height: int,
        padding_ratio: float,
    ) -> tuple[
        int,
        int,
        int,
        int,
    ]:
        x1, y1, x2, y2 = bbox

        if (
            x2 <= x1
            or y2 <= y1
        ):
            raise ValueError(
                f"Invalid bbox: {bbox}"
            )

        box_width = x2 - x1
        box_height = y2 - y1

        pad_x = (
            box_width
            * padding_ratio
        )
        pad_y = (
            box_height
            * padding_ratio
        )

        clipped_x1 = max(
            0,
            int(
                round(
                    x1 - pad_x
                )
            ),
        )
        clipped_y1 = max(
            0,
            int(
                round(
                    y1 - pad_y
                )
            ),
        )
        clipped_x2 = min(
            width,
            int(
                round(
                    x2 + pad_x
                )
            ),
        )
        clipped_y2 = min(
            height,
            int(
                round(
                    y2 + pad_y
                )
            ),
        )

        if (
            clipped_x2 <= clipped_x1
            or clipped_y2 <= clipped_y1
        ):
            raise ValueError(
                "BBox does not overlap image bounds: "
                f"bbox={bbox}, image={width}x{height}"
            )

        return (
            clipped_x1,
            clipped_y1,
            clipped_x2,
            clipped_y2,
        )

    # ======================================================
    # Comparison composition
    # ======================================================

    def _compose_comparison_panel(
        self,
        *,
        corrected_path: Path,
        reference_path: Path,
        output_path: Path,
        model_id: str,
        step_id: str,
    ) -> None:
        with Image.open(
            corrected_path
        ) as corrected_raw:
            corrected = (
                corrected_raw
                .convert("RGB")
            )

        with Image.open(
            reference_path
        ) as reference_raw:
            reference = (
                reference_raw
                .convert("RGB")
            )

        panel_width = 900
        panel_height = 675
        title_height = 90
        label_height = 56
        margin = 36
        gap = 28

        canvas_width = (
            margin * 2
            + panel_width * 2
            + gap
        )
        canvas_height = (
            margin * 2
            + title_height
            + label_height
            + panel_height
        )

        canvas = Image.new(
            "RGB",
            (
                canvas_width,
                canvas_height,
            ),
            "white",
        )

        corrected_fit = ImageOps.contain(
            corrected,
            (
                panel_width,
                panel_height,
            ),
        )

        reference_fit = ImageOps.contain(
            reference,
            (
                panel_width,
                panel_height,
            ),
        )

        left_x = margin
        right_x = (
            margin
            + panel_width
            + gap
        )
        image_y = (
            margin
            + title_height
            + label_height
        )

        self._paste_centered(
            canvas=canvas,
            image=corrected_fit,
            box=(
                left_x,
                image_y,
                panel_width,
                panel_height,
            ),
        )

        self._paste_centered(
            canvas=canvas,
            image=reference_fit,
            box=(
                right_x,
                image_y,
                panel_width,
                panel_height,
            ),
        )

        draw = ImageDraw.Draw(
            canvas
        )

        font_title = self._load_font(
            size=40
        )
        font_label = self._load_font(
            size=28
        )

        title = (
            f"{model_id} {step_id} "
            "Correction Comparison"
        )

        self._draw_centered_text(
            draw=draw,
            text=title,
            y=margin,
            canvas_width=canvas_width,
            font=font_title,
        )

        self._draw_centered_in_box(
            draw=draw,
            text="Corrected Result",
            x=left_x,
            y=(
                margin
                + title_height
            ),
            width=panel_width,
            font=font_label,
        )

        self._draw_centered_in_box(
            draw=draw,
            text="Correct Reference",
            x=right_x,
            y=(
                margin
                + title_height
            ),
            width=panel_width,
            font=font_label,
        )

        border_top = image_y
        border_bottom = (
            image_y
            + panel_height
        )

        draw.rectangle(
            [
                left_x,
                border_top,
                left_x + panel_width,
                border_bottom,
            ],
            outline="black",
            width=2,
        )

        draw.rectangle(
            [
                right_x,
                border_top,
                right_x + panel_width,
                border_bottom,
            ],
            outline="black",
            width=2,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_format = (
            "JPEG"
            if self.output_format == "jpeg"
            else self.output_format.upper()
        )

        save_kwargs: dict[
            str,
            Any,
        ] = {}

        if save_format == "JPEG":
            save_kwargs["quality"] = 95

        canvas.save(
            output_path,
            format=save_format,
            **save_kwargs,
        )

    @staticmethod
    def _paste_centered(
        *,
        canvas: Image.Image,
        image: Image.Image,
        box: tuple[
            int,
            int,
            int,
            int,
        ],
    ) -> None:
        x, y, width, height = box

        paste_x = (
            x
            + (width - image.width) // 2
        )
        paste_y = (
            y
            + (height - image.height) // 2
        )

        canvas.paste(
            image,
            (
                paste_x,
                paste_y,
            ),
        )

    @staticmethod
    def _load_font(
        *,
        size: int,
    ) -> ImageFont.ImageFont:
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        for candidate in candidates:
            path = Path(
                candidate
            )
            if path.is_file():
                try:
                    return ImageFont.truetype(
                        str(path),
                        size=size,
                    )
                except OSError:
                    continue

        return ImageFont.load_default()

    @staticmethod
    def _draw_centered_text(
        *,
        draw: ImageDraw.ImageDraw,
        text: str,
        y: int,
        canvas_width: int,
        font: ImageFont.ImageFont,
    ) -> None:
        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        text_width = (
            bbox[2]
            - bbox[0]
        )

        x = (
            canvas_width
            - text_width
        ) // 2

        draw.text(
            (
                x,
                y,
            ),
            text,
            fill="black",
            font=font,
        )

    @staticmethod
    def _draw_centered_in_box(
        *,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        width: int,
        font: ImageFont.ImageFont,
    ) -> None:
        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        text_width = (
            bbox[2]
            - bbox[0]
        )

        draw.text(
            (
                x
                + (
                    width
                    - text_width
                ) // 2,
                y,
            ),
            text,
            fill="black",
            font=font,
        )

    # ======================================================
    # Prompt / input helpers
    # ======================================================

    @staticmethod
    def _combine_prompts(
        *,
        global_policy: str,
        step_prompt: str,
        global_negative: str,
        step_negative: str,
        api_mode: str,
    ) -> str:
        if api_mode == "generate":
            role_block = (
                "REFERENCE-GUIDED GENERATION RULES:\n"
                "If an input image is provided, use it only to identify "
                "the target component. Produce a new standalone product shot; "
                "do not reproduce the full assembly photograph."
            )

        elif api_mode == "edit":
            role_block = (
                "EDITING RULES:\n"
                "The first input image is the editable base. "
                "Additional images are references only. "
                "Return only the edited version of the first image."
            )

        else:
            role_block = (
                "PYTHON COMPOSITION TASK:\n"
                "This instruction is metadata only and is not submitted "
                "to the image model."
            )

        negative = (
            step_negative.strip()
            or global_negative.strip()
        )

        blocks = [
            role_block,
            (
                "GLOBAL POLICY:\n"
                + global_policy.strip()
            ),
            (
                "CURRENT TASK:\n"
                + step_prompt.strip()
            ),
            (
                "PROHIBITED CHANGES:\n"
                + negative
            ),
        ]

        return "\n\n".join(
            block
            for block in blocks
            if block.strip()
        )

    @staticmethod
    def _index_image_inputs(
        raw_inputs: Any,
    ) -> dict[
        str,
        dict[str, Any],
    ]:
        if not isinstance(
            raw_inputs,
            list,
        ):
            return {}

        indexed: dict[
            str,
            dict[str, Any],
        ] = {}

        for item in raw_inputs:
            if not isinstance(
                item,
                dict,
            ):
                continue

            role = str(
                item.get(
                    "role",
                    "",
                )
            ).strip()

            if role:
                indexed[role] = item

        return indexed

    def _resolve_required_image(
        self,
        image_input: Optional[
            dict[str, Any]
        ],
        *,
        role: str,
    ) -> Path:
        if not image_input:
            raise KeyError(
                f"Required image input is missing: {role}"
            )

        value = image_input.get(
            "path"
        )

        if not value:
            raise FileNotFoundError(
                f"Required image path is empty: {role}"
            )

        return self._resolve_path(
            value
        )

    def _resolve_optional_image(
        self,
        image_input: Optional[
            dict[str, Any]
        ],
    ) -> Optional[Path]:
        if not image_input:
            return None

        value = image_input.get(
            "path"
        )

        if not value:
            return None

        return self._resolve_path(
            value
        )

    @staticmethod
    def _resolve_path(
        value: Any,
    ) -> Path:
        path = Path(
            str(value)
        ).expanduser()

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            )

        return path.resolve()

    @staticmethod
    def _validate_input_image(
        path: Path,
    ) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Input image not found: {path}"
            )

        if (
            path.suffix.lower()
            not in SUPPORTED_INPUT_SUFFIXES
        ):
            raise ValueError(
                "Unsupported input image format: "
                f"{path.suffix} ({path})"
            )

        size_bytes = (
            path.stat().st_size
        )

        if (
            size_bytes
            >= MAX_INPUT_FILE_BYTES
        ):
            raise ValueError(
                "Input image exceeds 49 MB safety limit: "
                f"{size_bytes / 1024 / 1024:.2f} MB ({path})"
            )

        try:
            with Image.open(
                path
            ) as image:
                image.verify()

        except Exception as exc:
            raise ValueError(
                f"Invalid or unreadable image: {path}"
            ) from exc

    # ======================================================
    # Dry-run preview and manifest
    # ======================================================

    def _save_request_preview(
        self,
        *,
        previews_dir: Path,
        raw_task: dict[str, Any],
        actual_operation: str,
        full_prompt: str,
        base_image_path: Optional[Path],
        reference_image_path: Optional[Path],
        annotated_image_path: Optional[Path],
        mask_path: Optional[Path],
        output_path: Path,
    ) -> None:
        sequence_index = int(
            raw_task.get(
                "sequence_index",
                0,
            )
        )
        action = str(
            raw_task.get(
                "action",
                "unknown",
            )
        )

        preview_path = (
            previews_dir
            / (
                f"sequence_{sequence_index:02d}_"
                f"{action}_request.json"
            )
        )

        payload = {
            "model": self.model,
            "actual_operation": actual_operation,
            "size": self.size,
            "quality": self.quality,
            "output_format": self.output_format,
            "background": self.background,
            "base_image": (
                str(base_image_path)
                if base_image_path
                else None
            ),
            "reference_image": (
                str(reference_image_path)
                if reference_image_path
                else None
            ),
            "annotated_image": (
                str(annotated_image_path)
                if annotated_image_path
                else None
            ),
            "mask": (
                str(mask_path)
                if mask_path
                else None
            ),
            "output_path": str(
                output_path
            ),
            "prompt": full_prompt,
        }

        preview_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _save_manifest(
        path: Path,
        manifest: GenerationManifestV2,
    ) -> None:
        path.write_text(
            json.dumps(
                manifest.to_dict(),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _write_markdown_manifest(
        path: Path,
        manifest: GenerationManifestV2,
    ) -> None:
        lines = [
            "# Step Image Generator V2 Manifest",
            "",
            f"- Created：`{manifest.created_at}`",
            f"- Model：`{manifest.image_model}`",
            f"- Size：`{manifest.size}`",
            f"- Quality：`{manifest.quality}`",
            f"- Input Fidelity：`{manifest.input_fidelity}`",
            f"- Dry Run：`{manifest.dry_run}`",
            f"- Use Mask：`{manifest.use_mask}`",
            f"- Requested：`{manifest.requested_task_count}`",
            f"- Successful：`{manifest.successful_task_count}`",
            f"- Existing：`{manifest.existing_task_count}`",
            f"- Failed：`{manifest.failed_task_count}`",
            f"- Skipped：`{manifest.skipped_task_count}`",
            f"- Final Assembly：`{manifest.final_assembly_image_path}`",
            f"- Final Comparison：`{manifest.final_comparison_image_path}`",
            "",
        ]

        if manifest.warnings:
            lines.extend(
                [
                    "## Warnings",
                    "",
                ]
            )
            lines.extend(
                f"- {warning}"
                for warning in manifest.warnings
            )
            lines.append("")

        lines.extend(
            [
                "## Tasks",
                "",
            ]
        )

        for task in manifest.tasks:
            lines.extend(
                [
                    (
                        f"### Sequence "
                        f"{task.sequence_index}"
                        f"｜{task.title}"
                    ),
                    "",
                    f"- Action：`{task.action}`",
                    f"- Branch：`{task.branch}`",
                    f"- API Mode：`{task.api_mode}`",
                    f"- Actual Operation：`{task.actual_api_operation}`",
                    f"- Status：`{task.status}`",
                    f"- Base：`{task.base_image_path}`",
                    f"- Reference：`{task.reference_image_path}`",
                    f"- Annotated：`{task.annotated_image_path}`",
                    f"- Mask：`{task.mask_path}`",
                    f"- Output：`{task.output_path}`",
                    f"- Attempts：`{task.attempts}`",
                    f"- Time：`{task.elapsed_seconds}`",
                ]
            )

            if task.error_message:
                lines.append(
                    f"- Error：`{task.error_type}: "
                    f"{task.error_message}`"
                )

            if task.warnings:
                lines.append(
                    "- Warnings:"
                )
                lines.extend(
                    f"  - {warning}"
                    for warning in task.warnings
                )

            lines.append("")

        path.write_text(
            "\n".join(
                lines
            ),
            encoding="utf-8",
        )

    def _copy_final_outputs(
        self,
        *,
        manifest: GenerationManifestV2,
        output_root: Path,
        dry_run: bool,
    ) -> None:
        if dry_run:
            return

        if manifest.final_assembly_image_path:
            source = Path(
                manifest.final_assembly_image_path
            )

            if source.is_file():
                destination = (
                    output_root
                    / (
                        "final_assembly."
                        f"{self.output_format}"
                    )
                )

                if source.resolve() != destination.resolve():
                    shutil.copy2(
                        source,
                        destination,
                    )

                manifest.final_assembly_image_path = str(
                    destination
                )

        if manifest.final_comparison_image_path:
            source = Path(
                manifest.final_comparison_image_path
            )

            if source.is_file():
                destination = (
                    output_root
                    / (
                        "final_comparison."
                        f"{self.output_format}"
                    )
                )

                if source.resolve() != destination.resolve():
                    shutil.copy2(
                        source,
                        destination,
                    )

                manifest.final_comparison_image_path = str(
                    destination
                )

    # ======================================================
    # Validation / utility
    # ======================================================

    def _validate_settings(
        self,
    ) -> None:
        if (
            self.model != "gpt-image-2"
            and not self.model.startswith(
                "gpt-image-2-"
            )
        ):
            raise ValueError(
                "This generator is configured for gpt-image-2."
            )

        if self.size not in SUPPORTED_SIZES:
            raise ValueError(
                f"Unsupported size: {self.size}. "
                f"Choose from {sorted(SUPPORTED_SIZES)}."
            )

        if (
            self.quality
            not in SUPPORTED_QUALITIES
        ):
            raise ValueError(
                f"Unsupported quality: {self.quality}"
            )

        if (
            self.output_format
            not in SUPPORTED_OUTPUT_FORMATS
        ):
            raise ValueError(
                "Unsupported output format: "
                f"{self.output_format}"
            )

        if (
            self.input_fidelity
            not in SUPPORTED_INPUT_FIDELITIES
        ):
            raise ValueError(
                "Unsupported input fidelity: "
                f"{self.input_fidelity}"
            )

        if self.background not in {
            "auto",
            "opaque",
            "transparent",
        }:
            raise ValueError(
                "background must be auto, opaque, or transparent."
            )

        if self.mask_padding_ratio < 0:
            raise ValueError(
                "mask_padding_ratio cannot be negative."
            )

        if self.max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative."
            )

    @staticmethod
    def _validate_prompt_package(
        package: dict[str, Any],
    ) -> None:
        schema_version = str(
            package.get(
                "schema_version",
                "",
            )
        )

        if not schema_version.startswith(
            "2."
        ):
            raise ValueError(
                "This generator requires a V2 prompt package. "
                f"Received schema_version={schema_version!r}."
            )

        tasks = package.get(
            "step_prompts"
        )

        if not isinstance(
            tasks,
            list,
        ):
            raise TypeError(
                "step_prompts must be a list."
            )

    @staticmethod
    def _select_tasks(
        *,
        raw_tasks: list[Any],
        branch_filter: str,
        max_tasks: Optional[int],
    ) -> list[Any]:
        selected: list[Any] = []

        for task in raw_tasks:
            if (
                branch_filter
                != "all"
                and isinstance(
                    task,
                    dict,
                )
            ):
                image_task = task.get(
                    "image_task",
                    {},
                )

                branch = (
                    str(
                        image_task.get(
                            "branch",
                            "",
                        )
                    )
                    if isinstance(
                        image_task,
                        dict,
                    )
                    else ""
                )

                if branch != branch_filter:
                    continue

            selected.append(
                task
            )

        if max_tasks is not None:
            if max_tasks < 1:
                raise ValueError(
                    "--max-tasks must be at least 1."
                )

            selected = selected[
                :max_tasks
            ]

        return selected

    def _failed_record(
        self,
        *,
        sequence_index: int,
        sop_step_no: int,
        action: str,
        title: str,
        api_mode: str,
        branch: str,
        exc: Exception,
    ) -> TaskRecord:
        return TaskRecord(
            sequence_index=sequence_index,
            sop_step_no=sop_step_no,
            action=action,
            title=title,
            api_mode=api_mode,
            branch=branch,
            status="failed",
            model=self.model,
            size=self.size,
            quality=self.quality,
            output_format=self.output_format,
            error_type=(
                type(exc).__name__
            ),
            error_message=str(
                exc
            ),
        )

    @staticmethod
    def _normalize_bbox(
        value: Any,
    ) -> Optional[list[float]]:
        if not isinstance(
            value,
            list,
        ):
            return None

        if len(value) != 4:
            return None

        try:
            return [
                float(item)
                for item in value
            ]

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(
                f"JSON file not found: {path}"
            )

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise TypeError(
                f"Expected JSON object: {path}"
            )

        return payload


def find_latest_prompts_json(
) -> Path:
    files = sorted(
        DEFAULT_PROMPTS_ROOT.glob(
            "*/step_prompts_v2.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            "No step_prompts_v2.json found under:\n"
            f"{DEFAULT_PROMPTS_ROOT}"
        )

    return files[0]


def build_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute V2.1 image tasks with gpt-image-2."
        )
    )

    parser.add_argument(
        "--prompts-json",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--size",
        choices=sorted(
            SUPPORTED_SIZES
        ),
        default=DEFAULT_SIZE,
    )

    parser.add_argument(
        "--quality",
        choices=sorted(
            SUPPORTED_QUALITIES
        ),
        default=DEFAULT_QUALITY,
    )

    parser.add_argument(
        "--output-format",
        choices=sorted(
            SUPPORTED_OUTPUT_FORMATS
        ),
        default=DEFAULT_OUTPUT_FORMAT,
    )

    parser.add_argument(
        "--background",
        choices=[
            "auto",
            "opaque",
            "transparent",
        ],
        default=DEFAULT_BACKGROUND,
    )

    parser.add_argument(
        "--input-fidelity",
        choices=sorted(
            SUPPORTED_INPUT_FIDELITIES
        ),
        default=DEFAULT_INPUT_FIDELITY,
    )

    parser.add_argument(
        "--allow-manual-review",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--branch",
        choices=[
            "all",
            "standalone",
            "assembly",
            "composition",
        ],
        default="all",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--use-mask",
        action="store_true",
    )

    parser.add_argument(
        "--mask-padding-ratio",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--exclude-annotated-image",
        action="store_true",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--retry-delay",
        type=float,
        default=3.0,
    )

    return parser


def main(
) -> int:
    load_dotenv(
        PROJECT_ROOT
        / ".env"
    )

    args = build_parser().parse_args()

    prompts_json = (
        args.prompts_json
        .expanduser()
        .resolve()
        if args.prompts_json
        is not None
        else find_latest_prompts_json()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
        if args.output_dir
        is not None
        else (
            prompts_json.parent
            / "generated_steps_v2"
        )
    )

    if (
        not args.dry_run
        and not os.getenv(
            "OPENAI_API_KEY"
        )
    ):
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. "
            "Add it to the project-root .env file."
        )

    generator = StepImageGeneratorV2(
        model=args.model,
        size=args.size,
        quality=args.quality,
        output_format=(
            args.output_format
        ),
        background=args.background,
        input_fidelity=(
            args.input_fidelity
        ),
        use_mask=args.use_mask,
        mask_padding_ratio=(
            args.mask_padding_ratio
        ),
        include_annotated_image=(
            not args.exclude_annotated_image
        ),
        max_retries=args.max_retries,
        retry_delay_seconds=(
            args.retry_delay
        ),
    )

    manifest = generator.run(
        prompts_json_path=prompts_json,
        output_dir=output_dir,
        allow_manual_review=(
            args.allow_manual_review
        ),
        dry_run=args.dry_run,
        max_tasks=args.max_tasks,
        branch_filter=args.branch,
        continue_on_error=(
            args.continue_on_error
        ),
        overwrite=args.overwrite,
    )

    print("=" * 70)
    print("Step Image Generator V2 finished")
    print("=" * 70)
    print(f"Source prompts:   {prompts_json}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run:          {manifest.dry_run}")
    print(f"Requested:        {manifest.requested_task_count}")
    print(f"Successful:       {manifest.successful_task_count}")
    print(f"Existing:         {manifest.existing_task_count}")
    print(f"Failed:           {manifest.failed_task_count}")
    print(f"Skipped:          {manifest.skipped_task_count}")
    print(f"Final assembly:   {manifest.final_assembly_image_path}")
    print(f"Final comparison: {manifest.final_comparison_image_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise