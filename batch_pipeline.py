"""
batch_pipeline.py

批次處理 logs/current_parsed_json 中的既有 Vision JSON。

預設行為
--------
1. 掃描 logs/current_parsed_json/*_parsed_*.json
2. 對每一張原始圖片，只選最新的一份 parsed JSON
3. 每張圖片建立獨立輸出資料夾，避免互相覆蓋
4. 依序執行：
   existing parsed JSON
       -> pipeline_smoke_test.process_one()
       -> correction_sop.json
       -> step_prompts_v2.json
       -> （可選）gpt-image-2 步驟圖片
       -> assembly_instruction_book.png
5. 寫出 batch_summary.json、batch_summary.csv

重要安全設計
------------
- 預設不呼叫 OpenAI Image API。
- 要正式生圖必須明確加入 --generate-images。
- 可先使用 --image-dry-run 驗證 58 組 Prompt 與路徑。
- 每次執行預設建立新的時間戳 batch 資料夾。
- 同一 batch 中，每張原始圖片使用自己的 case folder。
- 不依賴任何「最新 output」函式；各階段都傳入明確路徑。
- 單一案例失敗時，預設繼續處理下一案例。

預設輸出
--------
output/batch_runs/<batch_id>/
├── <image_stem_01>/
│   ├── results.json
│   ├── run_summary.json
│   ├── correction_sop.json
│   ├── correction_sop.md
│   ├── step_prompts_v2.json
│   ├── step_prompts_v2.md
│   ├── generated_steps_v2/              # 生圖或 image dry-run 時建立
│   └── assembly_instruction_book.png    # 有正式圖或 placeholder
├── <image_stem_02>/
│   └── ...
├── batch_summary.json
└── batch_summary.csv

常用指令
--------
先測前三份，不呼叫圖片 API：

python batch_pipeline.py --limit 3

前三份做 Image dry-run：

python batch_pipeline.py --limit 3 --image-dry-run

只正式測一份圖片：

python batch_pipeline.py ^
  --limit 1 ^
  --generate-images ^
  --allow-manual-review

正式跑全部：

python batch_pipeline.py ^
  --generate-images ^
  --allow-manual-review

重新使用指定批次資料夾：

python batch_pipeline.py ^
  --batch-id 20260805_153000 ^
  --resume

重新產生既有內容：

python batch_pipeline.py ^
  --batch-id 20260805_153000 ^
  --resume ^
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from correction_sop_generator import CorrectionSOPGenerator
from instruction_book_generator import InstructionBookGenerator
from pipeline_smoke_test import (
    image_stem_from_parsed_json,
    parsed_timestamp_key,
    process_one,
)
from step_prompt_builder_v2 import StepPromptBuilderV2


PROJECT_ROOT = Path(__file__).resolve().parent
PARSED_JSON_DIR = PROJECT_ROOT / "logs" / "current_parsed_json"
BATCH_RUNS_ROOT = PROJECT_ROOT / "output" / "batch_runs"

STEP_IMAGE_GENERATOR_PATH = PROJECT_ROOT / "step_image_generator_v2.py"

PARSED_FILE_PATTERN = "*_parsed_*.json"
SAFE_FOLDER_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class StageResult:
    name: str
    status: str
    output_path: Optional[str] = None
    elapsed_seconds: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class CaseResult:
    index: int
    image_stem: str
    parsed_json_path: str
    case_output_dir: str
    status: str
    stages: list[StageResult] = field(default_factory=list)
    final_instruction_path: Optional[str] = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchSummary:
    schema_version: str
    batch_id: str
    created_at: str
    finished_at: Optional[str]
    parsed_json_dir: str
    batch_output_dir: str
    latest_per_image: bool
    generate_images: bool
    image_dry_run: bool
    allow_manual_review: bool
    overwrite: bool
    discovered_json_count: int
    selected_case_count: int
    successful_case_count: int
    failed_case_count: int
    skipped_case_count: int
    elapsed_seconds: float
    cases: list[CaseResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def safe_folder_name(value: str) -> str:
    cleaned = SAFE_FOLDER_PATTERN.sub("_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed_case"


def find_all_parsed_files(parsed_dir: Path) -> list[Path]:
    parsed_dir = parsed_dir.expanduser().resolve()

    if not parsed_dir.is_dir():
        raise FileNotFoundError(
            f"Parsed JSON directory not found: {parsed_dir}"
        )

    return sorted(
        (
            path
            for path in parsed_dir.glob(PARSED_FILE_PATTERN)
            if path.is_file()
        ),
        key=lambda path: (
            image_stem_from_parsed_json(path),
            parsed_timestamp_key(path),
        ),
    )


def select_latest_per_image(parsed_files: list[Path]) -> list[Path]:
    """
    對每個 image stem，只保留時間戳最新的一份 JSON。

    注意：
    這不是整個資料夾只選一份最新檔案；
    而是每一張原始圖片各自選最新一份。
    """
    latest: dict[str, Path] = {}

    for path in parsed_files:
        image_stem = image_stem_from_parsed_json(path)
        current = latest.get(image_stem)

        if (
            current is None
            or parsed_timestamp_key(path)
            > parsed_timestamp_key(current)
        ):
            latest[image_stem] = path

    return sorted(
        latest.values(),
        key=lambda path: image_stem_from_parsed_json(path),
    )


def filter_parsed_files(
    parsed_files: list[Path],
    *,
    contains: Optional[str],
    limit: Optional[int],
    offset: int,
) -> list[Path]:
    selected = parsed_files

    if contains:
        needle = contains.lower()
        selected = [
            path
            for path in selected
            if needle
            in image_stem_from_parsed_json(path).lower()
        ]

    if offset < 0:
        raise ValueError("--offset cannot be negative.")

    selected = selected[offset:]

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1.")
        selected = selected[:limit]

    return selected


def validate_parsed_json(path: Path) -> tuple[bool, str]:
    try:
        payload = load_json(path)
    except Exception as exc:
        return False, str(exc)

    if not bool(payload.get("success", False)):
        return False, "Parsed Vision JSON has success=false."

    model_response = payload.get("model_response")

    if not isinstance(model_response, dict):
        return False, "model_response is missing or invalid."

    file_info = payload.get("file_info")

    if not isinstance(file_info, dict):
        return False, "file_info is missing or invalid."

    return True, ""


def prepare_batch_directory(
    *,
    batch_root: Path,
    batch_id: Optional[str],
    resume: bool,
) -> tuple[str, Path]:
    resolved_root = batch_root.expanduser().resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)

    resolved_batch_id = (
        batch_id
        or datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    batch_dir = resolved_root / safe_folder_name(resolved_batch_id)

    if batch_dir.exists():
        if not resume:
            raise FileExistsError(
                f"Batch directory already exists: {batch_dir}\n"
                "Use --resume to continue using it, or choose another --batch-id."
            )
    else:
        batch_dir.mkdir(parents=True, exist_ok=False)

    return resolved_batch_id, batch_dir


def stage_success(
    name: str,
    output_path: Optional[Path],
    started_at: float,
    *,
    status: str = "success",
) -> StageResult:
    return StageResult(
        name=name,
        status=status,
        output_path=str(output_path) if output_path else None,
        elapsed_seconds=round(time.perf_counter() - started_at, 3),
    )


def stage_failure(
    name: str,
    exc: Exception,
    started_at: float,
) -> StageResult:
    return StageResult(
        name=name,
        status="failed",
        elapsed_seconds=round(time.perf_counter() - started_at, 3),
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


def run_image_generator(
    *,
    prompts_json_path: Path,
    output_dir: Path,
    generate_images: bool,
    image_dry_run: bool,
    allow_manual_review: bool,
    overwrite: bool,
    quality: str,
    size: str,
    image_max_tasks: Optional[int],
    image_continue_on_error: bool,
) -> Path:
    """
    以 subprocess 執行 step_image_generator_v2.py。

    使用 subprocess 的原因：
    - 尊重使用者目前專案中已修正過的最新版 generator。
    - 避免 batch_pipeline 與 OpenAI SDK 物件互相綁死。
    - 每個案例的 stdout/stderr 可獨立寫入 log。
    """
    if not STEP_IMAGE_GENERATOR_PATH.is_file():
        raise FileNotFoundError(
            f"step_image_generator_v2.py not found: "
            f"{STEP_IMAGE_GENERATOR_PATH}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(STEP_IMAGE_GENERATOR_PATH),
        "--prompts-json",
        str(prompts_json_path),
        "--output-dir",
        str(output_dir),
        "--quality",
        quality,
        "--size",
        size,
    ]

    if image_dry_run:
        command.append("--dry-run")

    if generate_images and allow_manual_review:
        command.append("--allow-manual-review")

    if overwrite:
        command.append("--overwrite")

    if image_max_tasks is not None:
        command.extend(
            [
                "--max-tasks",
                str(image_max_tasks),
            ]
        )

    if image_continue_on_error:
        command.append("--continue-on-error")

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    (output_dir / "batch_image_generator_stdout.txt").write_text(
        completed.stdout,
        encoding="utf-8",
    )

    (output_dir / "batch_image_generator_stderr.txt").write_text(
        completed.stderr,
        encoding="utf-8",
    )

    if completed.returncode != 0:
        tail = completed.stderr.strip()[-2000:]
        raise RuntimeError(
            "step_image_generator_v2.py failed with "
            f"exit code {completed.returncode}.\n{tail}"
        )

    manifest_path = output_dir / "generation_manifest_v2.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Image generator completed but manifest was not created: "
            f"{manifest_path}"
        )

    return manifest_path


def generate_case(
    *,
    index: int,
    total: int,
    parsed_json_path: Path,
    case_dir: Path,
    overwrite: bool,
    generate_images: bool,
    image_dry_run: bool,
    allow_manual_review: bool,
    image_quality: str,
    image_size: str,
    image_max_tasks: Optional[int],
    image_continue_on_error: bool,
    book_columns: int,
    create_book_without_images: bool,
) -> CaseResult:
    image_stem = image_stem_from_parsed_json(parsed_json_path)

    result = CaseResult(
        index=index,
        image_stem=image_stem,
        parsed_json_path=str(parsed_json_path),
        case_output_dir=str(case_dir),
        status="running",
    )

    case_started = time.perf_counter()
    case_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"[{index}/{total}] {image_stem}")
    print(f"Parsed JSON: {parsed_json_path}")
    print(f"Output:      {case_dir}")
    print("=" * 78)

    valid, validation_message = validate_parsed_json(
        parsed_json_path
    )

    if not valid:
        result.status = "skipped"
        result.stages.append(
            StageResult(
                name="validate_parsed_json",
                status="skipped",
                error_type="InvalidParsedJSON",
                error_message=validation_message,
            )
        )
        result.elapsed_seconds = round(
            time.perf_counter() - case_started,
            3,
        )
        return result

    # ------------------------------------------------------
    # Stage 1: existing Vision JSON -> results.json
    # ------------------------------------------------------
    stage_started = time.perf_counter()

    try:
        results_path = process_one(
            parsed_json_path=parsed_json_path,
            output_dir=case_dir,
            overwrite=overwrite,
        )

        result.stages.append(
            stage_success(
                "pipeline_smoke_test",
                results_path,
                stage_started,
                status=(
                    "existing"
                    if results_path.is_file()
                    and not overwrite
                    else "success"
                ),
            )
        )

    except Exception as exc:
        result.stages.append(
            stage_failure(
                "pipeline_smoke_test",
                exc,
                stage_started,
            )
        )
        result.status = "failed"
        result.elapsed_seconds = round(
            time.perf_counter() - case_started,
            3,
        )
        return result

    # ------------------------------------------------------
    # Stage 2: results.json -> correction_sop.json
    # ------------------------------------------------------
    stage_started = time.perf_counter()
    sop_path = case_dir / "correction_sop.json"

    try:
        if sop_path.is_file() and not overwrite:
            result.stages.append(
                stage_success(
                    "correction_sop_generator",
                    sop_path,
                    stage_started,
                    status="existing",
                )
            )
        else:
            sop_generator = CorrectionSOPGenerator()
            sop = sop_generator.generate_from_results(
                results_path
            )
            sop_path, _ = sop_generator.save(
                sop,
                case_dir,
            )

            result.stages.append(
                stage_success(
                    "correction_sop_generator",
                    sop_path,
                    stage_started,
                )
            )

    except Exception as exc:
        result.stages.append(
            stage_failure(
                "correction_sop_generator",
                exc,
                stage_started,
            )
        )
        result.status = "failed"
        result.elapsed_seconds = round(
            time.perf_counter() - case_started,
            3,
        )
        return result

    # ------------------------------------------------------
    # Stage 3: correction_sop.json -> step_prompts_v2.json
    # ------------------------------------------------------
    stage_started = time.perf_counter()
    prompts_path = case_dir / "step_prompts_v2.json"

    try:
        if prompts_path.is_file() and not overwrite:
            result.stages.append(
                stage_success(
                    "step_prompt_builder_v2",
                    prompts_path,
                    stage_started,
                    status="existing",
                )
            )
        else:
            prompt_builder = StepPromptBuilderV2(
                block_on_manual_review=(
                    not allow_manual_review
                )
            )
            package = prompt_builder.build_from_sop(
                sop_path
            )
            prompts_path, _ = prompt_builder.save(
                package,
                case_dir,
            )

            result.stages.append(
                stage_success(
                    "step_prompt_builder_v2",
                    prompts_path,
                    stage_started,
                )
            )

    except Exception as exc:
        result.stages.append(
            stage_failure(
                "step_prompt_builder_v2",
                exc,
                stage_started,
            )
        )
        result.status = "failed"
        result.elapsed_seconds = round(
            time.perf_counter() - case_started,
            3,
        )
        return result

    # ------------------------------------------------------
    # Stage 4: prompts -> generated step images
    # ------------------------------------------------------
    generated_dir = case_dir / "generated_steps_v2"

    if generate_images or image_dry_run:
        stage_started = time.perf_counter()

        try:
            manifest_path = run_image_generator(
                prompts_json_path=prompts_path,
                output_dir=generated_dir,
                generate_images=generate_images,
                image_dry_run=image_dry_run,
                allow_manual_review=allow_manual_review,
                overwrite=overwrite,
                quality=image_quality,
                size=image_size,
                image_max_tasks=image_max_tasks,
                image_continue_on_error=image_continue_on_error,
            )

            result.stages.append(
                stage_success(
                    "step_image_generator_v2",
                    manifest_path,
                    stage_started,
                    status=(
                        "dry_run"
                        if image_dry_run
                        else "success"
                    ),
                )
            )

        except Exception as exc:
            result.stages.append(
                stage_failure(
                    "step_image_generator_v2",
                    exc,
                    stage_started,
                )
            )
            result.status = "failed"
            result.elapsed_seconds = round(
                time.perf_counter() - case_started,
                3,
            )
            return result

    else:
        result.stages.append(
            StageResult(
                name="step_image_generator_v2",
                status="skipped",
                output_path=str(generated_dir),
                error_message=(
                    "Image API stage disabled. "
                    "Use --generate-images or --image-dry-run."
                ),
            )
        )

    # ------------------------------------------------------
    # Stage 5: one-page instruction image
    # ------------------------------------------------------
    should_create_book = (
        generate_images
        or create_book_without_images
        or (
            generated_dir
            / "generation_manifest_v2.json"
        ).is_file()
    )

    if should_create_book:
        stage_started = time.perf_counter()
        book_path = (
            case_dir
            / "assembly_instruction_book.png"
        )

        try:
            if book_path.is_file() and not overwrite:
                result.stages.append(
                    stage_success(
                        "instruction_book_generator",
                        book_path,
                        stage_started,
                        status="existing",
                    )
                )
            else:
                book_generator = InstructionBookGenerator(
                    columns=book_columns,
                )

                book_path = book_generator.generate(
                    prompts_json_path=prompts_path,
                    output_path=book_path,
                    overwrite=overwrite,
                )

                result.stages.append(
                    stage_success(
                        "instruction_book_generator",
                        book_path,
                        stage_started,
                    )
                )

            result.final_instruction_path = str(
                book_path
            )

        except Exception as exc:
            result.stages.append(
                stage_failure(
                    "instruction_book_generator",
                    exc,
                    stage_started,
                )
            )
            result.status = "failed"
            result.elapsed_seconds = round(
                time.perf_counter() - case_started,
                3,
            )
            return result

    else:
        result.stages.append(
            StageResult(
                name="instruction_book_generator",
                status="skipped",
                error_message=(
                    "No generated images. Use "
                    "--create-book-with-placeholders to render placeholders."
                ),
            )
        )

    result.status = "success"
    result.elapsed_seconds = round(
        time.perf_counter() - case_started,
        3,
    )

    return result


def save_batch_summary(
    *,
    summary: BatchSummary,
    batch_dir: Path,
) -> tuple[Path, Path]:
    json_path = batch_dir / "batch_summary.json"
    csv_path = batch_dir / "batch_summary.csv"

    save_json(
        json_path,
        summary.to_dict(),
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "image_stem",
                "status",
                "parsed_json_path",
                "case_output_dir",
                "final_instruction_path",
                "elapsed_seconds",
                "failed_stage",
                "error_type",
                "error_message",
            ],
        )
        writer.writeheader()

        for case in summary.cases:
            failed_stage = next(
                (
                    stage
                    for stage in case.stages
                    if stage.status == "failed"
                ),
                None,
            )

            writer.writerow(
                {
                    "index": case.index,
                    "image_stem": case.image_stem,
                    "status": case.status,
                    "parsed_json_path": case.parsed_json_path,
                    "case_output_dir": case.case_output_dir,
                    "final_instruction_path": (
                        case.final_instruction_path
                        or ""
                    ),
                    "elapsed_seconds": case.elapsed_seconds,
                    "failed_stage": (
                        failed_stage.name
                        if failed_stage
                        else ""
                    ),
                    "error_type": (
                        failed_stage.error_type
                        if failed_stage
                        else ""
                    ),
                    "error_message": (
                        failed_stage.error_message
                        if failed_stage
                        else ""
                    ),
                }
            )

    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-process all existing Vision parsed JSON files "
            "into independent assembly-instruction outputs."
        )
    )

    parser.add_argument(
        "--parsed-json-dir",
        type=Path,
        default=PARSED_JSON_DIR,
        help=(
            "Directory containing *_parsed_*.json files."
        ),
    )

    parser.add_argument(
        "--batch-root",
        type=Path,
        default=BATCH_RUNS_ROOT,
    )

    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help=(
            "Batch folder name. Default: current timestamp."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Allow reuse of an existing batch directory."
        ),
    )

    parser.add_argument(
        "--all-json",
        action="store_true",
        help=(
            "Process every parsed JSON version. "
            "Default: latest parsed JSON per original image."
        ),
    )

    parser.add_argument(
        "--contains",
        type=str,
        default=None,
        help=(
            "Only process image stems containing this text."
        ),
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N selected cases."
        ),
    )

    parser.add_argument(
        "--generate-images",
        action="store_true",
        help=(
            "Actually call gpt-image-2. "
            "Without this flag, no Image API call is made."
        ),
    )

    parser.add_argument(
        "--image-dry-run",
        action="store_true",
        help=(
            "Run step_image_generator_v2.py in dry-run mode."
        ),
    )

    parser.add_argument(
        "--allow-manual-review",
        action="store_true",
        help=(
            "Override generation_allowed=false after manual confirmation."
        ),
    )

    parser.add_argument(
        "--image-quality",
        choices=[
            "low",
            "medium",
            "high",
            "auto",
        ],
        default="medium",
    )

    parser.add_argument(
        "--image-size",
        choices=[
            "1024x1024",
            "1024x1536",
            "1536x1024",
            "auto",
        ],
        default="1024x1024",
    )

    parser.add_argument(
        "--image-max-tasks",
        type=int,
        default=None,
        help=(
            "Limit the number of image tasks per case."
        ),
    )

    parser.add_argument(
        "--image-continue-on-error",
        action="store_true",
    )

    parser.add_argument(
        "--book-columns",
        type=int,
        choices=[1, 2],
        default=1,
    )

    parser.add_argument(
        "--create-book-with-placeholders",
        action="store_true",
        help=(
            "Create instruction PNG even when no generated images exist."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate existing files inside a resumed batch."
        ),
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help=(
            "Stop the whole batch when one case fails."
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.generate_images and args.image_dry_run:
        raise ValueError(
            "--generate-images and --image-dry-run cannot be used together."
        )

    if args.generate_images:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            # step_image_generator_v2.py itself also loads .env.
            # This warning does not stop here, because subprocess may load it.
            print(
                "[WARNING] OPENAI_API_KEY is not currently in the "
                "parent process environment. The image generator will "
                "still attempt to load the project-root .env.",
                file=sys.stderr,
            )

    all_files = find_all_parsed_files(
        args.parsed_json_dir
    )

    selected = (
        all_files
        if args.all_json
        else select_latest_per_image(
            all_files
        )
    )

    selected = filter_parsed_files(
        selected,
        contains=args.contains,
        limit=args.limit,
        offset=args.offset,
    )

    if not selected:
        raise FileNotFoundError(
            "No parsed JSON files were selected."
        )

    batch_id, batch_dir = prepare_batch_directory(
        batch_root=args.batch_root,
        batch_id=args.batch_id,
        resume=args.resume,
    )

    summary = BatchSummary(
        schema_version="1.0",
        batch_id=batch_id,
        created_at=datetime.now().astimezone().isoformat(),
        finished_at=None,
        parsed_json_dir=str(
            args.parsed_json_dir.expanduser().resolve()
        ),
        batch_output_dir=str(batch_dir),
        latest_per_image=not args.all_json,
        generate_images=args.generate_images,
        image_dry_run=args.image_dry_run,
        allow_manual_review=args.allow_manual_review,
        overwrite=args.overwrite,
        discovered_json_count=len(all_files),
        selected_case_count=len(selected),
        successful_case_count=0,
        failed_case_count=0,
        skipped_case_count=0,
        elapsed_seconds=0.0,
        cases=[],
        warnings=[],
    )

    if args.generate_images:
        summary.warnings.append(
            "Image generation is enabled and may incur API cost."
        )

    if args.allow_manual_review:
        summary.warnings.append(
            "Manual-review protection is overridden for this batch."
        )

    batch_started = time.perf_counter()

    print("=" * 78)
    print("BATCH ASSEMBLY INSTRUCTION PIPELINE")
    print("=" * 78)
    print(f"Parsed JSON discovered: {len(all_files)}")
    print(
        "Selection mode:        "
        + (
            "all JSON versions"
            if args.all_json
            else "latest JSON per original image"
        )
    )
    print(f"Cases selected:       {len(selected)}")
    print(f"Batch output:         {batch_dir}")
    print(f"Generate images:      {args.generate_images}")
    print(f"Image dry run:        {args.image_dry_run}")
    print("=" * 78)

    for index, parsed_path in enumerate(
        selected,
        start=1,
    ):
        image_stem = image_stem_from_parsed_json(
            parsed_path
        )

        # --all-json 時加入完整 parsed stem，防止同一圖片不同版本撞名。
        case_folder_name = (
            safe_folder_name(parsed_path.stem)
            if args.all_json
            else safe_folder_name(image_stem)
        )

        case_dir = batch_dir / case_folder_name

        case_result = generate_case(
            index=index,
            total=len(selected),
            parsed_json_path=parsed_path,
            case_dir=case_dir,
            overwrite=args.overwrite,
            generate_images=args.generate_images,
            image_dry_run=args.image_dry_run,
            allow_manual_review=args.allow_manual_review,
            image_quality=args.image_quality,
            image_size=args.image_size,
            image_max_tasks=args.image_max_tasks,
            image_continue_on_error=args.image_continue_on_error,
            book_columns=args.book_columns,
            create_book_without_images=(
                args.create_book_with_placeholders
            ),
        )

        summary.cases.append(
            case_result
        )

        if case_result.status == "success":
            summary.successful_case_count += 1
            print(
                f"[SUCCESS] {case_result.image_stem} "
                f"({case_result.elapsed_seconds:.1f}s)"
            )

        elif case_result.status == "skipped":
            summary.skipped_case_count += 1
            print(
                f"[SKIPPED] {case_result.image_stem}"
            )

        else:
            summary.failed_case_count += 1
            failed_stage = next(
                (
                    stage
                    for stage in case_result.stages
                    if stage.status == "failed"
                ),
                None,
            )
            print(
                f"[FAILED] {case_result.image_stem}: "
                f"{failed_stage.error_message if failed_stage else 'unknown error'}",
                file=sys.stderr,
            )

            if args.stop_on_error:
                break

        # 每完成一份就更新 summary，避免中途停止後紀錄全失。
        summary.elapsed_seconds = round(
            time.perf_counter() - batch_started,
            3,
        )
        save_batch_summary(
            summary=summary,
            batch_dir=batch_dir,
        )

    summary.finished_at = (
        datetime.now()
        .astimezone()
        .isoformat()
    )
    summary.elapsed_seconds = round(
        time.perf_counter() - batch_started,
        3,
    )

    summary_json, summary_csv = save_batch_summary(
        summary=summary,
        batch_dir=batch_dir,
    )

    print("=" * 78)
    print("BATCH PIPELINE FINISHED")
    print("=" * 78)
    print(f"Selected:   {summary.selected_case_count}")
    print(f"Successful: {summary.successful_case_count}")
    print(f"Failed:     {summary.failed_case_count}")
    print(f"Skipped:    {summary.skipped_case_count}")
    print(f"Elapsed:    {summary.elapsed_seconds:.1f}s")
    print(f"Summary:    {summary_json}")
    print(f"CSV:        {summary_csv}")
    print("=" * 78)

    return (
        0
        if summary.failed_case_count == 0
        else 1
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise