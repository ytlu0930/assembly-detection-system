"""Batch orchestration that delegates every case to ``main.run_pipeline``."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from main import PipelineManifest, run_pipeline
from pipeline_smoke_test import PARSED_JSON_DIR, image_stem_from_parsed_json, parsed_timestamp_key


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH_ROOT = PROJECT_ROOT / "output" / "batch_runs"


@dataclass
class CaseResult:
    parsed_json_path: str
    image_stem: str
    output_dir: str
    status: str
    final_instruction_path: str | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class BatchSummary:
    created_at: str
    output_dir: str
    requested_count: int
    completed_count: int
    failed_count: int
    image_provider: str
    execute_image_api: bool
    cases: list[CaseResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_all_parsed_files(parsed_dir: Path) -> list[Path]:
    return sorted(parsed_dir.expanduser().resolve().glob("*_parsed_*.json"))


def select_latest_per_image(parsed_files: list[Path]) -> list[Path]:
    latest: dict[str, Path] = {}
    for path in parsed_files:
        stem = image_stem_from_parsed_json(path)
        if stem not in latest or parsed_timestamp_key(path) > parsed_timestamp_key(latest[stem]):
            latest[stem] = path
    return sorted(latest.values(), key=lambda path: image_stem_from_parsed_json(path))


def run_batch(
    *,
    parsed_dir: str | Path = PARSED_JSON_DIR,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    generate_images: bool = False,
    image_provider: str = "mock",
    execute_image_api: bool = False,
    confirm_cost: bool = False,
    max_image_requests: int = 0,
    allow_manual_review: bool = False,
    overwrite: bool = False,
    pipeline_runner: Callable[..., PipelineManifest] = run_pipeline,
) -> BatchSummary:
    selected = select_latest_per_image(find_all_parsed_files(Path(parsed_dir)))
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    run_root = Path(output_dir).expanduser().resolve() if output_dir else DEFAULT_BATCH_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    live = bool(image_provider == "openai" and generate_images and execute_image_api and confirm_cost and max_image_requests > 0)
    cases: list[CaseResult] = []
    remaining = max(0, int(max_image_requests))
    for index, parsed in enumerate(selected, start=1):
        stem = image_stem_from_parsed_json(parsed)
        case_dir = run_root / f"{index:03d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', stem)}"
        budget = 1 if live and remaining > 0 else 0
        manifest = pipeline_runner(
            parsed_json_path=parsed, output_dir=case_dir, generate_images=generate_images,
            image_provider=image_provider, execute_image_api=live and budget > 0,
            confirm_cost=confirm_cost, image_max_requests=budget,
            allow_manual_review=allow_manual_review, overwrite=overwrite,
        )
        if manifest.execute_image_api:
            remaining -= budget
        cases.append(CaseResult(
            parsed_json_path=str(parsed), image_stem=stem, output_dir=str(case_dir),
            status=manifest.status, final_instruction_path=manifest.final_instruction_path,
            warnings=list(manifest.warnings), errors=list(manifest.errors),
        ))
    summary = BatchSummary(
        created_at=datetime.now().astimezone().isoformat(), output_dir=str(run_root),
        requested_count=len(selected), completed_count=sum(item.status in {"success", "partial"} for item in cases),
        failed_count=sum(item.status == "failed" for item in cases), image_provider=image_provider,
        execute_image_api=live, cases=cases,
    )
    (run_root / "batch_summary.json").write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical pipeline for the newest parsed JSON per image.")
    parser.add_argument("--parsed-dir", type=Path, default=PARSED_JSON_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generate-images", action="store_true")
    parser.add_argument("--image-provider", choices=["mock", "openai"], default="mock")
    parser.add_argument("--execute-image-api", action="store_true")
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--max-image-requests", type=int, default=0)
    parser.add_argument("--allow-manual-review", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_batch(
        parsed_dir=args.parsed_dir, output_dir=args.output_dir, limit=args.limit,
        generate_images=args.generate_images, image_provider=args.image_provider,
        execute_image_api=args.execute_image_api, confirm_cost=args.confirm_cost,
        max_image_requests=args.max_image_requests, allow_manual_review=args.allow_manual_review,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0 if summary.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
