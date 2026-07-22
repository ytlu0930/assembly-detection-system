"""Standardized, collision-safe output directory management."""

from __future__ import annotations

import json
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable


RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}(?:_\d{2})?$")
_ILLEGAL_WINDOWS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class RunOutputPaths:
    """Paths belonging to one output run."""

    output_root: Path
    category_dir: Path
    experiment_dir: Path
    run_dir: Path
    images_dir: Path
    json_path: Path
    csv_path: Path
    summary_path: Path
    run_id: str
    category: str
    experiment: str
    image_subdirs: dict[str, Path]


def generate_run_id(now: datetime | None = None) -> str:
    """Return a timestamp run identifier in ``YYYYMMDD_HHMMSS`` format."""
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def sanitize_component(value: str, field_name: str) -> str:
    """Clean a Windows-compatible path component while blocking traversal."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")

    raw = value.strip()
    windows_path = PureWindowsPath(raw)
    if windows_path.is_absolute() or ".." in windows_path.parts:
        raise ValueError(f"{field_name} must not escape the output root")

    cleaned = _ILLEGAL_WINDOWS_CHARS.sub("_", raw)
    cleaned = re.sub(r"_+", "_", cleaned).strip(" ._")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"{field_name} is invalid after sanitization")
    return cleaned


def _safe_image_subdirs(values: Iterable[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        cleaned = sanitize_component(value, "image subdirectory")
        if cleaned not in names:
            names.append(cleaned)
    return names


def _build_paths(
    output_root: Path,
    category: str,
    experiment: str,
    run_dir: Path,
    run_id: str,
    image_subdirs: Iterable[str] | None,
    nest_experiment: bool = True,
) -> RunOutputPaths:
    images_dir = run_dir / "images"
    subdir_names = _safe_image_subdirs(image_subdirs)
    if image_subdirs is None or subdir_names:
        images_dir.mkdir(parents=True, exist_ok=True)
    subdirs = {name: images_dir / name for name in subdir_names}
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    category_dir = output_root / category
    experiment_dir = (
        category_dir / experiment if nest_experiment else category_dir
    )
    return RunOutputPaths(
        output_root=output_root,
        category_dir=category_dir,
        experiment_dir=experiment_dir,
        run_dir=run_dir,
        images_dir=images_dir,
        json_path=run_dir / "results.json",
        csv_path=run_dir / "results.csv",
        summary_path=run_dir / "run_summary.json",
        run_id=run_id,
        category=category,
        experiment=experiment,
        image_subdirs=subdirs,
    )


def create_run_output(
    category: str,
    experiment: str,
    run_id: str | None = None,
    output_root: str | Path = "output",
    image_subdirs: list[str] | None = None,
    nest_experiment: bool = True,
) -> RunOutputPaths:
    """Create a standard run directory without overwriting an existing run."""
    safe_category = sanitize_component(category, "category")
    safe_experiment = sanitize_component(experiment, "experiment")
    root = Path(output_root).expanduser().resolve()
    experiment_dir = (
        root / safe_category / safe_experiment
        if nest_experiment
        else root / safe_category
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    base_run_id = sanitize_component(run_id or generate_run_id(), "run_id")
    if not RUN_ID_PATTERN.fullmatch(base_run_id):
        raise ValueError("run_id must use YYYYMMDD_HHMMSS with an optional _NN suffix")

    candidate_id = base_run_id
    suffix = 0
    while True:
        candidate = experiment_dir / candidate_id
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
            candidate_id = f"{base_run_id}_{suffix:02d}"

    return _build_paths(
        root,
        safe_category,
        safe_experiment,
        candidate,
        candidate_id,
        image_subdirs,
        nest_experiment,
    )


def use_explicit_output_dir(
    output_dir: str | Path,
    category: str,
    experiment: str,
    image_subdirs: list[str] | None = None,
) -> RunOutputPaths:
    """Use an exact caller-supplied output directory for CLI compatibility."""
    safe_category = sanitize_component(category, "category")
    safe_experiment = sanitize_component(experiment, "experiment")
    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name or generate_run_id()
    return _build_paths(
        run_dir.parent,
        safe_category,
        safe_experiment,
        run_dir,
        run_id,
        image_subdirs,
    )


def resolve_run_output(
    category: str,
    experiment: str,
    output_dir: str | Path | None = None,
    output_root: str | Path = "output",
    image_subdirs: list[str] | None = None,
    nest_experiment: bool = True,
) -> RunOutputPaths:
    """Prefer an explicit output directory, otherwise create a standard run."""
    if output_dir is not None:
        return use_explicit_output_dir(
            output_dir, category, experiment, image_subdirs=image_subdirs
        )
    return create_run_output(
        category,
        experiment,
        output_root=output_root,
        image_subdirs=image_subdirs,
        nest_experiment=nest_experiment,
    )


def write_run_summary(
    paths: RunOutputPaths,
    *,
    status: str,
    input_count: int,
    success_count: int,
    failure_count: int,
    parameters: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    output_paths: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> Path:
    """Write the additional per-run metadata summary."""
    if status not in {"completed", "partial", "failed"}:
        raise ValueError("status must be completed, partial, or failed")
    payload: dict[str, Any] = {
        "run_id": paths.run_id,
        "category": paths.category,
        "experiment": paths.experiment,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "status": status,
        "input_count": input_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "output_paths": output_paths or {},
        "parameters": parameters or {},
        "timing": timing or {},
        "notes": notes or [],
    }
    if runtime:
        payload.update(runtime)
    with paths.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return paths.summary_path
