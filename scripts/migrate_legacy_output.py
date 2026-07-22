"""Preview or apply migration of known legacy output directories."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.output_manager import create_run_output, write_run_summary


LEGACY_MAPPINGS = {
    "grounding": ("localization", "phase07_grounding_poc"),
    "grounding_experiments": ("localization", "phase07_grounding_experiments"),
    "bbox_selection_experiments": ("localization", "phase08_bbox_selection"),
    "localization_pipeline": ("pipeline", "localization_pipeline"),
    "annotated": ("vision", "annotations"),
}


def directory_stats(path: Path) -> dict[str, int]:
    """Return recursive file count and byte size for a directory."""
    files = [item for item in path.rglob("*") if item.is_file()] if path.exists() else []
    return {
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
    }


def plan_migration(
    output_root: str | Path,
    *,
    apply: bool = False,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Create a manifest and optionally move non-conflicting legacy folders."""
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_paths = create_run_output(
        "pipeline", "legacy_output_migration", output_root=root, image_subdirs=[]
    )
    entries: list[dict[str, Any]] = []

    for legacy_name, (category, experiment) in LEGACY_MAPPINGS.items():
        source = root / legacy_name
        destination = root / category / experiment / f"legacy_{stamp}"
        before = directory_stats(source)
        entry: dict[str, Any] = {
            "source": str(source),
            "destination": str(destination),
            "mode": "apply" if apply else "dry-run",
            "status": "missing" if not source.exists() else "planned",
            "before": before,
            "after": None,
            "conflict": False,
            "error": None,
        }

        if source.exists() and destination.exists():
            entry.update(status="conflict", conflict=True)
        elif apply and source.exists():
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                entry["after"] = directory_stats(destination)
                entry["status"] = "moved"
                if entry["after"] != before:
                    entry["status"] = "verification_failed"
            except Exception as exc:  # Record and stop only this folder.
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
        entries.append(entry)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "apply" if apply else "dry-run",
        "output_root": str(root),
        "legacy_timestamp": stamp,
        "entries": entries,
        "totals": {
            "folder_count": len(entries),
            "existing_folder_count": sum(item["status"] != "missing" for item in entries),
            "file_count": sum(item["before"]["file_count"] for item in entries),
            "total_bytes": sum(item["before"]["total_bytes"] for item in entries),
            "moved_folder_count": sum(item["status"] == "moved" for item in entries),
            "conflict_count": sum(item["conflict"] for item in entries),
        },
    }
    manifest_path = manifest_paths.run_dir / "migration_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    failed = sum(item["status"] in {"failed", "verification_failed"} for item in entries)
    write_run_summary(
        manifest_paths,
        status="failed" if failed else "completed",
        input_count=len(entries),
        success_count=len(entries) - failed,
        failure_count=failed,
        parameters={"apply": apply, "output_root": str(root)},
        output_paths={"migration_manifest": str(manifest_path)},
        notes=[
            "Dry-run does not move legacy folders."
            if not apply
            else "Only non-conflicting legacy folders were moved."
        ],
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply planned moves.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = plan_migration(args.output_root, apply=args.apply)
    for item in manifest["entries"]:
        print(
            f"[{item['status']}] {item['source']} -> {item['destination']} "
            f"files={item['before']['file_count']} bytes={item['before']['total_bytes']}"
        )
    print(f"migration_manifest: {manifest['manifest_path']}")
    return 0 if not any(
        item["status"] in {"failed", "verification_failed"}
        for item in manifest["entries"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
