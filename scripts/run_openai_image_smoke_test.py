"""Dry-run-first smoke harness for the guarded GPT Image 2 edit adapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from utils.openai_image_provider import OpenAIImageProvider
from utils.output_manager import resolve_run_output, write_run_summary
from utils.step_prompt_builder import build_step_prompts


CASES = {
    "missingpart-A01": ROOT / "input/missingpart/model03_step03/model03_step03_missingpart-A01_front_01.jpg",
}
REFERENCE = ROOT / "input/normal/model03_step03/model03_step03_correct-01_front_01.jpg"


def _true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() == "true"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--case", choices=sorted(CASES), default="missingpart-A01")
    result.add_argument("--dry-run", action="store_true", help="Plan only; this is also the default")
    result.add_argument("--execute-api", action="store_true")
    result.add_argument("--confirm-cost", action="store_true")
    result.add_argument("--output-dir", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.execute_api and not args.dry_run:
        load_dotenv(ROOT / ".env", override=True)
    paths = resolve_run_output(
        "pipeline",
        "openai_step_image_smoke",
        output_dir=args.output_dir,
        output_root=ROOT / "output",
    )
    prompts_dir = paths.run_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    source = CASES[args.case]
    sop = {"steps": [{
        "step_number": 1,
        "action": "insert",
        "instruction": "Insert the missing red short pin in the reference-matched location.",
        "visual_instruction": "Insert only the missing red short pin and show the placement direction.",
        "affected_parts": ["PIN_RED_SHORT"],
    }]}
    task = build_step_prompts(
        sop,
        test_image_path=str(source),
        reference_image_path=str(REFERENCE),
        model_id="model03",
        step_id="step03",
        view_angle="front",
    )[0]
    prompt_path = prompts_dir / "step_01.txt"
    prompt_path.write_text(task["prompt"], encoding="utf-8")
    output_path = paths.images_dir / "step_01.png"
    configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
    env_enabled = _true("ENABLE_OPENAI_IMAGE_API")
    env_confirmed = _true("CONFIRM_OPENAI_IMAGE_API_EXECUTION")
    execute = bool(
        args.execute_api and not args.dry_run and args.confirm_cost
        and env_enabled and env_confirmed and configured
    )
    plan = {
        "case": args.case,
        "mode": "execute" if execute else "dry-run",
        "source_image": str(source),
        "reference_image": str(REFERENCE),
        "prompt_path": str(prompt_path),
        "prompt": task["prompt"],
        "model": os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
        "quality": os.getenv("OPENAI_IMAGE_QUALITY", "low"),
        "size": os.getenv("OPENAI_IMAGE_SIZE", "1536x1024"),
        "output_path": str(output_path),
        "estimated_requests": 1,
        "flags": {
            "execute_api": args.execute_api,
            "confirm_cost": args.confirm_cost,
            "environment_enabled": env_enabled,
            "environment_confirmed": env_confirmed,
            "api_key_configured": configured,
        },
    }
    manifest = {"source_image": str(source), "reference_image": str(REFERENCE), "case": args.case}
    (paths.run_dir / "inputs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.execute_api and not args.dry_run and not execute:
        plan["status"] = "disabled"
        plan["error"] = "Execution requires --confirm-cost, both environment authorization flags, and a configured API key."
        exit_code = 2
    elif execute:
        provider = OpenAIImageProvider(
            model=plan["model"], quality=plan["quality"], size=plan["size"],
            output_format=os.getenv("OPENAI_IMAGE_OUTPUT_FORMAT", "png"),
            timeout_seconds=float(os.getenv("OPENAI_IMAGE_TIMEOUT_SECONDS", "120")),
            max_retries=int(os.getenv("OPENAI_IMAGE_MAX_RETRIES", "2")),
            max_requests_per_run=1, enabled=True,
        )
        result = provider.generate_step_image(
            str(source), str(REFERENCE), task["prompt"], str(output_path), task, execute_api=True,
        )
        plan["result"] = result.to_dict()
        plan["status"] = result.status
        exit_code = 0 if result.success else 1
    else:
        plan["status"] = "dry-run"
        exit_code = 0

    paths.json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_summary(
        paths,
        status="completed" if exit_code == 0 else "failed",
        input_count=1,
        success_count=1 if plan["status"] in {"success", "dry-run"} else 0,
        failure_count=0 if exit_code == 0 else 1,
        parameters={key: plan[key] for key in ("case", "mode", "model", "quality", "size", "estimated_requests")},
        notes=["No API request is made in dry-run mode."],
        output_paths={"run_dir": str(paths.run_dir), "results": str(paths.json_path)},
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
