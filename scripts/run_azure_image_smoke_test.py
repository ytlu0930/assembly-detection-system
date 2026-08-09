"""Dry-run-first, one-request Azure GPT Image 2 smoke harness."""

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

from utils.azure_openai_image_provider import (
    AzureOpenAIImageProvider,
    build_azure_image_edit_endpoint,
)
from utils.output_manager import resolve_run_output, write_run_summary


CASES = {
    "missingpart-A01": ROOT / "input/missingpart/model03_step03/model03_step03_missingpart-A01_front_01.jpg",
}
REFERENCE = ROOT / "input/normal/model03_step03/model03_step03_correct-01_front_01.jpg"
TEACHER_ENDPOINT = "https://undergraduateproject2eastus2.cognitiveservices.azure.com/"


def _true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() == "true"


def _prompt() -> str:
    return (
        "Create one instructional correction illustration. Image 1 is the current assembly state. "
        "The correct target state requires one short red pin inserted at the reference-indicated front connector. "
        "Current action: insert only the missing short red pin. Expected visual state: the pin is aligned and seated "
        "in that connector. Modify only the target location; preserve every non-target brick, camera angle, lighting, "
        "background, brick colors, geometry, shapes, and part counts. Show one clear insertion direction. "
        "Do not add unrelated parts, people, hands, text, or a collage. Output only the next correction state."
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--case", choices=sorted(CASES), default="missingpart-A01")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--execute-api", action="store_true")
    result.add_argument("--confirm-cost", action="store_true")
    result.add_argument("--output-dir", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)

    # Dry-run 也讀取設定，但不代表允許 API execution
    load_dotenv(ROOT / ".env")

    paths = resolve_run_output(
        "pipeline",
        "azure_step_image_smoke",
        output_dir=args.output_dir,
        output_root=ROOT / "output",
    )

    prompts_dir = paths.run_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Build and save the smoke-test prompt
    prompt = _prompt()
    prompt_path = prompts_dir / "step_01.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    endpoint_base = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    deployment = os.getenv("AZURE_IMAGE_DEPLOYMENT", "gpt-image-2").strip()
    api_version = os.getenv(
        "AZURE_IMAGE_API_VERSION",
        "2024-02-01",
    ).strip()
    auth_mode = os.getenv(
        "AZURE_IMAGE_AUTH_MODE",
        "bearer",
    ).strip()

    try:
        endpoint = build_azure_image_edit_endpoint(
            endpoint_base or TEACHER_ENDPOINT,
            deployment,
            api_version,
        )
    except ValueError:
        endpoint = "<not configured>"

    configured = bool(
        os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    )
    env_enabled = _true("ENABLE_OPENAI_IMAGE_API")
    env_confirmed = _true(
        "CONFIRM_OPENAI_IMAGE_API_EXECUTION"
    )

    execute = bool(
        args.execute_api
        and not args.dry_run
        and args.confirm_cost
        and env_enabled
        and env_confirmed
        and configured
        and bool(endpoint_base)
    )

    output_path = paths.images_dir / "step_01.png"
    plan: dict[str, object] = {
        "case": args.case,
        "mode": "execute" if execute else "dry-run",
        "provider": "azure_openai",
        "endpoint": endpoint,
        "endpoint_configured": bool(endpoint_base),
        "deployment": deployment,
        "api_version": api_version,
        "auth_mode": auth_mode,
        "source_image": str(CASES[args.case]),
        "reference_image": str(REFERENCE),
        "reference_binary_supported": False,
        "prompt": prompt,
        "prompt_path": str(prompt_path),
        "quality": os.getenv("OPENAI_IMAGE_QUALITY", "low"),
        "size": os.getenv("OPENAI_IMAGE_SIZE", "1536x1024"),
        "output_format": os.getenv("OPENAI_IMAGE_OUTPUT_FORMAT", "png"),
        "output_path": str(output_path),
        "estimated_requests": 1,
        "api_key_configured": configured,
        "environment_enabled": env_enabled,
        "environment_confirmed": env_confirmed,
        "execute_api": args.execute_api,
        "confirm_cost": args.confirm_cost,
    }
    (paths.run_dir / "inputs_manifest.json").write_text(
        json.dumps({"source_image": plan["source_image"], "reference_image": plan["reference_image"], "reference_binary_supported": False}, indent=2),
        encoding="utf-8",
    )

    if args.execute_api and not args.dry_run and not execute:
        plan["status"] = "disabled"
        plan["error"] = "Execution requires --confirm-cost, Azure configuration, both environment gates, and a configured API key."
        exit_code = 2
    elif execute:
        provider = AzureOpenAIImageProvider(
            endpoint=endpoint_base, deployment=deployment, api_version=api_version,
            auth_mode=auth_mode, quality=str(plan["quality"]), size=str(plan["size"]),
            output_format=str(plan["output_format"]), max_requests_per_run=1, enabled=True,
        )
        result = provider.generate_step_image(
            str(CASES[args.case]), str(REFERENCE), prompt, str(output_path),
            {"case": args.case}, execute_api=True,
        )
        plan["result"] = result.to_dict()
        plan["status"] = result.status
        exit_code = 0 if result.success else 1
    else:
        plan["status"] = "dry-run"
        exit_code = 0

    paths.json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_summary(
        paths, status="completed" if exit_code == 0 else "failed",
        input_count=1, success_count=1 if exit_code == 0 else 0,
        failure_count=0 if exit_code == 0 else 1,
        parameters={"case": args.case, "provider": "azure_openai", "mode": plan["mode"], "deployment": deployment, "api_version": api_version, "estimated_requests": 1},
        notes=["Dry-run makes no Azure request."],
        output_paths={"run_dir": str(paths.run_dir), "results": str(paths.json_path)},
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
