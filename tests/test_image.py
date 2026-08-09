"""Manual Azure image-generation smoke script (never runs during pytest)."""

import argparse
import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize one paid image call")
    args = parser.parse_args()
    if not args.execute:
        print("Dry run only. Re-run with --execute after provider and budget approval; planned calls: 1")
        return 0
    root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=root / ".env")
    endpoint, api_key, deployment = (
        os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("IMAGE_DEPLOYMENT")
    )
    if not all((endpoint, api_key, deployment)):
        raise EnvironmentError("Azure endpoint, API key, and image deployment are required")
    client = AzureOpenAI(api_version="2025-04-01-preview", azure_endpoint=endpoint, api_key=api_key)
    result = client.images.generate(model=deployment, prompt="One clean construction instruction card", size="1024x1024")
    output = root / "output" / "manual_image_smoke.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(base64.b64decode(result.data[0].b64_json))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
