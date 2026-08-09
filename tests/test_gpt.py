"""Manual Azure Vision connectivity smoke script (never runs during pytest)."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize one paid API call")
    args = parser.parse_args()
    if not args.execute:
        print("Dry run only. Re-run with --execute after API budget approval; planned calls: 1")
        return 0
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)
    endpoint, api_key, deployment = (
        os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("GPT4O_DEPLOYMENT")
    )
    if not all((endpoint, api_key, deployment)):
        raise EnvironmentError("Azure endpoint, API key, and GPT4O deployment are required")
    client = AzureOpenAI(api_version="2024-12-01-preview", azure_endpoint=endpoint, api_key=api_key)
    response = client.chat.completions.create(model=deployment, messages=[{"role": "user", "content": "Return OK."}])
    print(response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
