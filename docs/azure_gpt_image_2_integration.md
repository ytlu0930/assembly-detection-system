# Azure-hosted GPT Image 2 Integration

## Status

- Formal deployment provider: Azure-hosted OpenAI GPT Image 2
- Azure resource endpoint: configured locally through `AZURE_OPENAI_ENDPOINT`
- Deployment/model: `gpt-image-2` / `gpt-image-2`
- Teacher deployment version: `2026-04-21`
- Image API version: `2024-02-01` by default
- Operation: Image Editing
- Adapter implementation: complete
- API configuration: pending user verification
- One-request smoke: pending
- Image quality validation: pending
- Safe runtime default: Mock

Codex did not execute a real Azure request while implementing or testing this adapter.

## Endpoint and authentication

The provider builds:

```text
{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_IMAGE_DEPLOYMENT}/images/edits?api-version={AZURE_IMAGE_API_VERSION}
```

The base endpoint must be HTTPS and contain no query, fragment, credentials, or extra path. Deployment and API version are URL encoded. `AZURE_IMAGE_AUTH_MODE=bearer` sends only `Authorization: Bearer <key>`; `api_key` sends only `api-key: <key>`. Headers and credentials are never stored in result metadata.

## First-version request contract

The request uses `httpx` and `multipart/form-data`:

- required binary `image`: current/source assembly state
- required text `prompt`: Step Prompt Builder V2 edit prompt
- optional binary `mask`

Quality, size, and output format are recorded as requested settings but are not sent as multipart fields until Azure deployment support is verified. The correct reference remains in metadata and prompt context. `supports_multi_image_reference` is `false`; no second image binary is sent.

Step 1 uses the test/current image. A later step uses the previous successful generated image. A failed or disabled step stops later image generation while SOP, prompts, instruction-book fallback, manifest, and UI contract remain available.

## Safety

Network execution requires all of:

```text
ENABLE_OPENAI_IMAGE_API=true
CONFIRM_OPENAI_IMAGE_API_EXECUTION=true
execute_api=True
```

The smoke CLI additionally requires `--confirm-cost`. The Azure key preflight rejects empty/whitespace values, leading `=`, newlines, placeholders, URLs, Markdown links, and accidentally pasted `KEY=value` lines. It reports only:

```text
AZURE_OPENAI_API_KEY appears malformed. Check the local .env file.
```

## Retry and validation

Timeout, connection failures, HTTP 408/429, and HTTP 5xx may retry using 2- and 4-second backoff, subject to the request budget. HTTP 400/401/403/404, malformed configuration, malformed key, and invalid inputs do not retry. Default timeout is 120 seconds, maximum retries is two, and request budget is one.

Successful responses must contain `data[0].b64_json`. The decoded bytes and written file must be non-empty, Pillow-decodable, have positive dimensions, and use a supported image format.

## Dry-run and user smoke

Safe dry-run:

```powershell
.\venv\Scripts\python.exe scripts\run_azure_image_smoke_test.py --dry-run
```

After reviewing `.env` and separately approving one paid request, the user may run:

```powershell
.\venv\Scripts\python.exe scripts\run_azure_image_smoke_test.py --case missingpart-A01 --execute-api --confirm-cost
```

The first smoke is limited to one request, one image, and one SOP step. Do not run batch or multi-step Azure generation until the response contract and image quality are reviewed.
