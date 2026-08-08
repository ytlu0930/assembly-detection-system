# OpenAI Image API Integration

## Current status

- Provider/model: OpenAI Image API, `gpt-image-2`
- Operation: Images Edit, SDK method `client.images.edit(...)` (`/v1/images/edits`)
- Adapter implemented: yes
- API configured: not asserted by repository code or tests
- API smoke tested: no
- Image quality validated: no
- Default runtime: `MockStepImageProvider`

No real image API request was made while implementing or validating this adapter.

## Input and output contract

The request sends two files in a fixed order: `image[0]` is the current assembly state and `image[1]` is the correct reference. Step 1 uses the test image as current state. A later step uses the previous successful output, while retaining the same correct reference. The prompt requests only the next correction state and preservation of non-target bricks, viewpoint, lighting, background, colors, and geometry.

The adapter returns a `StepImageResult` with `status`, provider/model/mode, output path, duration, request and retry counts, quality/size/format, warning/error, and metadata. Supported outcomes include `success`, `disabled`, `not_configured`, `invalid_input`, `api_error`, `timeout`, `rate_limited`, `invalid_response`, and `output_validation_failed`.

Decoded bytes are validated as a non-empty image before writing. Existing outputs are never overwritten, and paths inside `input/` or `regression_subset/` are rejected.

## Authorization and secrets

Copy `.env.example` to an ignored local `.env` and configure it privately. Never commit `.env` or print the key. Live execution requires all of:

```text
ENABLE_OPENAI_IMAGE_API=true
CONFIRM_OPENAI_IMAGE_API_EXECUTION=true
execute_api=True
```

`OPENAI_API_KEY` must also be configured. Missing credentials return `not_configured`. The OpenAI SDK is imported and a client is created only after authorization, credential, input, path, and budget checks. Exception messages redact key values and token-like strings.

## Retry, timeout, and budget

Defaults are 120 seconds, two retries, one step, and one request per run. Timeout, connection errors, HTTP 429, and HTTP 5xx may retry with finite 2- and 4-second backoff, subject to request budget. Authentication, bad request, invalid input/response, unsupported parameters, and output validation failures do not retry. A one-request smoke therefore performs no retry unless its request budget is deliberately increased.

## Smoke CLI

Plan the smoke without network access:

```powershell
.\venv\Scripts\python.exe scripts\run_openai_image_smoke_test.py --case missingpart-A01 --dry-run
```

After separate approval, the only supported live form is:

```powershell
.\venv\Scripts\python.exe scripts\run_openai_image_smoke_test.py --case missingpart-A01 --execute-api --confirm-cost
```

The CLI still refuses execution unless both environment flags and the key are present. It allows one request/image and writes `inputs_manifest.json`, `prompts/step_01.txt`, `images/step_01.png` on success, `results.json`, and `run_summary.json` under `output/pipeline/openai_step_image_smoke/<run_id>/`. It prints only whether a key is configured, never the key itself.

If OpenAI editing is unavailable or fails, the full pipeline preserves analysis, ErrorReports, annotated image, text correction SOP, flowchart, warnings, and the fixed UI result shape. Runtime should remain mock until the smoke result and human image-quality review are accepted.

## V2 pipeline injection

`step_image_generator_v2.py` no longer imports or constructs an OpenAI client. Every model-backed task is delegated to the provider contract. `--generate-images` alone uses mock by default and cannot authorize API execution. OpenAI additionally requires `--image-provider openai --execute-image-api --confirm-cost`, both environment flags, a valid key, and a positive request budget. Clearly malformed keys—including leading `=`, whitespace, newlines, and placeholders—stop before client creation with a generic non-secret error. The instruction book now replaces the historical flowchart fallback.
