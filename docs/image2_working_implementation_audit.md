# GPT Image 2 working implementation audit

## Audit scope and conclusion

This audit is based only on committed Git history and the current working-tree implementation. No API was called, no secret value was read, and no program file was changed.

The pre-convergence `step_image_generator_v2.py` is commit `32375aee34a2097183dedd8b71eb6b7d7a17f52b` (`Add files via upload`, 2026-08-04 18:26:57 +0800, author `Brian940329`). It is the only version of that file before convergence commit `79b6c78` and is therefore the complete member-A V2 candidate:

```powershell
git show 32375ae:step_image_generator_v2.py
```

The important distinction is:

1. Member A's committed V2 generator uses the public OpenAI SDK client `OpenAI()` and supports both `client.images.generate(...)` and `client.images.edit(...)`.
2. The repository's concrete Azure GPT Image 2 smoke implementation is `tests/test_image.py`. It uses `AzureOpenAI(...)` and `client.images.generate(...)`.
3. Git attributes the original addition of `tests/test_image.py` to commit `d41ff75` by `mirror-creator`, not to the `Brian940329` commit. Commit `32375ae` merely contains that already-existing file in its tree.
4. Git contains no committed `output/output.png`, API response, run log, request ID, or other execution artifact proving that either implementation completed a real GPT Image 2 request. The success message in the source is printed only after a write, but source text is not proof that the script ran.

Accordingly, Git history clearly identifies the implementations and their exact call shapes, but it does **not** independently prove the statement "member A successfully generated an image." Any claim of successful execution requires external evidence such as the original output image, terminal log, Azure request log, or confirmation from the operator.

## Requested Git history

### `step_image_generator_v2.py`

```text
79b6c78 feat: converge correction pipeline with GPT Image 2 provider
32375ae Add files via upload
```

### `main.py`

```text
79b6c78 feat: converge correction pipeline with GPT Image 2 provider
e4646ad Update main.py
2bb0e3d Update main.py
b197fd9 Add image analysis and error detection for LEGO assembly
```

### `.env.example`

```text
79b6c78 feat: converge correction pipeline with GPT Image 2 provider
```

There is no pre-convergence committed `.env.example` to use as credential/configuration evidence.

## Pre-convergence member-A V2 implementation

Source: `32375ae:step_image_generator_v2.py`.

### Client and configuration

```python
from openai import OpenAI

DEFAULT_MODEL = "gpt-image-2"

# Constructor
self.client = client or OpenAI()
```

The module documentation and CLI preflight name `OPENAI_API_KEY`. It does not use `AzureOpenAI`, an Azure endpoint, Azure deployment name, Azure API version, `Authorization`, or `api-key` explicitly. `OpenAI()` lets the OpenAI SDK obtain its normal platform configuration from the environment.

### Generate path

For an `api_mode="generate"` task without a reference image, the V2 implementation calls:

```python
self.client.images.generate(
    model=self.model,
    prompt=prompt,
    size=self.size,
    quality=self.quality,
    output_format=self.output_format,
    background=self.background,
    n=1,
)
```

For a nominal generate task that has a reference image, it deliberately switches to `images.edit`, because the generations endpoint does not accept an input image.

### Edit path

For assembly correction, the implementation calls:

```python
request = {
    "model": self.model,
    "image": image_files,
    "prompt": prompt,
    "size": self.size,
    "quality": self.quality,
    "output_format": self.output_format,
    "background": self.background,
    "n": 1,
}
if mask_file is not None:
    request["mask"] = mask_file

self.client.images.edit(**request)
```

The first edit input is the editable base. Depending on the task, the list may additionally contain the correct reference and annotated localization image. It can generate and send a PNG mask derived from a bbox. Later assembly edits use the preceding assembly output as their base.

### Direct answers for member-A V2

| Question | Answer from `32375ae:step_image_generator_v2.py` |
|---|---|
| Generate or edit? | Both. Text-only standalone tasks use `generate`; reference-guided standalone and assembly correction tasks use `edit`. |
| Client class | `openai.OpenAI` |
| Endpoint/base URL | Not specified; SDK default/public OpenAI configuration. |
| Deployment/model | `model=self.model`, default `gpt-image-2`; this is a model ID, not an Azure deployment env value. |
| Auth/env | `OPENAI_API_KEY`; header construction is delegated to the OpenAI SDK. |
| API version | None specified. |
| Reference image | Yes. It can be one of multiple `image` files sent to `images.edit`. |
| Mask | Optional; generated from bbox and passed as `mask`. |
| Transport | OpenAI Python SDK, not REST written by the project and not Azure SDK configuration. |

## Azure GPT Image 2 implementation found in history

The exact Azure call shape appears in `tests/test_image.py` at `d41ff75` and remains visible in the pre-convergence `32375ae` tree:

```python
from openai import AzureOpenAI

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
image_deployment = os.getenv("IMAGE_DEPLOYMENT")

client = AzureOpenAI(
    api_version="2025-04-01-preview",
    azure_endpoint=endpoint,
    api_key=api_key,
)

result = client.images.generate(
    model=image_deployment,
    prompt="一隻可愛的黑貓坐在書桌前",
    size="1024x1024",
)
```

Its direct answers are:

| Question | Historical Azure implementation |
|---|---|
| Generate or edit? | `generate` only. |
| Client class | `openai.AzureOpenAI` |
| Endpoint/base URL | `azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")`; expected value is the Azure resource base endpoint, not a hand-built `/openai/deployments/...` URL. |
| Deployment/model | Azure deployment name from `IMAGE_DEPLOYMENT`, passed as `model=image_deployment`. |
| Auth/env | `AZURE_OPENAI_API_KEY`, passed as `api_key`; the SDK builds the Azure API-key request header. No explicit `Authorization` header exists in the script. |
| API version | `2025-04-01-preview` |
| Reference image | No. |
| Mask | No. |
| Transport | OpenAI Python SDK's `AzureOpenAI` client. It is Azure-configured OpenAI SDK usage, not hand-written REST. |

`AZURE_API_KEY` is not used by this image script. `IMAGE_DEPLOYMENT` is used, while the current provider instead uses `AZURE_IMAGE_DEPLOYMENT`.

## Comparison with current `AzureOpenAIImageProvider`

Current source: `utils/azure_openai_image_provider.py`.

| Dimension | Historical Azure call | Current provider |
|---|---|---|
| Operation | `client.images.generate(...)` | HTTP POST to Azure `/images/edits` |
| Client | `openai.AzureOpenAI` | Lazy `httpx.Client` |
| Endpoint input | Azure resource base passed as `azure_endpoint` | Resource base is validated, then the provider constructs `/openai/deployments/{deployment}/images/edits?api-version={version}` |
| Deployment | `IMAGE_DEPLOYMENT`, passed as SDK `model` | `AZURE_IMAGE_DEPLOYMENT`, embedded in URL path |
| API version | `2025-04-01-preview` | `AZURE_IMAGE_API_VERSION`, default `2024-02-01` |
| API key | `AZURE_OPENAI_API_KEY` passed to SDK | Same env name, but provider builds either `Authorization: Bearer ...` or `api-key: ...` itself |
| Auth default | SDK API-key handling | `AZURE_IMAGE_AUTH_MODE=bearer` by default; `api_key` mode optional |
| Input image | None | Exactly one source image binary |
| Reference image | None | Metadata only; never sent as a second binary |
| Mask | None | Optional multipart `mask` |
| Request fields | `model`, `prompt`, `size` | Multipart `image`, optional `mask`, and `prompt`; deployment is in URL |
| Output | Decodes `data[0].b64_json` | Decodes and strictly validates `data[0].b64_json`, then validates the written image |
| Safety/reliability | No gates, budget, retry classification, or response validation beyond indexing | Triple gate, key preflight/redaction, budget, timeout, selective retries, status mapping, and Pillow validation |

The largest compatibility risks are operation and API-version mismatch. The only Azure image call found in history proves the intended SDK syntax for **generation** with `2025-04-01-preview`; it does not demonstrate that the teacher deployment supports the current hand-built **edit** route at `2024-02-01`.

## Minimal migration plan (proposal only)

Because Git does not contain proof of a successful run, this plan is conditional: use it only after the team confirms that `tests/test_image.py` is the known-good Azure GPT Image 2 execution.

To reproduce that known call shape inside the current provider while retaining its gates, validation, budgets, and structured results:

1. In `utils/azure_openai_image_provider.py`, replace lazy `httpx.Client` creation with lazy `openai.AzureOpenAI(api_version=self.api_version, azure_endpoint=self.endpoint_base, api_key=key)` construction.
2. Change the default API version from `2024-02-01` to the demonstrated `2025-04-01-preview`, or require it explicitly through configuration.
3. Accept `IMAGE_DEPLOYMENT` as the deployment setting (preferably as a compatibility fallback after `AZURE_IMAGE_DEPLOYMENT`) and pass it as `model=self.deployment`.
4. Replace the multipart REST `post(...)` block with `client.images.generate(model=self.deployment, prompt=prompt, size=self.size)` and keep the existing base64/output validation around its response.
5. Remove endpoint-path construction and explicit `Authorization`/`api-key` header creation from the request path; the SDK owns both.
6. Mark source image, reference image, and mask as unsupported for this generate-compatible mode. Do not silently claim correction/edit semantics when none of those inputs is submitted.

That is the smallest faithful migration to the historical Azure call. It is **not** a drop-in correction-image solution: it changes the provider from image editing to text-only image generation. If correction must preserve an assembly source image, the team needs a separately verified Azure `images.edit` request contract before migrating.

## Evidence limits and next verification

No API was called during this audit. To turn the historical Azure implementation from a code candidate into a verified baseline, obtain at least one of:

- the original `output/output.png` together with a trustworthy timestamp/run record;
- terminal output from the actual execution;
- Azure request/application logs showing deployment, operation, API version, status, and time;
- confirmation from the operator identifying the exact commit/script and successful invocation.

Secrets should not be added to Git or copied into this audit.
