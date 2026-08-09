# Step-Image Provider Decision

## Formal Provider Selected

- **Selected deployment provider:** Azure-hosted OpenAI GPT Image 2
- Deployment/model: `gpt-image-2` / `gpt-image-2` (teacher deployment version `2026-04-21`)
- Operation: Image Editing
- Endpoint: `{azure_endpoint}/openai/deployments/gpt-image-2/images/edits?api-version=2024-02-01`
- Status: **Adapter implementation / Smoke pending**
- Runtime default: `MockStepImageProvider`
- Real API execution: not performed
- API key: not read, requested, or validated by this task
- Activation gate: approve budget/data handling and explicitly authorize one API smoke test with both environment flags and `execute_api=True`

The selected provider supports image editing and multiple image inputs, making it suitable for the planned “previous-step image + correct reference image + current SOP step” flow. High-fidelity image inputs can reinforce appearance and viewpoint, and the existing `StepImageProvider` contract already supplies source, reference, prompt, output, and metadata. This fits a ShowHowTo-style sequence while keeping each step independently reviewable.

Selection does not mean API or quality validation. `AzureOpenAIImageProvider` implements the teacher-provided single-image multipart contract through `httpx`. The correct-reference path remains metadata and prompt context; it is not sent as a second binary input until the deployment capability is verified. The OpenAI Platform provider remains an alternate provider. CI, offline replay, and UI continue to use Mock.

## Candidate comparison

| Candidate | Image editing | Multiple references | Preserve assembly appearance / angle | Previous-step input | Local edit | Cost / latency | API or GPU | Cross-step risk | Integration difficulty | Best fit | Status / limitation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenAI Image API / `gpt-image-2` | Selected editing operation | Two ordered image inputs implemented | Preservation still requires measurement | Implemented sequentially | Supported semantically, not CAD-constrained | Paid; network latency | API, approval, credential setup | Medium; each step can drift | Medium | MVP and research evaluation | **Selected / Adapter Implemented** |
| Local diffusion / image editing | Yes with inpainting/editing stack | Possible through conditioning | Requires tuning and stable masks | Yes | Yes | No per-call fee; potentially slow | Capable GPU and model storage | Medium to high | High | Research alternative | No approved model or reproducible GPU baseline |
| Existing instruction-image template fallback | Deterministic composition | Yes through code/layout | Strong because source pixels/assets are reused | Yes | Limited to predefined operations | Low; fast | CPU only | Low | Medium | Deterministic fallback | Cannot synthesize unseen disassembly states |
| MockStepImageProvider | No semantic editing | Metadata only | Not applicable | Chaining metadata supported | No | Free; fast | CPU only | None | Complete | Tests and offline smoke | Not a formal GPT Image 2 result |

## Risks and required evaluation

- GPT Image 2 is not a CAD system and cannot guarantee exact geometry, counts, holes, or connection points.
- It may change non-target parts, background, lighting, or camera angle.
- Repeated edits may accumulate errors, so every step must re-reference the correct image.
- Formal research must measure content consistency, target-only edit correctness, latency, and cost.
- Mock output must never be described as a GPT Image 2 result.
- A separately authorized, one-request smoke test and human quality review are still required.

The provider satisfies `utils/step_image_provider_contract.py` with status, model, mode, output, duration, request/retry counts, quality, size, format, warning/error, and metadata. Adapter implemented does not mean API configured, API smoke tested, or image quality validated.

## Canonical V2 integration

The formal consumer is `step_image_generator_v2.StepImageGeneratorV2`. It supports `mock`, `openai`, and `azure_openai` through the same provider contract. Mock remains the safe default. Azure requires provider selection, execute and cost confirmation, both environment gates, valid configuration, and request budget.
| Azure-hosted OpenAI / `gpt-image-2` | Selected deployment editing operation | First adapter intentionally single-image | Prompt-guided; requires smoke and quality validation | Implemented sequentially | Supported semantically | Hosted API | Teacher Azure credential | Medium | Medium | Formal deployment provider | **Adapter implemented / Smoke pending** |
