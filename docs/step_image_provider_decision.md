# Step-Image Provider Decision

## Formal Provider Selected

- **Selected:** OpenAI Image API / GPT Image 2 (model id: `gpt-image-2`)
- Operation: Image Editing
- Endpoint: `/v1/images/edits`
- Status: **Selected / Adapter Implemented**
- Runtime default: `MockStepImageProvider`
- Real API execution: not performed
- API key: not read, requested, or validated by this task
- Activation gate: approve budget/data handling and explicitly authorize one API smoke test with both environment flags and `execute_api=True`

The selected provider supports image editing and multiple image inputs, making it suitable for the planned “previous-step image + correct reference image + current SOP step” flow. High-fidelity image inputs can reinforce appearance and viewpoint, and the existing `StepImageProvider` contract already supplies source, reference, prompt, output, and metadata. This fits a ShowHowTo-style sequence while keeping each step independently reviewable.

Selection does not mean API or quality validation. `OpenAIImageProvider` now implements a guarded `client.images.edit(...)` adapter with lazy client creation, source/reference ordering, finite retry, budgets, response validation, and structured statuses. CI, offline replay, and UI smoke tests continue to use `MockStepImageProvider`; this implementation task made no real request.

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

The formal consumer is `step_image_generator_v2.StepImageGeneratorV2`. It receives a provider through dependency injection or the explicit factory and contains no direct OpenAI SDK request path. Mock remains the default for `main.py`, batch, UI, tests, and `--generate-images` unless OpenAI and every execution/cost gate are separately selected.
