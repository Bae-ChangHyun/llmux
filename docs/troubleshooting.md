# Troubleshooting

Common problems when starting and running profiles, with the symptom, the
underlying cause, and the fix. If a container fails, the fastest first step is
always to open the profile in the TUI and stream its logs (`l`) — llmux already
surfaces the last log lines on a failed start, but the full log tells the rest
of the story.

## Container stuck at `health: starting`

**Symptom.** The dashboard shows the profile as `starting` and it never moves
to `healthy`. `docker ps` shows `(health: starting)`.

**Cause.** This is almost always a model still downloading or loading. The
health checks only pass once the server is actually serving:

- vLLM's healthcheck has a **600s `start_period`**; llmux additionally waits
  for `/v1/models` to return a served model id.
- llama.cpp's healthcheck has a **300s `start_period`**; llmux waits for
  `/health` to respond.

A first start that downloads weights — a multi-GB HF repo, or a GGUF pulled by
`llama-server` — can easily exceed those windows on a slow link.

**Fix.** Stream the logs and watch the download/load progress. Once the model
finishes loading, the status flips to `healthy` on the next refresh.
Subsequent starts reuse the cached files and come up fast.

## GPU out of memory

**Symptom.** The container exits or goes unhealthy seconds after start. llmux
reports `container exited during startup` with the last log lines, which
mention CUDA OOM during engine/model init.

**Cause.** The model plus its KV cache does not fit in the allocated VRAM.

**Fix.**

=== "vLLM"

    - Lower `gpu-memory-utilization` in `config/vllm/<name>.yaml` (it is the
      fraction of each GPU's VRAM vLLM may use).
    - Lower `max-model-len` — a shorter context length means a smaller KV
      cache.
    - Raise `tensor_parallel_size` on the profile (and add GPUs to `gpu_id`,
      e.g. `"0,1"`) to shard the model across more GPUs.

=== "llama.cpp"

    - Lower `n-gpu-layers` in `config/llamacpp/<name>.yaml` to offload more
      layers to CPU.
    - Quantize the KV cache with `cache-type-k` / `cache-type-v` (e.g.
      `q8_0`).
    - For MoE models, add an `override-tensors` rule (e.g.
      `".ffn_.*_exps.=CPU"`) to push expert tensors to system RAM.

!!! tip "Size it before you start it"
    Use the memory estimator (`m` in the TUI, or `llmux system mem-estimate
    <model>`) to check whether a model fits *before* launching it.

## Port conflict

**Symptom.** The start is refused immediately with
`Error: Port <n> is already in use by ...`.

**Cause.** llmux checks the profile's port before starting. The port may be
held by:

- another llmux profile's container (the message names the profile),
- a non-llmux Docker container (the message names the container), or
- a plain local process bound to `127.0.0.1:<port>` (the message says "another
  local process").

**Fix.** Change the profile's `port` in `profiles.yaml` (or via `llmux profile
edit <name> --port <n>`), or stop whatever is holding the port. The
cross-backend conflict gate also warns pre-start when another profile — on
*either* backend — overlaps.

## `.env.common` not found

**Symptom.** The start fails with
`Error: .env.common not found` (vLLM) or
`✗ .env.common 없음` (llama.cpp).

**Cause.** `.env.common` holds shared host paths and tokens and is gitignored —
it does not exist until you create it.

**Fix.**

```bash
cp .env.common.example .env.common
$EDITOR .env.common      # set HF_CACHE_PATH, MODEL_DIR, HF_TOKEN, ...
```

Then run `llmux env-check` to confirm. Note that `HF_CACHE_PATH` **must be an
absolute path** — a relative value is rejected. If the profile uses LoRA,
`LORA_BASE_PATH` must also be set and absolute.

## Gated model download fails (missing `HF_TOKEN`)

**Symptom.** The container starts but the engine cannot fetch the weights;
logs show a 401/403 from Hugging Face, or "gated repo" / "access denied".

**Cause.** `HF_TOKEN` in `.env.common` is empty or lacks access to the repo.
Public models download fine without a token, but gated repos (some Llama and
Gemma weights, for example) do not.

**Fix.** Set `HF_TOKEN` in `.env.common` to a token with access to that repo.
It is passed into both backends' containers (`HUGGING_FACE_HUB_TOKEN` for vLLM,
`HF_TOKEN` for llama.cpp), so the in-container download can authenticate. Make
sure you have also accepted the model's license on the Hugging Face website.

## Dev image not found — "build it first"

**Symptom.** Starting a profile that points at a `vllm-dev:<tag>` or
`llamacpp-dev:<tag>` image fails with `Image ... not found` and, for
llama.cpp, a hint to build it.

**Cause.** Dev images are built locally and are not in any registry. The tag
either was never built, or — for the auto-build paths — its repo/branch labels
do not match the requested source.

**Fix.**

- If you selected **Dev Build** in the start dialog, llmux builds the image on
  demand; just let it run (the first build is slow).
- If the profile *pins* a dev tag via `image_tag` or `--tag`, build it
  explicitly first:

    ```bash
    llmux image build-dev --branch main --tag custom               # vLLM
    llmux image build-dev --backend llamacpp --branch master       # llama.cpp
    ```

- llmux **rebuilds automatically** when a cached dev tag's labels show it came
  from a different repo/branch — this is expected, not an error.

!!! note "Swapping the source repo"
    If `.vllm-src/` or `.llamacpp-src/` already exists with a different
    `origin` URL, llmux refuses to switch remotes silently. Move or delete that
    checkout directory yourself, then rebuild.

## `Local Latest` shows "no images" (vLLM)

**Symptom.** The vLLM start dialog's **Local Latest** option is labeled "no
images" and the start is refused.

**Cause.** llmux only counts **semver-style** tags (`vX.Y.Z`) as "local
latest". `latest` and `nightly` are deliberately skipped because they do not
describe their own contents. If you only have those aliases locally, there is
no versioned tag to pick.

**Fix.** Pull a specific version, then retry:

```bash
docker pull vllm/vllm-openai:v0.19.1
```

Or choose **Official Release** in the dialog, which resolves Docker Hub's
current stable version and pulls it by explicit tag.

## No GPU / `nvidia-smi` unavailable

**Symptom.** A dev build logs that it could not detect the GPU and falls back
to a multi-arch build; or starting any profile fails because the GPU runtime is
not visible.

**Cause.** The host has no working `nvidia-smi`, or Docker's NVIDIA Container
Toolkit is not set up.

**Fix.**

- Confirm GPU passthrough works at the Docker level first:

    ```bash
    docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
    ```

    If this does not print your GPUs, fix the
    [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
    setup — llmux cannot work around a broken GPU runtime.

- For **dev builds** specifically, a missing `nvidia-smi` only costs you the
  fast arch-detected build path; the build still succeeds via the slower
  multi-arch fallback.

## Corrupted or incomplete model file

**Symptom.** The container starts but the engine crashes while loading the
weights — checksum mismatch, truncated file, "failed to load model", or a
GGUF/safetensors parse error.

**Cause.** A download that was interrupted (network drop, killed mid-pull) can
leave a partial file in the Hugging Face cache.

**Fix.** Delete the bad file/blob from `HF_CACHE_PATH` (the host directory
mounted into the container) and start again — both backends re-download
anything missing from the cache. If it keeps failing on the same file, the
upstream repo file itself may be the problem; try a different quantization or a
mirror repo.

## Copying logs out of the TUI

**Symptom.** You want to paste a container log somewhere but the TUI captures
the keyboard.

**Fix.** Shift+drag to select text, then Ctrl+C to copy — this is standard
Textual behavior. See the
[Textual FAQ](https://textual.textualize.io/FAQ/#how-can-i-select-and-copy-text-in-a-textual-app).

## See also

- [vLLM backend](backends/vllm.md) — the vLLM start pipeline in detail
- [llama.cpp backend](backends/llamacpp.md) — the llama.cpp start pipeline in
  detail
- [Backend comparison](backends/comparison.md) — feature matrix
