# llama.cpp Backend

The llama.cpp backend serves **GGUF models** through `llama-server`. It is the
right choice for quantized inference (Q4/Q5/Q8 and friends), running large
models that only fit with CPU/GPU layer offload, and MoE models whose experts
can be pushed to system RAM.

llmux never installs llama.cpp on your host — every profile runs in the
official `ghcr.io/ggml-org/llama.cpp:server-cuda` container image (or one you
built from source). llmux renders the Compose files, injects the
`llama-server` command, streams logs, and validates the result.

## How a llama.cpp profile starts

When you start a llama.cpp profile, llmux runs this pipeline:

1. **`.env.common` check** — refuses to start if `.env.common` is missing.
2. **Profile lookup** — the profile must exist under `llamacpp/` in
   `profiles.yaml`.
3. **Image resolution** — the selected image source is resolved to a concrete
   tag (see [Image sources](#image-sources)).
4. **Render the per-profile `.env`** — `profiles.yaml` is rendered to
   `.runtime/llamacpp/<name>.env`.
5. **Render the command override** — `scripts/llamacpp/render-override.py`
   reads `config/llamacpp/<name>.yaml` and writes
   `.runtime/llamacpp/override-<name>.yaml`, which injects the `command:` block
   for `llama-server`.
6. **`docker compose up -d`** — the base Compose file, the optional dev
   overlay, and the per-profile override are merged and the container starts.
7. **Post-start validation** — llmux polls `docker inspect`; if the container
   exits or goes unhealthy during startup it surfaces the last log lines as an
   error. It then waits for `/health` to respond before reporting success.
8. **Persist last-active profile** — the profile name is written to
   `.current-profile.llamacpp`.

## Container shape

The base service is defined in `compose/llamacpp/docker-compose.yaml`:

```yaml
services:
  llama-server:
    image: ${LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}
    container_name: ${CONTAINER_NAME}
    ports:
      - "${LLAMA_PORT:-8080}:8080"
    volumes:
      - ${HF_CACHE_PATH}:/root/.cache/huggingface
    environment:
      - HF_TOKEN=${HF_TOKEN}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['${GPU_ID:-0}']
              capabilities: [gpu]
    # command is injected by the per-profile override file.
```

Key points:

- **`command:` is not in the base file** — it is injected by
  `.runtime/llamacpp/override-<name>.yaml`, generated per profile by
  `render-override.py`. This lets multiple llama.cpp profiles run concurrently
  without clobbering each other's command.
- **Only the HF cache is mounted** — `${HF_CACHE_PATH}` is bound into the
  container as `/root/.cache/huggingface`. There is no per-model bind mount;
  `llama-server` downloads the GGUF itself (see below).
- **GPU access via `deploy.resources`** — the profile's `gpu_id` is passed as a
  Compose device-id list. Because that list format cannot reliably expand a
  comma-separated value, llama.cpp profiles are effectively single-GPU; vLLM
  uses `NVIDIA_VISIBLE_DEVICES` instead for multi-GPU tensor parallelism.

## GGUF auto-download

You do **not** need the `hf` CLI on the host or a manual `hf download` step.
`render-override.py` resolves the model from the profile's `hf_repo` /
`hf_file` and emits `-hf <repo> -hff <file>` into the `llama-server` command.
`llama-server` then downloads the GGUF directly inside the container.

Because `${HF_CACHE_PATH}` is mounted into the container, the downloaded file
lands in the host's Hugging Face cache and persists across restarts — the first
start is slow, subsequent starts reuse the cached file. This mirrors the vLLM
flow exactly.

```yaml
# profiles.yaml
profiles:
  - name: gemma-3-4b
    backend: llamacpp
    config_name: gemma-3-4b
    hf_repo: unsloth/gemma-3-4b-it-GGUF
    hf_file: gemma-3-4b-it-q4_k_m.gguf
```

!!! note "`hf_repo` and `hf_file` are required"
    `render-override.py` raises an error if `hf_repo` is empty or no `.gguf`
    filename can be resolved. The profile's `hf_repo` / `hf_file` are the
    source of truth; the config's `model-file` key is only a display fallback.

!!! tip "Gated repos still need `HF_TOKEN`"
    `HF_TOKEN` from `.env.common` is passed into the container as an
    environment variable, so `llama-server` can download gated GGUF repos.

## config: llama-server flags as YAML

`config/llamacpp/<name>.yaml` is a flat list of `llama-server` flags. The key
naming rule is simple: drop the leading `--`. `--ctx-size 32768` becomes
`ctx-size: 32768`; a boolean flag like `--jinja` becomes `jinja: true`.

`render-override.py` translates the YAML into the command line:

- Scalar values → `--key value`.
- `true` → `--key` (bare flag); `false` and `null` are skipped.
- List values → the flag is repeated once per item.
- `--host 0.0.0.0 --port 8080 --no-webui` are always prepended; the WebUI is
  forced off (`webui` / `no-webui` keys in the config are ignored).
- `override-tensors` entries are rendered as repeated `-ot <pattern>`.
- `extra-args` entries are appended verbatim at the end.

A representative config (see `config/llamacpp/example.yaml`):

```yaml
ctx-size: 8192
n-gpu-layers: 99          # layers on GPU; 99 = all
cache-type-k: bf16        # KV cache precision: f16 / bf16 / q8_0 / q4_0 ...
cache-type-v: bf16
temp: 0.7
top-p: 0.9
top-k: 40
alias: your-model         # name shown in /v1/models
jinja: true               # Jinja chat template (recommended for recent models)
# parallel: 4             # parallel request slots
# cont-batching: true     # continuous batching
# flash-attn: true        # flash attention
override-tensors: []      # MoE expert offload — see below
extra-args: []            # verbatim args for flags the schema can't express
```

### Distinctive flags

| Concern | Flag(s) | Effect |
|---|---|---|
| **GPU/CPU split** | `n-gpu-layers` | How many layers to offload to GPU. `99` keeps everything on GPU; lower it to fit a model that overflows VRAM. |
| **KV cache precision** | `cache-type-k`, `cache-type-v` | Quantize the KV cache (`q8_0`, `q4_0`, ...) to stretch context length within a fixed VRAM budget. |
| **MoE expert offload** | `override-tensors` | Regex-match tensor names and pin matches to CPU — see below. |
| **Throughput** | `parallel`, `cont-batching`, `flash-attn` | Parallel request slots, continuous batching, and flash attention. |

### MoE expert CPU offload (`override-tensors`)

For Mixture-of-Experts models — the A3B-style families in particular — the
expert FFN tensors dominate memory while the attention and embedding tensors
are comparatively small. `override-tensors` lets you keep attention and
embeddings on the GPU while pushing the expert tensors to system RAM, which is
often what makes a large MoE model fit at all.

Each entry is a regex matched against tensor names, paired with a destination:

```yaml
# config/llamacpp/qwen3-a3b.yaml
override-tensors:
  - ".ffn_.*_exps.=CPU"
```

`render-override.py` turns each entry into a repeated `-ot <pattern>` argument.
The example above matches the expert FFN tensors (`ffn_*_exps`) and routes them
to CPU, leaving everything else on the GPU. The TUI's Quick Setup pre-fills
this exact pattern for MoE models.

## Image sources

llama.cpp ships as a **single official `ghcr.io` tag**
(`ghcr.io/ggml-org/llama.cpp:server-cuda`), so there is no "Local Latest /
Official Release / Nightly" split like vLLM has. The start dialog offers three
options instead:

=== "Default Image"

    `ghcr.io/ggml-org/llama.cpp:server-cuda`, or whatever `LLAMACPP_IMAGE` is
    set to in `.env.common`. This is the normal choice.

=== "Dev Build"

    A local `llamacpp-dev:<branch>` image built from a llama.cpp git checkout.
    Built on demand if missing — or rebuilt if its repo/branch labels do not
    match. See [Dev builds](#dev-builds) below.

=== "Custom Tag"

    Any `<repo>:<tag>` you type, passed through as-is.

A profile can also pin a dev image permanently with `image_tag` in
`profiles.yaml` (e.g. `image_tag: llamacpp-dev:master`); the backend layers
`compose/llamacpp/docker-compose.dev.yaml` automatically when it sees a
`llamacpp-dev:` tag.

## Dev builds

A dev build produces a local `llamacpp-dev:<tag>` image from a llama.cpp git
checkout — useful for testing an unreleased branch, a PR, or a fork (for
example, one with [MTP](#mtp-multi-token-prediction) support). Trigger it from
the start dialog ("Dev Build") or with
`llmux image build-dev --backend llamacpp`.

The pipeline (shared with vLLM, in `tui/common/dev_build.py`):

1. **Clone or update** `.llamacpp-src/` from `LLAMACPP_REPO_URL` (default
   `https://github.com/ggml-org/llama.cpp.git`) at `LLAMACPP_BRANCH` (default
   `master`). Both are overridable in `.env.common`. If the existing checkout's
   `origin` URL differs from the requested repo, llmux refuses to switch
   remotes silently.
2. **Detect GPU arch** — `nvidia-smi` is queried for compute capability and
   passed as `--build-arg CUDA_DOCKER_ARCH=<arch>` (e.g. `89` or `86;89`), so
   the build only targets your GPU. Without `nvidia-smi`, it falls back to the
   Dockerfile's slower multi-arch path.
3. **`docker build`** — `.devops/cuda.Dockerfile`, target `server`, tagged both
   `llamacpp-dev:<tag>` and `llamacpp-dev:<branch>`. The image is labeled with
   the repo URL, branch, and commit hash so llmux can tell whether a cached tag
   still matches the requested source.

When you start with a dev tag, llmux rebuilds automatically if the image is
missing **or** if its labels show it was built from a different repo/branch. A
profile pinned to a dev tag that does not exist locally fails with a hint to
build it first.

## MTP (Multi-Token Prediction)

Multi-Token Prediction is a speculative-decoding-style technique where the
model predicts several tokens per step. Running it under llama.cpp through
llmux requires three things to line up:

**1. An MTP-capable `llama.cpp` build.** MTP support lands on specific
branches before it reaches the official `server-cuda` image. Use a
[dev build](#dev-builds) to build a `llamacpp-dev:<branch>` image from a
branch that has MTP support:

```bash
# point .env.common at the branch, then:
llmux image build-dev --backend llamacpp --branch <mtp-branch>
```

Then pin the profile to that image: `image_tag: llamacpp-dev:<mtp-branch>`.

**2. A GGUF that includes the MTP head.** A regular GGUF has no MTP weights.
You need a model exported *with* the MTP head — for example the
`*-MTP-GGUF` repos published by unsloth. Set the profile's `hf_repo` /
`hf_file` to that repo and file.

**3. The MTP flags in the config.** Add them to `extra-args` in
`config/llamacpp/<name>.yaml` so `render-override.py` appends them verbatim:

```yaml
# config/llamacpp/qwen3-mtp.yaml
n-gpu-layers: 99
ctx-size: 32768
extra-args:
  - --spec-type
  - mtp
  - --spec-draft-n-max
  - "4"           # tokens predicted per step
  - --parallel
  - "1"
```

!!! warning "MTP wants `--parallel 1`"
    MTP is paired with `--parallel 1` — the speculative path is built around a
    single request stream. Combining it with multiple parallel slots is not
    the supported configuration.

!!! note "Why `extra-args`?"
    The config schema only knows the common `llama-server` flags. `extra-args`
    is the escape hatch for newer or niche flags — `render-override.py` appends
    every item in the list to the command line untouched.

## See also

- [vLLM backend](vllm.md) — the HF Transformers counterpart
- [Backend comparison](comparison.md) — feature matrix and "when to use which"
- [Troubleshooting](../troubleshooting.md) — stuck downloads, OOM, dev image
  not found
