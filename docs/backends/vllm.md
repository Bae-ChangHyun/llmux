# vLLM Backend

The vLLM backend serves **Hugging Face Transformers models** for high-throughput
inference. It is the right choice when you want to run a model directly from its
HF repo (safetensors weights), need tensor parallelism across multiple GPUs, or
want to attach LoRA adapters at serve time.

llmux never installs vLLM on your host — every profile runs in an official
`vllm/vllm-openai` container image (or one you built from source). llmux only
renders the Compose files, resolves the image tag, streams logs, and verifies
the result.

## How a vLLM profile starts

When you start a vLLM profile, llmux runs a fixed pipeline before and after
`docker compose up`:

1. **Port conflict check** — refuses to start if the profile's port is already
   held by another container or a local process.
2. **`.env.common` validation** — `HF_CACHE_PATH` must be set and absolute. If
   the profile has LoRA enabled, `LORA_BASE_PATH` must also be set and absolute.
3. **Config validation** — the linked `config/vllm/<name>.yaml` must exist and
   point at a real model. If none is linked, llmux auto-links a default config
   named after the profile and generates a stub.
4. **GPU overlap warning** — if any other *running* container (vLLM or
   llama.cpp) uses an intersecting `gpu_id`, llmux warns but continues.
5. **Image resolution** — the selected image source is resolved to a concrete
   tag (see [Image sources](#image-sources)).
6. **`docker compose up -d`** — Compose files are merged and the container
   starts detached.
7. **Post-start validation** — llmux polls `docker inspect`; if the container
   exits or goes unhealthy during startup (for example a GPU OOM during engine
   init), it surfaces the last log lines as an error instead of reporting a
   false success. It then waits for `/v1/models` to return a served model id.
8. **Version verification** — for versioned tags, llmux execs into the
   container (`import vllm; print(vllm.__version__)`) and warns if the running
   version does not match the tag you picked.

## Container shape

The base service is defined in `compose/vllm/docker-compose.yaml`:

```yaml
services:
  vllm:
    image: vllm/vllm-openai:${VLLM_VERSION}
    container_name: ${CONTAINER_NAME}
    runtime: nvidia
    volumes:
      - ${HF_CACHE_PATH}:/root/.cache/huggingface
      - ./config/vllm:/config/
    ports:
      - "${VLLM_PORT}:8000"
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
      - NVIDIA_VISIBLE_DEVICES=${GPU_ID}
      - VLLM_ENABLE_V1_MULTIPROCESSING=0
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
      - VLLM_DISABLE_COMPILE_CACHE=1
    shm_size: "16g"
    command: --config /config/${CONFIG_NAME}.yaml --tensor-parallel-size=${TENSOR_PARALLEL_SIZE} --trust-remote-code ${LORA_OPTIONS:-}
```

Key points:

- **`runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES`** — the profile's `gpu_id` is
  passed straight through as `NVIDIA_VISIBLE_DEVICES`. A value of `"0,1"`
  exposes both GPUs to the container, which is what makes multi-GPU tensor
  parallelism work. (llama.cpp uses Compose's `deploy.resources` device list
  instead — see the [comparison](comparison.md).)
- **`--config /config/<name>.yaml`** — your `config/vllm/<name>.yaml` is mounted
  read-only into the container and passed to `vllm serve` as its config file.
  Every key in that YAML maps 1:1 to a `vllm serve` CLI flag.
- **`--tensor-parallel-size`** — injected from the profile, not the config.
- **`--trust-remote-code`** — always passed so models with custom code load.
- **`shm_size: "16g"`** — required headroom for vLLM's multiprocessing workers.

!!! note "Config vs. profile"
    The **profile** (`profiles.yaml`) holds launcher-level settings: port,
    `gpu_id`, `tensor_parallel_size`, LoRA toggles, extra pip packages, env
    overrides. The **config** (`config/vllm/<name>.yaml`) holds per-model engine
    flags passed to `vllm serve`. See the example at `config/vllm/example.yaml`.

A minimal config:

```yaml
# config/vllm/qwen3-8b.yaml
model: Qwen/Qwen3-8B            # HF repo id or local path
gpu-memory-utilization: 0.90   # fraction of each GPU's VRAM vLLM may use
max-model-len: 32768           # context length
# tensor-parallel-size: 2      # set on the profile instead
```

## Image sources

The start dialog offers five ways to choose the image. llmux deliberately
**refuses the `:latest` alias** — it is a moving target whose contents change
without the tag name changing, so "what version is actually running" becomes
unknowable. Stable picks always resolve to an explicit semver tag first.

=== "Local Latest"

    The highest `vX.Y.Z` tag you already have locally. Only semver-style tags
    are considered — `latest` and `nightly` are skipped because they do not
    self-describe. **No pull happens**; if you have no versioned tag locally,
    the picker shows "no images" and the start is refused with a hint to
    `docker pull vllm/vllm-openai:<version>`.

=== "Official Release"

    Docker Hub's current stable release, resolved to its explicit version
    (e.g. `v0.19.1`). Pulled with `--pull always` on that exact version tag —
    never the bare `:latest` alias.

=== "Nightly"

    `vllm/vllm-openai:nightly`. Pulled with `--pull always` every time —
    `nightly` is an intentionally rolling target with no versioned alternative,
    so llmux always wants the freshest copy.

=== "Dev Build"

    A local `vllm-dev:<tag>` image built from a vLLM git checkout. If the image
    is missing — or its repo/branch labels do not match the requested
    repo/branch — llmux builds it first, then starts. See
    [Dev builds](#dev-builds) below.

=== "Custom Tag"

    Any tag you type, **except `latest`** (rejected). Uses default Compose pull
    behavior — no forced `--pull`.

!!! tip "Rule of thumb"
    If you ever pull by hand, pull an explicit version:
    `docker pull vllm/vllm-openai:v0.19.1`. Never `docker pull
    vllm/vllm-openai` or `:latest`.

### Version verification

After a versioned-tag start, llmux execs into the running container and reads
`vllm.__version__`. If the reported version does not match the tag you picked,
it prints a warning in the log panel — the tag may have been retagged by hand
or pulled at a different time. Verification failures are warnings, not errors;
they never block the container.

## Tensor parallelism (multi-GPU)

Set the profile's `gpu_id` to a comma-separated list and its
`tensor_parallel_size` to match:

```yaml
# profiles.yaml
profiles:
  - name: qwen3-32b-tp2
    backend: vllm
    config_name: qwen3-32b
    model_id: Qwen/Qwen3-32B
    gpu_id: "0,1"
    tensor_parallel_size: 2
```

`gpu_id: "0,1"` becomes `NVIDIA_VISIBLE_DEVICES=0,1` inside the container, and
`tensor_parallel_size: 2` is injected as `--tensor-parallel-size=2`. The
cross-backend conflict gate also understands comma-separated GPU sets, so
overlap with another running profile is detected before start.

## LoRA adapters

vLLM can serve LoRA adapters alongside the base model. Enable it on the profile:

```yaml
# profiles.yaml
profiles:
  - name: qwen3-8b-lora
    backend: vllm
    config_name: qwen3-8b
    model_id: Qwen/Qwen3-8B
    enable_lora: true
    max_loras: 4
    max_lora_rank: 64
    lora_modules: "sql-adapter=/app/lora/sql,chat-adapter=/app/lora/chat"
```

When `enable_lora` is `true`, llmux:

- Requires `LORA_BASE_PATH` to be set and absolute in `.env.common`.
- Layers `compose/vllm/docker-compose.lora.yaml`, which bind-mounts
  `${LORA_BASE_PATH}` into the container as `/app/lora:ro`.
- Builds the `${LORA_OPTIONS}` string appended to the `vllm serve` command:
  `--enable-lora`, plus `--max-loras`, `--max-lora-rank`, and `--lora-modules`
  when those fields are set. Commas in `lora_modules` are split into separate
  arguments.

!!! warning "Adapters must live under `LORA_BASE_PATH`"
    `lora_modules` paths are resolved inside the container, where
    `LORA_BASE_PATH` is mounted at `/app/lora`. Point each adapter at
    `/app/lora/<subdir>`.

## Extra pip packages and env overrides

Two profile fields let you customize the container without rebuilding an image:

- **`extra_pip_packages`** — extra packages installed into the container at
  start (surfaced as `EXTRA_PIP_PACKAGES`). The override Compose file runs an
  entrypoint wrapper that installs them before `vllm serve` launches. Example:
  `extra_pip_packages: flash-attn==2.7.4.post1`.
- **`env_vars`** — arbitrary environment variables merged into the container
  environment. Useful for vLLM feature flags, e.g.
  `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8: "1"`.

```yaml
# profiles.yaml
profiles:
  - name: gpt-oss-120b
    backend: vllm
    config_name: gpt-oss-120b
    model_id: openai/gpt-oss-120b
    extra_pip_packages: flash-attn==2.7.4.post1
    env_vars:
      VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8: "1"
```

## Dev builds

A dev build produces a local `vllm-dev:<tag>` image from a vLLM git checkout —
useful for testing an unreleased branch or a fork. Trigger it from the start
dialog ("Dev Build") or with `llmux image build-dev`.

The pipeline (shared with llama.cpp, in `tui/common/dev_build.py`):

1. **Clone or update** `.vllm-src/` from `VLLM_REPO_URL` (default
   `https://github.com/vllm-project/vllm.git`) at `VLLM_BRANCH` (default
   `main`). Both are overridable in `.env.common`. If the existing checkout's
   `origin` URL differs from the requested repo, llmux refuses to switch
   remotes silently — move or delete `.vllm-src/` yourself to swap.
2. **Detect GPU arch** — `nvidia-smi` is queried for compute capability and
   passed as `--build-arg torch_cuda_arch_list=...` so the build only targets
   your GPU (much faster than the multi-arch default). A DeepEP stage in the
   upstream Dockerfile that hard-codes architectures is patched to respect the
   detected value.
3. **`docker build`** — `docker/Dockerfile`, target `vllm-openai`, tagged both
   `vllm-dev:<tag>` and `vllm-dev:<branch>`. The image is labeled with the
   repo URL, branch, and commit hash so llmux can later tell whether a cached
   tag still matches the requested source.

When you start with a dev tag, llmux rebuilds automatically if the image is
missing **or** if its labels show it was built from a different repo/branch.
Passing an explicit `--tag` that does not exist is treated as an error (with a
list of available `vllm-dev` tags) rather than a silent rebuild.

!!! note "GPU required for the fast path"
    The arch-detection step needs a working `nvidia-smi`. On a host without it,
    use the official-Dockerfile path (builds all architectures, slower) — the
    dev-build flow falls back accordingly.

## See also

- [llama.cpp backend](llamacpp.md) — the GGUF counterpart
- [Backend comparison](comparison.md) — feature matrix and "when to use which"
- [Troubleshooting](../troubleshooting.md) — OOM, stuck health checks, image
  resolution failures
