# Installation

This page walks through everything you need to get `llmux` installed and
configured: system requirements, the install command itself, and the three
one-time setup steps (`.env.common`, `profiles.yaml`, and a sanity check).

## Requirements

llmux orchestrates GPU containers, so it expects a Linux host with NVIDIA
hardware and a working Docker + CUDA stack.

| Requirement | Notes |
|---|---|
| **Linux + NVIDIA GPU** | One or more NVIDIA GPUs with a recent driver installed (`nvidia-smi` should work). |
| **Docker + NVIDIA Container Toolkit** | Docker Engine with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so containers can see the GPU. |
| **Python 3.10+** | Managed through [uv](https://docs.astral.sh/uv/) — uv installs and isolates the tool for you. |
| **Hugging Face CLI (`hf`)** | Only needed for llama.cpp GGUF auto-download: `uv tool install huggingface_hub`. |

!!! note "Why Docker?"
    llmux never installs vLLM or llama.cpp on your host. Every profile runs in
    an official container image, and llmux only renders the Compose files and
    streams logs. That keeps the two toolchains fully isolated from each other
    and from your system Python.

!!! warning "GPU passthrough must already work"
    Before installing llmux, confirm `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
    prints your GPUs. If it doesn't, fix the
    [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
    setup first — llmux can't work around a broken GPU runtime.

## Install llmux

llmux is installed as a [uv tool](https://docs.astral.sh/uv/concepts/tools/) in
editable mode. This puts the `llmux` binary on your `PATH` while keeping it
pinned to your local checkout, so any code edits are picked up live.

```bash
git clone https://github.com/Bae-ChangHyun/llmux.git
cd llmux

# Install the `llmux` command globally (editable — local edits apply immediately)
uv tool install --editable .

# One-time: add ~/.local/bin to your PATH, then re-source your shell
uv tool update-shell
```

Once installed, the `llmux` binary lives in `~/.local/bin/llmux` and runs from
any directory:

```bash
llmux --help
```

??? tip "Upgrading and uninstalling"
    - **Upgrade:** pull the latest commit and reinstall —
      `git pull && uv tool install --editable . --reinstall`
    - **Uninstall:** `uv tool uninstall llmux`

??? tip "Run without installing globally"
    From inside the repo you can also run it ad-hoc with `uv run llmux`.
    llmux resolves `profiles.yaml`, `compose/`, `config/`, `scripts/`, and
    `.env.common` relative to the project root — so if you run it from
    **outside** the repo, set `LLMUX_ROOT=/path/to/llmux` so it can find them.

## Initial setup

llmux needs two files before it can start a model: `.env.common` (shared host
paths and tokens) and `profiles.yaml` (your model definitions). Both are
gitignored — they're yours to edit freely.

### 1. Create and edit `.env.common`

`.env.common` holds host-side settings shared by every profile: where to cache
Hugging Face downloads, where GGUF files live, the llama.cpp image tag, and so
on. Copy the template and edit the values:

```bash
cp .env.common.example .env.common
$EDITOR .env.common
```

The variables you can set:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face access token for gated model downloads. Leave empty if you only use public repos. |
| `HF_CACHE_PATH` | Host path mounted into containers as the Hugging Face cache. **Must be an absolute path.** |
| `MODEL_DIR` | Host directory where llama.cpp GGUF files are stored (mounted into the container as `/models`). |
| `LORA_BASE_PATH` | vLLM only: host path holding LoRA adapters (mounted into the container as `/app/lora`). |
| `LLAMACPP_IMAGE` | llama.cpp server image tag — defaults to `ghcr.io/ggml-org/llama.cpp:server-cuda`. |
| `LLAMACPP_REPO_URL` | Source repo URL used by `llmux image build-dev` for llama.cpp dev builds. Defaults to the upstream `ggml-org/llama.cpp` repo. |
| `LLAMACPP_BRANCH` | Branch cloned for llama.cpp dev builds. Defaults to `master`. |
| `TZ` | Container timezone, e.g. `Asia/Seoul`. |

!!! warning "`HF_CACHE_PATH` must be absolute"
    `env-check` rejects a relative `HF_CACHE_PATH`. Use a full path such as
    `/home/$USER/.cache/huggingface` so the bind mount resolves correctly.

!!! note "Gated models need `HF_TOKEN`"
    Public models download fine with an empty `HF_TOKEN`. For gated repos
    (some Llama and Gemma weights, for example), set a token with access — and
    for llama.cpp's GGUF auto-download, make sure the `hf` CLI is installed too.

### 2. Prepare `profiles.yaml`

`profiles.yaml` is the single source of truth for your model profiles across
both backends. Start from the bundled template:

```bash
cp profiles.example.yaml profiles.yaml
$EDITOR profiles.yaml
```

The template defines a `defaults` block per backend plus one example profile
for each engine:

```yaml
version: 1

defaults:
  vllm:
    port: 8000
    gpu_id: "0"
    tensor_parallel_size: 1
    enable_lora: false
  llamacpp:
    port: 8080
    gpu_id: "0"

profiles:
  - name: qwen3-0-6b
    backend: vllm
    config_name: qwen3-0-6b       # → config/vllm/qwen3-0-6b.yaml
    model_id: Qwen/Qwen3-0.6B
    gpu_id: "0"

  - name: gemma-3-4b
    backend: llamacpp
    config_name: gemma-3-4b       # → config/llamacpp/gemma-3-4b.yaml
    model_file: gemma-3-4b-it-q4_k_m.gguf
    hf_repo: unsloth/gemma-3-4b-it-GGUF
    hf_file: gemma-3-4b-it-q4_k_m.gguf
```

Any field you omit from a profile inherits from `defaults.<backend>`. You don't
need to hand-write the example profiles — the [Quick Start](quickstart.md)
generates a fresh profile for you with `llmux profile quick-setup`.

!!! note "You never hand-write `.env` files"
    At launch, each selected profile is rendered to
    `.runtime/<backend>/<name>.env` and passed to `docker compose --env-file`.
    `profiles.yaml` is the whole interface — the runtime renders are gitignored.

See the [Profiles guide](../guide/profiles.md) for the full schema.

### 3. Verify the installation

`llmux env-check` validates `.env.common` and reports the resolved project
paths. Run it from inside the repo (or with `LLMUX_ROOT` set):

```bash
llmux env-check
```

A healthy setup ends with `Status       : OK`. If something is missing, the
command lists each issue and exits non-zero — for example a missing
`.env.common`, an unset `HF_TOKEN`, or a relative `HF_CACHE_PATH`.

You can also confirm the GPU runtime is visible to llmux:

```bash
llmux gpu          # nvidia-smi summary, one row per GPU
```

!!! tip "`--json` everywhere"
    `env-check`, `gpu`, and every `list`/`show` command accept `--json` for
    clean piping into scripts and agents.

## Next steps

With llmux installed and configured, head to the
[Quick Start](quickstart.md) to create your first profile and serve a model in
about five minutes.
