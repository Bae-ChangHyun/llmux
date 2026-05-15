# `.env.common` Reference

`.env.common` holds the **host-wide** settings shared by every profile across
both backends — Hugging Face credentials, cache and model paths, dev-build
sources, and the timezone. It lives at the repo root and is **gitignored**;
create it once by copying the template:

```bash
cp .env.common.example .env.common
# then edit the values
```

llmux loads this file at launch time and merges it into the Compose
environment for both backends. Per-model settings (sampling flags, context
length, GPU ids, …) belong in `profiles.yaml` and `config/<backend>/<name>.yaml`,
**not** here.

!!! tip "Validate it"
    Run [`llmux env-check`](cli.md#system-env-check) to confirm the file
    exists, that `HF_TOKEN` / `HF_CACHE_PATH` / `VLLM_VERSION` are set, and
    that `HF_CACHE_PATH` is an absolute path.

!!! note "Shell expansion"
    `$USER` in the template is expanded by your shell when Docker Compose
    reads the file, so paths like `/home/$USER/.cache/huggingface` resolve to
    your home directory. Keep paths **absolute**.

## All variables at a glance

| Variable | Required | Default (template) | Backend | Purpose |
|---|---|---|---|---|
| `HF_TOKEN` | Optional¹ | _(empty)_ | both | Hugging Face token for gated/private model downloads. |
| `HF_CACHE_PATH` | **Yes** | `/home/$USER/.cache/huggingface` | both | Host path mounted as the in-container HF cache. |
| `LORA_BASE_PATH` | Optional² | _(empty)_ | vLLM | Host directory of LoRA adapters, mounted at `/app/lora`. |
| `VLLM_VERSION` | Conditional³ | _(not in template)_ | vLLM | Tag for the official `vllm/vllm-openai` image. |
| `VLLM_REPO_URL` | Optional | `https://github.com/vllm-project/vllm.git` | vLLM | Source repo for `image build-dev --backend vllm`. |
| `VLLM_BRANCH` | Optional | `main` | vLLM | Default source branch for vLLM dev builds. |
| `MODEL_DIR` | Optional⁴ | `/home/$USER/Project/docker/llmux/models` | llama.cpp | Host GGUF directory (mounted at `/models`). |
| `LLAMACPP_IMAGE` | Optional | `ghcr.io/ggml-org/llama.cpp:server-cuda` | llama.cpp | Official llama.cpp server image tag. |
| `LLAMACPP_REPO_URL` | Optional | `https://github.com/ggml-org/llama.cpp.git` | llama.cpp | Source repo for `image build-dev --backend llamacpp`. |
| `LLAMACPP_BRANCH` | Optional | `master` | llama.cpp | Default source branch for llama.cpp dev builds. |
| `TZ` | Optional | `Asia/Seoul` | both | Container timezone. |

¹ Required only for gated/private Hugging Face repos — public models download
without it.
² Required only when a vLLM profile has `enable_lora: true`.
³ `vllm container up` resolves a tag automatically from local images when
`--tag` is empty; `env-check` still flags `VLLM_VERSION` as unset because it is
the variable the official `docker-compose.yaml` substitutes.
⁴ Reserved for the host GGUF directory; current llama.cpp profiles download
GGUF files directly into the HF cache via `llama-server -hf` (see below).

---

## Hugging Face

### `HF_TOKEN`

Hugging Face access token. Used to download **gated or private** models on both
backends — vLLM passes it as `HUGGING_FACE_HUB_TOKEN`, llama.cpp as `HF_TOKEN`.
If you only ever serve public repositories you can leave it empty.

```ini
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The token is also read by [`llmux system mem-estimate`](cli.md#system-mem-estimate)
to size gated models, and is masked in `env-check` output.

### `HF_CACHE_PATH`

**Required.** Absolute host path mounted into every container as the Hugging
Face cache (`/root/.cache/huggingface`). Both backends download model weights
here, so the cache is shared and persistent across container restarts and
across the two backends.

```ini
HF_CACHE_PATH=/home/$USER/.cache/huggingface
```

!!! warning "Must be absolute"
    `env-check` rejects a relative `HF_CACHE_PATH`. Point it at a disk with
    enough room for the largest models you plan to serve.

---

## vLLM

### `LORA_BASE_PATH`

Host directory containing LoRA adapter folders. When set, it is mounted
read-only at `/app/lora` inside the vLLM container (via
`compose/vllm/docker-compose.lora.yaml`), so a profile's `lora_modules` entries
can reference `/app/lora/<adapter>`. Only consumed when the profile has
`enable_lora: true`.

```ini
LORA_BASE_PATH=/home/$USER/lora-adapters
```

### `VLLM_VERSION`

Tag of the official `vllm/vllm-openai` image used by
`compose/vllm/docker-compose.yaml` (`image: vllm/vllm-openai:${VLLM_VERSION}`).
In practice `llmux container up` selects the highest local versioned tag (or
the tag you pass with `--tag`) and injects it for you — but setting
`VLLM_VERSION` gives `env-check` and a bare `docker compose` invocation a
known-good default.

```ini
VLLM_VERSION=v0.20.1
```

### `VLLM_REPO_URL` / `VLLM_BRANCH`

Defaults for the vLLM **dev-build** pipeline
([`llmux image build-dev --backend vllm`](cli.md#image-build-dev)). When
`--repo-url` / `--branch` are omitted on the CLI, these values are used; if
they are also unset, the pipeline falls back to
`https://github.com/vllm-project/vllm.git` and `main`.

```ini
VLLM_REPO_URL=https://github.com/vllm-project/vllm.git
VLLM_BRANCH=main
```

---

## llama.cpp

### `MODEL_DIR`

Host directory intended for GGUF files (mounted at `/models` in the container).

```ini
MODEL_DIR=/home/$USER/Project/docker/llmux/models
```

!!! note "How llama.cpp gets its models"
    Current llama.cpp profiles do **not** rely on a pre-populated `MODEL_DIR`.
    `render-override.py` builds a `llama-server -hf <repo> -hff <file>`
    command from the profile's `hf_repo` / `hf_file`, so `llama-server`
    downloads the GGUF straight into the shared `HF_CACHE_PATH` — exactly like
    the vLLM flow, with no host-side `hf` CLI dependency.

### `LLAMACPP_IMAGE`

Tag of the official llama.cpp server image used by
`compose/llamacpp/docker-compose.yaml`
(`image: ${LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}`). Override
it to pin a specific upstream build.

```ini
LLAMACPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
```

To run a profile on a **locally-built** image instead, build with
`llmux image build-dev --backend llamacpp` and set `image_tag:
llamacpp-dev:<branch>` on that profile in `profiles.yaml` — that is a
per-profile override and does not touch `LLAMACPP_IMAGE`.

### `LLAMACPP_REPO_URL` / `LLAMACPP_BRANCH`

Defaults for the llama.cpp **dev-build** pipeline
([`llmux image build-dev --backend llamacpp`](cli.md#image-build-dev)). When
`--repo-url` / `--branch` are omitted, these are used; if also unset, the
pipeline falls back to `https://github.com/ggml-org/llama.cpp.git` and
`master`.

```ini
LLAMACPP_REPO_URL=https://github.com/ggml-org/llama.cpp.git
LLAMACPP_BRANCH=master
```

---

## Common

### `TZ`

Timezone applied to both backend containers (`TZ=${TZ:-UTC}` in both Compose
files). Affects log timestamps inside the container. Defaults to `UTC` if
unset.

```ini
TZ=Asia/Seoul
```

---

## Template

The shipped `.env.common.example` you copy from:

```ini
# llmux 공통 설정 템플릿
# cp .env.common.example .env.common 후 값 수정

# ─── HuggingFace ───────────────────────────────────────────────
# 게이트 모델 다운로드에 사용 (공개 repo 만 쓰면 비워도 됨)
HF_TOKEN=
HF_CACHE_PATH=/home/$USER/.cache/huggingface

# ─── vLLM 전용 ─────────────────────────────────────────────────
# LoRA adapter 경로 (container 안 /app/lora 로 mount)
LORA_BASE_PATH=

# ─── llama.cpp 전용 ────────────────────────────────────────────
# GGUF 저장 경로 (container 안 /models 로 mount)
MODEL_DIR=/home/$USER/Project/docker/llmux/models

# llama.cpp 이미지 태그
LLAMACPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda

# llama.cpp dev-build (per-fork / PR branch). 비워두면 ggml-org/master 사용.
LLAMACPP_REPO_URL=https://github.com/ggml-org/llama.cpp.git
LLAMACPP_BRANCH=master

# 타임존
TZ=Asia/Seoul
```

!!! info "Not in the template, still honored"
    `VLLM_VERSION`, `VLLM_REPO_URL`, and `VLLM_BRANCH` are not in the shipped
    template but are read from `.env.common` when present (consumed by
    `compose/vllm/docker-compose.yaml` and the vLLM dev-build defaults). Add
    them if you want a fixed vLLM image tag or a custom dev-build source.

For where these values flow at launch time, see
[Architecture](architecture.md). For the per-command flags that override
these defaults, see the [CLI Reference](cli.md).
