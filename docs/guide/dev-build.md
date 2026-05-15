# Dev Builds

By default llmux runs the **official** prebuilt images — `vllm/vllm-openai` from
Docker Hub and `ghcr.io/ggml-org/llama.cpp:server-cuda`. A **dev build** instead
compiles the engine **from source** into a local image, so you can run a fork, a
PR branch, or an unreleased commit.

The result is a local image tagged `<backend>-dev:<tag>`:

- `vllm-dev:<tag>`
- `llamacpp-dev:<tag>`

You then [pin a profile](#pinning-a-profile-to-a-dev-image) to that image, or
select it once from the [TUI start screen](tui.md#container-start-screen).

## When to use a dev build

| Use the official image | Use a dev build |
| --- | --- |
| Running a released model on a stable engine | Testing an upstream PR or your own fork |
| You want the fastest path to serving | You need a feature not in any release yet |
| — | You need engine patches (e.g. custom CUDA archs) |

## Building from source

The unified builder lives behind one command:

```bash
llmux image build-dev --backend vllm
llmux image build-dev --backend llamacpp
```

It runs the same three-stage pipeline for both backends:

1. **git clone / update** the source checkout (`.vllm-src/` or `.llamacpp-src/`)
   — clone if missing, otherwise `fetch` + `checkout` + `pull`.
2. **`docker build`** with the backend's Dockerfile and target stage.
3. **Label** the image with `repo.url`, `repo.branch`, `commit.hash`, and
   `build.date` so llmux can later tell which source a tag was built from.

Docker output streams live to your terminal.

!!! info "The source checkout is sticky"
    The builder refuses to silently switch the `.vllm-src/` / `.llamacpp-src/`
    checkout to a different remote URL. If you want to swap repos, move or
    delete the checkout directory yourself first.

### `build-dev` options

| Option | Applies to | Description |
| --- | --- | --- |
| `--backend` | both | `vllm` or `llamacpp`. Required. |
| `--branch`, `-b` | both | Source branch. Default from `.env.common` (`VLLM_BRANCH` / `LLAMACPP_BRANCH`). |
| `--repo-url` | both | Source repo URL. Default from `.env.common` (`VLLM_REPO_URL` / `LLAMACPP_REPO_URL`). |
| `--tag`, `-t` | both | Custom output tag. Default = the branch name. |
| `--official` | vLLM only | Build with upstream Dockerfile defaults — skips llmux's local GPU-arch detection patch. Builds for **all** architectures (portable, slower). |
| `--cuda-arch` | llama.cpp only | Override GPU auto-detection. CMake format, e.g. `89` (Ada) or `86;89` (mixed). |
| `--multi-arch` | llama.cpp only | Disable GPU auto-detection; build for all archs (portable, slow). |

### Output tags

Every build applies **two** tags:

- `<prefix>:<tag>` — the unique tag (your `--tag`, or `<branch>-YYYYMMDD`)
- `<prefix>:<branch>` — a stable alias always pointing at the latest build of
  that branch

So `llmux image build-dev --backend llamacpp --branch master` produces both
`llamacpp-dev:master-20260515` and `llamacpp-dev:master`.

## GPU architecture auto-detection

By default the builder compiles **only for your local GPU(s)**, which is
dramatically faster than an all-architecture build. It runs `nvidia-smi` to read
each GPU's compute capability and passes it to the build:

=== "vLLM"

    Detected caps are passed as `torch_cuda_arch_list` (PyTorch's dotted,
    space-separated format, e.g. `8.6 8.9`). The builder also patches the
    upstream Dockerfile's DeepEP stage so it respects the local arch list.
    `--official` skips both — useful if detection fails or you need a portable
    image. If `nvidia-smi` is unavailable and you didn't pass `--official`, the
    build aborts with an error.

=== "llama.cpp"

    Detected caps are passed as `CUDA_DOCKER_ARCH` (CMake's no-dot,
    semicolon-separated format, e.g. `86;89`). If `nvidia-smi` is unavailable,
    the builder falls back to a multi-arch build automatically. Override
    detection with `--cuda-arch '89'` or force the portable path with
    `--multi-arch`.

## Pinning a profile to a dev image

Once the image exists, point a profile at it via the
[`image_tag`](profiles.md#per-profile-image_tag) field in `profiles.yaml`:

```yaml
- name: gemma-3-4b-mtp
  backend: llamacpp
  config_name: gemma-3-4b
  image_tag: llamacpp-dev:master   # use the dev image instead of the default
  hf_repo: unsloth/gemma-3-4b-it-GGUF
  hf_file: gemma-3-4b-it-q4_k_m.gguf
```

Or set it from the CLI / [TUI profile form](tui.md#profile-and-config-forms):

```bash
llmux profile edit gemma-3-4b-mtp --image-tag llamacpp-dev:master
```

`image_tag` is a **per-profile** override — one profile can run a dev image
while another stays on the official image. For llama.cpp, when `image_tag`
starts with `llamacpp-dev:`, llmux layers `docker-compose.dev.yaml` into the
compose invocation to swap the `image:` line.

If you `up` a profile pinned to a dev image that doesn't exist locally, llmux
tells you to build it first rather than failing obscurely.

## Dev Build from the TUI

The [Container Start screen](tui.md#container-start-screen) has a **Dev Build**
mode in its Version radio set (for both backends). Selecting it reveals
**Repo URL** and **Branch** inputs, pre-filled from `.env.common`. On **Start**,
llmux:

1. Checks whether `<backend>-dev:<branch>` already exists locally **and** was
   built from the same repo/branch (verified via the image labels).
2. If missing — or if the cached tag was built from a *different* repo/branch —
   it rebuilds from source, streaming the build log.
3. Brings the container up on the freshly built image.

This is a launch-time override: it doesn't modify `profiles.yaml`. To make it
permanent, set `image_tag` on the profile.

!!! tip "Tag-name collisions are handled"
    `llamacpp-dev:master` built from `ggml-org/llama.cpp` and `llamacpp-dev:master`
    built from a fork would collide on tag name alone. llmux stamps every dev
    image with its source repo/branch labels and rebuilds rather than silently
    reusing the wrong source.

## Repo / branch defaults

Dev builds resolve their source from `.env.common` when you don't pass
`--repo-url` / `--branch` explicitly:

| Backend | Repo URL key | Branch key | Default branch |
| --- | --- | --- | --- |
| vLLM | `VLLM_REPO_URL` | `VLLM_BRANCH` | `main` |
| llama.cpp | `LLAMACPP_REPO_URL` | `LLAMACPP_BRANCH` | `master` |

If a key is unset, llmux falls back to the upstream repository and that
backend's default branch.

## See also

- [Profiles](profiles.md) — the `image_tag` field that pins a dev image
- [Container Lifecycle](container-lifecycle.md) — the `up --dev` launch path
- [Using the TUI](tui.md) — the Dev Build start mode
- [.env.common](../reference/env-common.md) — repo/branch defaults
